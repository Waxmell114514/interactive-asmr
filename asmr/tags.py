"""LLM 输出协议解析。

LLM 输出普通文本加两类标记：

- 方括号 ``[key: value]`` 是舞台指令（pos / move / style / sfx），由解析器拆掉；
- 尖括号 ``<sigh>`` 等是 Gemini TTS 原生声音事件，白名单内的原样留在台词里。

解析是流式的：``feed()`` 接收任意切分的文本增量，返回已经确定的事件；
``flush()`` 在流结束时吐出剩余内容。规则（见 Spec「LLM 输出协议」）：

1. 指令持续有效，直到被下一条同类指令覆盖（解析器持有当前 style / 位置）。
2. 按 ``。！？…～`` 或满 40 字切句；换行和舞台指令也会切句。
3. 未知指令和未列出的尖括号标记一律丢弃并记日志，绝不进入 TTS 文本。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Iterable, Union

log = logging.getLogger(__name__)

# Gemini 3.8 TTS 文档「Vocal bursts and non-speech sounds」列出的标记。
VOCAL_TAGS = frozenset(
    {
        "argh", "breath", "heavy breath", "exhales", "cackle", "cheer", "chuckle",
        "chuckles", "cough", "cry", "gasp", "giggle", "groan", "growl", "grunt",
        "grr", "hiss", "laugh", "laughter", "moan", "pant", "pff", "phew",
        "scream", "shout", "shriek", "sigh", "sighs", "sneeze", "snicker", "snort",
        "sob", "throat-clearing", "tsk", "whimper", "whispers", "whispering",
        "yawn", "short pause", "long pause",
    }
)

DEFAULT_SFX = ("page_turn", "tea_pour", "rain_up", "rain_down", "fabric_rustle")

AZIMUTHS = {
    "left": "left", "左": "left", "左边": "left", "左耳": "left",
    "right": "right", "右": "right", "右边": "right", "右耳": "right",
    "front": "front", "前": "front", "前面": "front", "面前": "front",
    "behind": "behind", "back": "behind", "后": "behind", "后面": "behind", "身后": "behind",
}
DISTANCES = {
    "near": "near", "近": "near", "贴近": "near", "耳边": "near",
    "mid": "mid", "middle": "mid", "中": "mid",
    "far": "far", "远": "far", "远处": "far",
}
DIRECTIVE_KEYS = {
    "pos": "pos", "position": "pos", "位置": "pos",
    "move": "move", "移动": "move",
    "style": "style", "语气": "style",
    "sfx": "sfx", "音效": "sfx",
}

TERMINATORS = set("。！？…～!?~")
CLOSERS = set("」』”’）)\"'】")
SOFT_BREAKS = set("，,、；;：: ")
OPEN_DIRECTIVE = {"[": "]", "［": "］", "【": "】"}
OPEN_VTAG = {"<": ">", "＜": "＞"}

MAX_STYLE_CHARS = 20
MAX_DIRECTIVE_CHARS = 80
MAX_VTAG_CHARS = 32
DEFAULT_MOVE_SECS = 2.0

_TAG_RE = re.compile(r"<[^<>]*>")
_WORD_RE = re.compile(r"[\w一-鿿]")
_SPLIT_RE = re.compile(r"[,，、\s]+")
_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|sec|secs|秒)?$")


@dataclass(frozen=True)
class Position:
    azimuth: str = "front"
    distance: str = "mid"


@dataclass(frozen=True)
class Cue:
    """发往前端的舞台事件，在所属段落开始播放的时刻生效。"""

    kind: str  # "pos" | "move" | "sfx"
    azimuth: str | None = None
    distance: str | None = None
    duration: float | None = None
    name: str | None = None

    def to_json(self) -> dict:
        d: dict = {"kind": self.kind}
        for key in ("azimuth", "distance", "duration", "name"):
            value = getattr(self, key)
            if value is not None:
                d[key] = value
        return d

    def render(self) -> str:
        """还原成协议文本，写回对话历史，让 LLM 看到自己一致的输出格式。"""
        if self.kind == "pos":
            return f"[pos: {self.azimuth}, {self.distance}]"
        if self.kind == "move":
            return f"[move: {self.azimuth}, {self.distance}, {self.duration:g}s]"
        return f"[sfx: {self.name}]"


@dataclass(frozen=True)
class StyleChange:
    style: str

    def render(self) -> str:
        return f"[style: {self.style}]"


@dataclass(frozen=True)
class Sentence:
    text: str  # 含白名单声音事件的 TTS 文本
    style: str

    @property
    def plain(self) -> str:
        return strip_vocal_tags(self.text)


Event = Union[Cue, StyleChange, Sentence]


def strip_vocal_tags(text: str) -> str:
    """去掉尖括号标记，用于字幕显示。"""
    return re.sub(r"\s{2,}", " ", _TAG_RE.sub("", text)).strip()


def visible_len(text: str) -> int:
    return len(_TAG_RE.sub("", text))


def _has_content(text: str) -> bool:
    return bool(_WORD_RE.search(_TAG_RE.sub("", text))) or bool(_TAG_RE.search(text))


class TagParser:
    """流式解析器。每个回复新建一个，初始 style / 位置取自会话当前状态。"""

    def __init__(
        self,
        *,
        style: str = "",
        position: Position | None = None,
        sfx_whitelist: Iterable[str] = DEFAULT_SFX,
        max_chars: int = 40,
    ):
        self.style = style
        self.position = position or Position()
        self._sfx = frozenset(sfx_whitelist)
        self._max_chars = max_chars
        self._sent = ""  # 当前句（含声音事件标记）
        self._pending_end = False  # 已遇到句末标点，等下一个非标点字符再切
        self._tag_close: str | None = None  # 正在读取的括号类型对应的右括号
        self._tag_kind = ""  # "directive" | "vtag"
        self._tag_buf = ""
        self._out: list[Event] = []

    # ---- 公共接口 ---------------------------------------------------------

    def feed(self, chunk: str) -> list[Event]:
        for ch in chunk:
            self._feed_char(ch)
        return self._drain()

    def flush(self) -> list[Event]:
        if self._tag_close is not None:
            log.warning("丢弃未闭合的标记: %r", self._tag_buf)
            self._tag_close = None
            self._tag_buf = ""
        self._emit_sentence()
        return self._drain()

    # ---- 内部 -------------------------------------------------------------

    def _drain(self) -> list[Event]:
        out, self._out = self._out, []
        return out

    def _feed_char(self, ch: str) -> None:
        if self._tag_close is not None:
            self._feed_tag_char(ch)
            return

        if ch in OPEN_DIRECTIVE or ch in OPEN_VTAG:
            if self._pending_end:
                self._emit_sentence()
            self._tag_kind = "directive" if ch in OPEN_DIRECTIVE else "vtag"
            self._tag_close = OPEN_DIRECTIVE.get(ch) or OPEN_VTAG[ch]
            self._tag_buf = ""
            return

        if ch in "\r\n":
            self._emit_sentence()
            return

        if self._pending_end and ch not in TERMINATORS and ch not in CLOSERS:
            self._emit_sentence()

        self._sent += ch
        if ch in TERMINATORS:
            self._pending_end = True
        elif not self._pending_end and visible_len(self._sent) >= self._max_chars:
            self._split_long()

    def _feed_tag_char(self, ch: str) -> None:
        limit = MAX_DIRECTIVE_CHARS if self._tag_kind == "directive" else MAX_VTAG_CHARS
        if ch == self._tag_close:
            body, kind = self._tag_buf, self._tag_kind
            self._tag_close = None
            self._tag_buf = ""
            if kind == "directive":
                self._handle_directive(body)
            else:
                self._handle_vtag(body)
            return
        if ch in "\r\n" or len(self._tag_buf) >= limit:
            log.warning("丢弃异常标记: %r", self._tag_buf + ch)
            self._tag_close = None
            self._tag_buf = ""
            if ch in "\r\n":
                self._emit_sentence()
            return
        self._tag_buf += ch

    def _handle_vtag(self, body: str) -> None:
        name = " ".join(body.strip().lower().split())
        if name in VOCAL_TAGS:
            if self._sent and not self._sent.endswith(" "):
                self._sent += " "
            self._sent += f"<{name}> "
        else:
            log.warning("丢弃未列出的声音标记: <%s>", body)

    def _handle_directive(self, body: str) -> None:
        key, sep, value = body.replace("：", ":").partition(":")
        key = DIRECTIVE_KEYS.get(key.strip().lower())
        value = value.strip()
        if not sep or key is None:
            log.warning("丢弃未知指令: [%s]", body)
            return
        # 舞台指令同时是切句点：之前的台词先成句，指令作用于之后的台词。
        self._emit_sentence()
        if key == "style":
            if len(value) > MAX_STYLE_CHARS:
                log.warning("style 超过 %d 字，已截断: %s", MAX_STYLE_CHARS, value)
                value = value[:MAX_STYLE_CHARS]
            self.style = value
            self._out.append(StyleChange(value))
        elif key == "pos":
            pos = self._parse_position(value, allow_duration=False)
            if pos is not None:
                self.position = pos[0]
                self._out.append(Cue("pos", pos[0].azimuth, pos[0].distance))
        elif key == "move":
            pos = self._parse_position(value, allow_duration=True)
            if pos is not None:
                self.position = pos[0]
                self._out.append(Cue("move", pos[0].azimuth, pos[0].distance, duration=pos[1]))
        elif key == "sfx":
            name = value.lower()
            if name in self._sfx:
                self._out.append(Cue("sfx", name=name))
            else:
                log.warning("丢弃白名单外音效: %s", value)

    def _parse_position(self, value: str, *, allow_duration: bool) -> tuple[Position, float] | None:
        azimuth = distance = None
        duration = DEFAULT_MOVE_SECS
        for token in filter(None, _SPLIT_RE.split(value.lower())):
            if token in AZIMUTHS:
                azimuth = AZIMUTHS[token]
            elif token in DISTANCES:
                distance = DISTANCES[token]
            elif allow_duration and (m := _DURATION_RE.match(token)):
                duration = min(max(float(m.group(1)), 0.2), 30.0)
            else:
                log.warning("忽略无法识别的方位参数: %s", token)
        if azimuth is None and distance is None:
            log.warning("丢弃无效方位指令: %s", value)
            return None
        pos = replace(
            self.position,
            **({"azimuth": azimuth} if azimuth else {}),
            **({"distance": distance} if distance else {}),
        )
        return pos, duration

    def _split_long(self) -> None:
        """满 max_chars 字仍无句末标点：优先在最后一个逗号类停顿处切。"""
        text = self._sent
        cut = -1
        for i in range(len(text) - 1, 0, -1):
            if text[i] in SOFT_BREAKS and not _inside_tag(text, i):
                cut = i + 1
                break
        if cut <= 0 or visible_len(text[:cut]) < self._max_chars // 4:
            cut = len(text)
        head, self._sent = text[:cut], text[cut:]
        self._push_sentence(head)

    def _emit_sentence(self) -> None:
        text, self._sent = self._sent, ""
        self._pending_end = False
        self._push_sentence(text)

    def _push_sentence(self, text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if text and _has_content(text):
            self._out.append(Sentence(text, self.style))


def _inside_tag(text: str, index: int) -> bool:
    return text.rfind("<", 0, index) > text.rfind(">", 0, index)


def parse_all(text: str, **kwargs) -> list[Event]:
    """一次性解析整段文本（开场白、测试用）。"""
    parser = TagParser(**kwargs)
    return parser.feed(text) + parser.flush()
