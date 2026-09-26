"""把解析事件组装成 TTS 段落。

逐句请求会让句间语调不连贯，所以（Spec「风险」一节）：

- 首句单独合成，保证首包延迟；
- 后续每 2–3 句合并成一次请求；
- 独白整段合成。

舞台指令（位置 / 音效 / style 变化）一律作为段落边界：它们要在某一句开始播放时生效，
而一次 TTS 请求只能带一个 style。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tags import Cue, Event, Position, Sentence, StyleChange, strip_vocal_tags, visible_len


@dataclass
class Segment:
    text: str  # TTS 文本，可为空（只有舞台指令的段落）
    style: str
    cues: list[Cue] = field(default_factory=list)
    style_directive: str | None = None  # 段首出现的 [style: ...]，写回历史用
    position: Position = field(default_factory=Position)  # 应用 cues 之后的位置

    @property
    def subtitle(self) -> str:
        return strip_vocal_tags(self.text)

    def render(self) -> str:
        """还原为协议文本（写回对话历史）。"""
        parts = [cue.render() for cue in self.cues]
        if self.style_directive is not None:
            parts.append(f"[style: {self.style_directive}]")
        head = " ".join(parts)
        if head and self.text:
            return f"{head}\n{self.text}"
        return head or self.text


class Segmenter:
    def __init__(
        self,
        *,
        position: Position,
        whole: bool = False,
        first_alone: bool = True,
        group_size: int = 3,
        max_chars: int = 120,
    ):
        self._whole = whole
        self._first_alone = first_alone and not whole
        self._group_size = group_size
        self._max_chars = max_chars
        self._position = position
        self._sentences: list[Sentence] = []
        self._cues: list[Cue] = []
        self._style_directive: str | None = None
        self._emitted_text = False

    def push(self, event: Event) -> list[Segment]:
        out: list[Segment] = []
        if isinstance(event, Cue):
            out += self._emit()
            self._cues.append(event)
            if event.kind in ("pos", "move"):
                self._position = Position(event.azimuth, event.distance)
        elif isinstance(event, StyleChange):
            out += self._emit()
            self._style_directive = event.style
        elif isinstance(event, Sentence):
            self._sentences.append(event)
            text_len = sum(visible_len(s.text) for s in self._sentences)
            if (
                (self._first_alone and not self._emitted_text)
                or (not self._whole and len(self._sentences) >= self._group_size)
                or text_len >= self._max_chars
            ):
                out += self._emit()
        return out

    def flush(self) -> list[Segment]:
        out = self._emit()
        if self._cues or self._style_directive is not None:
            out.append(self._make("", ""))
        return out

    def _emit(self) -> list[Segment]:
        if not self._sentences:
            return []
        style = self._sentences[0].style
        text = " ".join(s.text for s in self._sentences)
        self._sentences = []
        self._emitted_text = True
        return [self._make(text, style)]

    def _make(self, text: str, style: str) -> Segment:
        seg = Segment(
            text=text,
            style=style,
            cues=self._cues,
            style_directive=self._style_directive,
            position=self._position,
        )
        self._cues = []
        self._style_directive = None
        return seg
