"""角色卡：沿用酒馆（SillyTavern）Character Card V2 格式，支持 .json 与嵌入 JSON 的 .png。

ASMR 专用字段放在 ``data.extensions.asmr``（酒馆会原样保留未知扩展）::

    {"voice": "voice_xxx", "style": "温柔，很轻", "position": "front, near",
     "ambience": "rain", "user_name": "你"}
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .tags import Position, TagParser

PROTOCOL_PROMPT = """\
# 输出格式（必须遵守）
你的每一句话都会被合成为贴在对方耳边的声音。输出普通口语文本，可以夹带两类标记：

1. 方括号舞台指令，单独成行，写在它要生效的台词之前，持续有效直到被同类指令覆盖：
   - [pos: 方位, 距离]  方位 left / right / front / behind；距离 near（贴耳）/ mid / far
   - [move: 方位, 距离, 秒数]  慢慢移动到新位置，例如 [move: right, near, 3s]
   - [style: 语气]  不超过 20 字的语气描述，例如 [style: 耳语，很轻，带一点笑意]
   - [sfx: 音效名]  只能用：{sfx}
2. 尖括号声音事件，直接写在台词里：<sigh> <laugh> <chuckle> <breath> <yawn> <short pause> <long pause>

示例：
[pos: left, near] [style: 耳语，很轻，带一点笑意]
嗯……今天辛苦了。<sigh> 闭上眼睛，听我说就好。
[move: right, near, 3s]
我在这边哦。<short pause> 感觉到了吗？

# 说话方式
- 短句、口语、少用书面语；每轮回复 1–4 句。
- 不要用括号写动作描写，动作用舞台指令或声音事件表达。
- 对方安静时不要追问，不要问“你还在吗”“睡着了吗”。
- 不要使用 Markdown、表情符号、列表。
"""

MONOLOGUE_PROMPT = "（对方安静着，可能快睡着了。按你的节奏继续，{n} 句，不要提问。）"
WAKE_HINT = "（对方刚从浅睡中轻声开口。用最轻、最慢的声音回应。）"
INTRO_PROMPT = "（对方刚戴上耳机，环境音已经响起。轻轻地开场，1–2 句。）"


@dataclass
class Character:
    id: str
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    voice: str = ""
    gptsovits: dict = field(default_factory=dict)  # GPT-SoVITS 参考音频等配置，见 tts_gptsovits.py
    style: str = ""
    position: Position = field(default_factory=Position)
    ambience: str = "rain"
    user_name: str = "你"

    def render(self, text: str) -> str:
        return text.replace("{{char}}", self.name).replace("{{user}}", self.user_name)

    def build_system_prompt(self, sfx: tuple[str, ...]) -> str:
        parts = []
        if self.system_prompt:
            parts.append(self.render(self.system_prompt))
        else:
            parts.append(
                f"你是{self.name}，正在和{self.user_name}进行一场戴耳机的中文 ASMR 角色扮演。"
                "你的声音温和、贴耳，节奏很慢，让对方放松、入睡。"
            )
        for title, body in (
            ("角色", self.description),
            ("性格", self.personality),
            ("场景", self.scenario),
            ("对话示例", self.mes_example),
        ):
            if body:
                parts.append(f"# {title}\n{self.render(body)}")
        parts.append(PROTOCOL_PROMPT.replace("{sfx}", "、".join(sfx)))
        if self.post_history_instructions:
            parts.append(self.render(self.post_history_instructions))
        return "\n\n".join(parts)

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "ambience": self.ambience, "scenario": self.render(self.scenario)}


def _parse_position(value: str) -> Position:
    parser = TagParser()
    parser.feed(f"[pos: {value}]")
    return parser.position


def character_from_card(card: dict, card_id: str) -> Character:
    data = card.get("data", card)  # V2 包在 data 里，V1 平铺
    ext = (data.get("extensions") or {}).get("asmr") or {}
    return Character(
        id=card_id,
        name=data.get("name", card_id),
        description=data.get("description", ""),
        personality=data.get("personality", ""),
        scenario=data.get("scenario", ""),
        first_mes=data.get("first_mes", ""),
        mes_example=data.get("mes_example", ""),
        system_prompt=data.get("system_prompt", ""),
        post_history_instructions=data.get("post_history_instructions", ""),
        voice=ext.get("voice", ""),
        gptsovits=ext.get("gptsovits") or {},
        style=ext.get("style", ""),
        position=_parse_position(ext.get("position", "front, mid")),
        ambience=ext.get("ambience", "rain"),
        user_name=ext.get("user_name", "你"),
    )


def read_png_card(path: Path) -> dict:
    """读取酒馆 PNG 角色卡：tEXt/iTXt/zTXt 块 ``chara``（或 V3 的 ``ccv3``）里是 base64 JSON。"""
    blob = path.read_bytes()
    if blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} 不是 PNG")
    offset, found = 8, {}
    while offset + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[offset : offset + 4])
        ctype = blob[offset + 4 : offset + 8]
        body = blob[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if ctype == b"tEXt":
            key, _, value = body.partition(b"\x00")
        elif ctype == b"zTXt":
            key, _, rest = body.partition(b"\x00")
            value = zlib.decompress(rest[1:])
        elif ctype == b"iTXt":
            key, _, rest = body.partition(b"\x00")
            compressed, rest = rest[0], rest[2:]
            _, _, rest = rest.partition(b"\x00")  # language tag
            _, _, value = rest.partition(b"\x00")  # translated keyword
            if compressed:
                value = zlib.decompress(value)
        else:
            continue
        found[key.decode("latin-1").lower()] = value
    for key in ("ccv3", "chara"):
        if key in found:
            return json.loads(base64.b64decode(found[key]).decode("utf-8"))
    raise ValueError(f"{path} 里没有角色卡数据")


def load_characters(directory: Path) -> dict[str, Character]:
    characters: dict[str, Character] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() == ".json":
            card = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() == ".png":
            card = read_png_card(path)
        else:
            continue
        characters[path.stem] = character_from_card(card, path.stem)
    return characters
