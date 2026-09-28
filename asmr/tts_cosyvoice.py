"""本地 TTS：CosyVoice 3（也兼容 CosyVoice 2），调用官方 ``runtime/python/fastapi/server.py``。

启动服务（在 CosyVoice 仓库里）::

    cd runtime/python/fastapi
    python server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B

和 GPT-SoVITS 不同，CosyVoice 能直接听懂语气指令：

- ``[style: ...]`` 非空时走 ``/inference_instruct2``：参考音频定音色，style 转成一句指令定语气；
- style 为空时走 ``/inference_zero_shot``：完全照着参考音频的音色和语气读。

参考音频同样按 style 关键词挑（可以同时有耳语参考 + “轻声”指令）。官方服务要求上传参考音频文件，
所以这里的 ``audio`` 是**本项目这台机器**上的路径。

Gemini 的声音事件转成 CosyVoice 的细粒度标记：``<sigh>`` → ``[sigh]``，``<breath>`` → ``[breath]``，
``<laugh>`` → ``[laughter]``，``<cough>`` → ``[cough]``；停顿换成标点，其余去掉。

配置写在角色卡 ``extensions.asmr.cosyvoice``，或 ``COSYVOICE_CONFIG`` 指向的 JSON，格式见 ``cosyvoice.example.json``。

单独试听：``python -m asmr.tts_cosyvoice "嗯……今天辛苦了。" --style 耳语 -o test.wav``
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx

from .refs import PcmStream, RefAudio, parse_refs, pick_ref
from .tts import SAMPLE_RATE, TTSError

log = logging.getLogger(__name__)

TAG_MAP = {
    "sigh": "[sigh]", "sighs": "[sigh]",
    "breath": "[breath]", "heavy breath": "[breath]", "exhales": "[breath]", "pant": "[quick_breath]",
    "laugh": "[laughter]", "laughter": "[laughter]", "chuckle": "[laughter]", "chuckles": "[laughter]",
    "giggle": "[laughter]", "snicker": "[laughter]",
    "cough": "[cough]", "throat-clearing": "[cough]",
    "tsk": "[clucking]",
    "short pause": "，", "long pause": "……",
}
_TAG_RE = re.compile(r"<([^<>]*)>")
_SPEAKABLE_RE = re.compile(r"[\w一-鿿]")


@dataclass
class StyleInstruction:
    match: list[str]
    instruction: str


@dataclass
class CosyVoiceConfig:
    url: str = "http://127.0.0.1:50000"
    version: int = 3  # 3：提示词要带 system_prompt 和 <|endofprompt|>；2：不带 system_prompt
    sample_rate: int = 24000  # CosyVoice 2/3 均为 24kHz；CosyVoice 1 为 22050
    system_prompt: str = "You are a helpful assistant."
    # style 不命中下面的映射时，用这个模板把 style 变成指令
    instruct_template: str = "请用{style}的语气说这句话。"
    # style 含关键词时直接用这句指令（可以填模型训练时见过的原句，效果更稳）
    instructions: list[StyleInstruction] = field(default_factory=list)
    refs: list[RefAudio] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict, *, url: str = "") -> "CosyVoiceConfig":
        data = dict(data)
        refs = parse_refs(data.pop("refs", []))
        instructions = [StyleInstruction(**item) for item in data.pop("instructions", [])]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known, refs=refs, instructions=instructions)
        if url and "url" not in data:
            cfg.url = url
        return cfg

    @classmethod
    def from_file(cls, path: Path, *, url: str = "") -> "CosyVoiceConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), url=url)

    def instruction_for(self, style: str) -> str:
        for item in self.instructions:
            if any(word and word in style for word in item.match):
                return item.instruction
        return self.instruct_template.format(style=style)

    def instruct_text(self, style: str) -> str:
        body = self.instruction_for(style)
        if self.version >= 3:
            body = f"{self.system_prompt} {body}"
        return f"{body}<|endofprompt|>"

    def prompt_text(self, ref: RefAudio) -> str:
        if self.version >= 3:
            return f"{self.system_prompt}<|endofprompt|>{ref.text}"
        return ref.text

    def cache_key(self) -> str:
        blob = json.dumps(
            [self.url, self.version, self.instruct_template, [i.__dict__ for i in self.instructions],
             [r.__dict__ for r in self.refs]],
            ensure_ascii=False,
            sort_keys=True,
        )
        return "cosyvoice|" + hashlib.sha1(blob.encode()).hexdigest()[:12]


def prepare_text(text: str) -> str:
    """把声音事件换成 CosyVoice 的标记；只剩标点和标记时返回空串。"""
    out = _TAG_RE.sub(lambda m: TAG_MAP.get(m.group(1).strip().lower(), ""), text)
    out = re.sub(r"\s+", " ", out).strip(" ，")
    return out if _SPEAKABLE_RE.search(out) else ""  # 单独一个 [sigh] 也保留


class CosyVoiceTTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, config: CosyVoiceConfig, *, client: httpx.AsyncClient | None = None):
        self.config = config
        self.cache_key = config.cache_key()
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        self._wavs: dict[str, bytes] = {}

    def _ref_bytes(self, ref: RefAudio) -> bytes:
        if ref.audio not in self._wavs:
            path = Path(ref.audio).expanduser()
            if not path.exists():
                raise TTSError(f"找不到参考音频：{path}")
            self._wavs[ref.audio] = path.read_bytes()
        return self._wavs[ref.audio]

    def build_request(self, text: str, style: str) -> tuple[str, dict]:
        """返回 (接口名, 表单字段)。"""
        cfg, ref = self.config, pick_ref(self.config.refs, style)
        if style.strip():
            return "inference_instruct2", {"tts_text": text, "instruct_text": cfg.instruct_text(style)}
        return "inference_zero_shot", {"tts_text": text, "prompt_text": cfg.prompt_text(ref)}

    async def synthesize(self, text: str, style: str = "") -> AsyncIterator[bytes]:
        text = prepare_text(text)
        if not text:
            return
        endpoint, form = self.build_request(text, style)
        ref = pick_ref(self.config.refs, style)
        files = {"prompt_wav": (Path(ref.audio).name, self._ref_bytes(ref), "audio/wav")}
        url = f"{self.config.url.rstrip('/')}/{endpoint}"
        stream = PcmStream(self.config.sample_rate)
        got_audio = False
        try:
            async with self._client.stream("POST", url, data=form, files=files) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise TTSError(f"CosyVoice HTTP {resp.status_code}: {detail}")
                async for data in resp.aiter_bytes():
                    out = stream.push(data)
                    if out:
                        got_audio = True
                        yield out
        except httpx.HTTPError as e:
            raise TTSError(f"连不上 CosyVoice（{self.config.url}）：{e}") from e
        tail = stream.finish()
        if tail:
            got_audio = True
            yield tail
        if not got_audio:
            raise TTSError("CosyVoice 没有返回音频")

    async def aclose(self) -> None:
        await self._client.aclose()


async def _main() -> None:
    import argparse
    import wave

    from .character import load_characters
    from .config import Settings

    parser = argparse.ArgumentParser(description="用 CosyVoice 合成一句话试听")
    parser.add_argument("text")
    parser.add_argument("--style", default="", help="语气，例如 耳语，很轻；留空走 zero-shot")
    parser.add_argument("--character", default="", help="用这张角色卡里的 cosyvoice 配置")
    parser.add_argument("-o", "--output", default="cosyvoice_test.wav")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.character:
        character = load_characters(settings.characters_dir)[args.character]
        cfg = CosyVoiceConfig.from_dict(character.cosyvoice, url=settings.cosyvoice_url)
    else:
        if not settings.cosyvoice_config:
            raise SystemExit("请设置 COSYVOICE_CONFIG，或用 --character 指定带 cosyvoice 配置的角色卡")
        cfg = CosyVoiceConfig.from_file(settings.cosyvoice_config, url=settings.cosyvoice_url)
    tts = CosyVoiceTTS(cfg)
    endpoint, form = tts.build_request(prepare_text(args.text), args.style)
    print(f"接口：/{endpoint}  参考音频：{pick_ref(cfg.refs, args.style).audio}")
    print(f"文本：{form['tts_text']}")
    if "instruct_text" in form:
        print(f"指令：{form['instruct_text']}")
    loop = asyncio.get_running_loop()
    start = loop.time()
    first = None
    chunks = []
    async for chunk in tts.synthesize(args.text, args.style):
        first = first if first is not None else loop.time() - start
        chunks.append(chunk)
    await tts.aclose()
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(chunks))
    print(f"首包 {(first or 0) * 1000:.0f}ms，总计 {loop.time() - start:.2f}s → {args.output}")


if __name__ == "__main__":
    asyncio.run(_main())
