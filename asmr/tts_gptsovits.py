"""本地 TTS：GPT-SoVITS（api_v2.py 的 ``POST /tts``）。

GPT-SoVITS 不接受语气描述，语气由参考音频决定。所以 ``[style: ...]`` 按关键词挑参考音频：
style 里含“耳语”就用耳语参考，含“笑”就用带笑意的参考，都不匹配用默认参考。
Gemini 的声音事件（``<sigh>`` 等）GPT-SoVITS 不认识，送进去之前换成标点或删掉。

输出按 WAV 头里的采样率（v2/v2Pro 为 32kHz）重采样到 24kHz，与前端约定一致。

配置写在角色卡 ``extensions.asmr.gptsovits``，或者用 ``GPTSOVITS_CONFIG`` 指向同结构的 JSON 文件::

    {
      "url": "http://127.0.0.1:9880",
      "text_lang": "zh",
      "speed_factor": 0.9,
      "gpt_weights": "GPT_weights_v2Pro/xxx.ckpt",        # 可选：启动时切换模型
      "sovits_weights": "SoVITS_weights_v2Pro/xxx.pth",   # 可选
      "refs": [
        {"match": ["耳语", "气声", "悄悄"], "audio": "refs/whisper.wav", "text": "参考音频对应的文字", "lang": "zh"},
        {"match": ["笑"], "audio": "refs/smile.wav", "text": "……"},
        {"audio": "refs/normal.wav", "text": "……", "default": true}
      ]
    }

``audio`` 是 GPT-SoVITS 服务那一侧能读到的路径（同一台机器上直接写本地路径即可）。

单独试听：``python -m asmr.tts_gptsovits "嗯……今天辛苦了。" --style 耳语 -o test.wav``
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import httpx

from .refs import PcmStream, RefAudio, parse_refs, pick_ref
from .tts import SAMPLE_RATE, TTSError

log = logging.getLogger(__name__)

# <short pause> 之类换成标点，让 GPT-SoVITS 自己停顿；其它声音事件直接去掉。
_PAUSES = {"short pause": "，", "long pause": "……", "breath": "，"}
_TAG_RE = re.compile(r"<([^<>]*)>")
_SPEAKABLE_RE = re.compile(r"[\w一-鿿]")


@dataclass
class GPTSoVITSConfig:
    url: str = "http://127.0.0.1:9880"
    text_lang: str = "zh"
    refs: list[RefAudio] = field(default_factory=list)
    speed_factor: float = 1.0
    streaming_mode: int = 1  # 0 关闭；1 质量最好（按句返回）；2/3 更快但音质下降
    text_split_method: str = "cut5"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    seed: int = -1
    gpt_weights: str = ""
    sovits_weights: str = ""

    @classmethod
    def from_dict(cls, data: dict, *, url: str = "") -> "GPTSoVITSConfig":
        data = dict(data)
        refs = parse_refs(data.pop("refs", []))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known, refs=refs)
        if url and "url" not in data:
            cfg.url = url
        return cfg

    @classmethod
    def from_file(cls, path: Path, *, url: str = "") -> "GPTSoVITSConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), url=url)

    def pick_ref(self, style: str) -> RefAudio:
        return pick_ref(self.refs, style)

    def cache_key(self) -> str:
        blob = json.dumps(
            [self.url, self.gpt_weights, self.sovits_weights, [r.__dict__ for r in self.refs], self.speed_factor],
            ensure_ascii=False,
            sort_keys=True,
        )
        return "gptsovits|" + hashlib.sha1(blob.encode()).hexdigest()[:12]


def prepare_text(text: str) -> str:
    """去掉 GPT-SoVITS 不认识的声音事件；只剩标点时返回空串。"""

    def repl(m: re.Match) -> str:
        return _PAUSES.get(m.group(1).strip().lower(), "")

    out = _TAG_RE.sub(repl, text)
    out = re.sub(r"\s+", " ", out).strip(" ，")
    return out if _SPEAKABLE_RE.search(out) else ""


def _parse_wav_header(buf: bytes) -> tuple[int, int] | None:
    """返回 (采样率, PCM 数据起始偏移)；头还没收全时返回 None。"""
    if len(buf) < 12:
        return None
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise TTSError("GPT-SoVITS 返回的不是 WAV（请用 media_type=wav）")
    offset, rate = 12, None
    while offset + 8 <= len(buf):
        cid, size = buf[offset : offset + 4], struct.unpack("<I", buf[offset + 4 : offset + 8])[0]
        if cid == b"fmt ":
            if offset + 8 + 16 > len(buf):
                return None
            channels, rate, _, _, bits = struct.unpack("<HIIHH", buf[offset + 10 : offset + 24])
            if channels != 1 or bits != 16:
                raise TTSError(f"只支持单声道 16-bit，收到 {channels} 声道 {bits}-bit")
        elif cid == b"data":
            if rate is None:
                raise TTSError("WAV 头缺少 fmt 块")
            return rate, offset + 8
        offset += 8 + size + (size & 1)
    return None


class GPTSoVITSTTS:
    sample_rate = SAMPLE_RATE

    def __init__(self, config: GPTSoVITSConfig, *, client: httpx.AsyncClient | None = None):
        self.config = config
        self.cache_key = config.cache_key()
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        self._weights_ready = not (config.gpt_weights or config.sovits_weights)
        self._weights_lock = asyncio.Lock()

    async def _ensure_weights(self) -> None:
        if self._weights_ready:
            return
        async with self._weights_lock:
            if self._weights_ready:
                return
            base = self.config.url.rstrip("/")
            for endpoint, path in (
                ("set_gpt_weights", self.config.gpt_weights),
                ("set_sovits_weights", self.config.sovits_weights),
            ):
                if not path:
                    continue
                resp = await self._client.get(f"{base}/{endpoint}", params={"weights_path": path})
                if resp.status_code != 200:
                    raise TTSError(f"GPT-SoVITS {endpoint} 失败: {resp.text[:300]}")
            self._weights_ready = True

    def build_request(self, text: str, style: str) -> dict:
        cfg, ref = self.config, self.config.pick_ref(style)
        return {
            "text": text,
            "text_lang": cfg.text_lang,
            "ref_audio_path": ref.audio,
            "prompt_text": ref.text,
            "prompt_lang": ref.lang,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "temperature": cfg.temperature,
            "text_split_method": cfg.text_split_method,
            "speed_factor": cfg.speed_factor,
            "seed": cfg.seed,
            "media_type": "wav",
            "streaming_mode": cfg.streaming_mode,
        }

    async def synthesize(self, text: str, style: str = "") -> AsyncIterator[bytes]:
        text = prepare_text(text)
        if not text:
            return
        await self._ensure_weights()
        body = self.build_request(text, style)
        url = f"{self.config.url.rstrip('/')}/tts"
        buf = b""
        header: tuple[int, int] | None = None
        stream: PcmStream | None = None
        try:
            async with self._client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                    raise TTSError(f"GPT-SoVITS HTTP {resp.status_code}: {detail}")
                async for data in resp.aiter_bytes():
                    if stream is None:
                        buf += data
                        header = _parse_wav_header(buf)
                        if header is None:
                            continue
                        rate, start = header
                        stream = PcmStream(rate)
                        data, buf = buf[start:], b""
                    out = stream.push(data)
                    if out:
                        yield out
        except httpx.HTTPError as e:
            raise TTSError(f"连不上 GPT-SoVITS（{self.config.url}）：{e}") from e
        if stream is None:
            raise TTSError("GPT-SoVITS 没有返回音频")
        tail = stream.finish()
        if tail:
            yield tail

    async def aclose(self) -> None:
        await self._client.aclose()


async def _main() -> None:
    import argparse
    import wave

    from .character import load_characters
    from .config import Settings

    parser = argparse.ArgumentParser(description="用 GPT-SoVITS 合成一句话试听")
    parser.add_argument("text")
    parser.add_argument("--style", default="", help="用来挑参考音频，例如 耳语")
    parser.add_argument("--character", default="", help="用这张角色卡里的 gptsovits 配置")
    parser.add_argument("-o", "--output", default="gptsovits_test.wav")
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.character:
        character = load_characters(settings.characters_dir)[args.character]
        cfg = GPTSoVITSConfig.from_dict(character.gptsovits, url=settings.gptsovits_url)
    else:
        if not settings.gptsovits_config:
            raise SystemExit("请设置 GPTSOVITS_CONFIG，或用 --character 指定带 gptsovits 配置的角色卡")
        cfg = GPTSoVITSConfig.from_file(settings.gptsovits_config, url=settings.gptsovits_url)
    tts = GPTSoVITSTTS(cfg)
    ref = cfg.pick_ref(args.style)
    print(f"参考音频：{ref.audio}")
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
