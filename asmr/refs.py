"""本地 TTS 后端共用的小工具：按 style 关键词挑参考音频、流式重采样到 24kHz。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tts import SAMPLE_RATE


@dataclass
class RefAudio:
    audio: str
    text: str = ""
    lang: str = "zh"
    match: list[str] = field(default_factory=list)
    default: bool = False


def parse_refs(items: list[dict]) -> list[RefAudio]:
    refs = [RefAudio(**item) for item in items]
    if not refs:
        raise ValueError("配置里至少要有一段参考音频（refs）")
    return refs


def pick_ref(refs: list[RefAudio], style: str) -> RefAudio:
    """style 含某段参考的任一关键词就用它；都不匹配用 default（没有就用最后一段）。"""
    for ref in refs:
        if any(word and word in style for word in ref.match):
            return ref
    return next((r for r in refs if r.default), refs[-1])


class PcmStream:
    """把任意切分的 16-bit PCM 字节流对齐到采样点，并重采样到 24kHz。"""

    def __init__(self, rate: int):
        self._carry = b""
        self._resampler = None
        if rate != SAMPLE_RATE:
            import soxr

            self._resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="int16")

    def push(self, data: bytes) -> bytes:
        buf = self._carry + data
        cut = len(buf) - len(buf) % 2
        pcm, self._carry = buf[:cut], buf[cut:]
        return self._convert(pcm, last=False) if pcm else b""

    def finish(self) -> bytes:
        return self._convert(b"", last=True)

    def _convert(self, pcm: bytes, *, last: bool) -> bytes:
        if self._resampler is None:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16)
        return self._resampler.resample_chunk(x, last=last).astype(np.int16).tobytes()
