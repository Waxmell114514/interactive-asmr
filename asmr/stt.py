"""本地 STT：SenseVoice（FunASR，首选）或 faster-whisper（备选）。输入 16kHz 单声道 16-bit PCM。"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class STT(Protocol):
    async def transcribe(self, pcm16: bytes) -> str: ...


def _to_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0


class SenseVoiceSTT:
    def __init__(self, *, model: str = "", device: str = "cpu", language: str = "zh"):
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self._post = rich_transcription_postprocess
        self._language = language or "auto"
        self._model = AutoModel(model=model or "iic/SenseVoiceSmall", device=device, disable_update=True)
        self._lock = asyncio.Lock()

    async def transcribe(self, pcm16: bytes) -> str:
        audio = _to_float(pcm16)

        def run() -> str:
            result = self._model.generate(input=audio, cache={}, language=self._language, use_itn=True)
            return self._post(result[0]["text"]).strip() if result else ""

        async with self._lock:
            return await asyncio.to_thread(run)


class WhisperSTT:
    def __init__(self, *, model: str = "", device: str = "cpu", language: str = "zh"):
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model or "large-v3-turbo", device=device, compute_type=compute_type)
        self._language = language or None
        self._lock = asyncio.Lock()

    async def transcribe(self, pcm16: bytes) -> str:
        audio = _to_float(pcm16)

        def run() -> str:
            segments, _ = self._model.transcribe(audio, language=self._language, vad_filter=False, beam_size=1)
            return "".join(s.text for s in segments).strip()

        async with self._lock:
            return await asyncio.to_thread(run)


INSTALL_HINTS = {"sensevoice": 'pip install -e ".[sensevoice]"', "whisper": 'pip install -e ".[whisper]"'}


def create_stt(backend: str, *, model: str, device: str, language: str) -> STT:
    cls = {"sensevoice": SenseVoiceSTT, "whisper": WhisperSTT}.get(backend)
    if cls is None:
        raise ValueError(f"未知 STT 后端: {backend}（可选 sensevoice / whisper）")
    try:
        return cls(model=model, device=device, language=language)
    except ImportError as e:
        raise RuntimeError(f"STT 后端 {backend} 缺少依赖（{e}）。请运行：{INSTALL_HINTS[backend]}") from e
