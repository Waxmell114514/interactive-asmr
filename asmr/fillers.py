"""应声词缓存：用同一音色预先合成“嗯……”“这样啊”，用户说完立刻播，把等待包装成倾听。

缓存按 (模型, 音色, 文本) 存到 data/fillers/，只在第一次生成时花 API 费用。
也可以手动生成：``python -m asmr.fillers``。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from pathlib import Path

from .tts import TTS, synthesize_all

log = logging.getLogger(__name__)

FILLERS = [
    ("嗯……", "轻声，若有所思"),
    ("嗯。", "轻声，温柔地回应"),
    ("嗯嗯……", "轻声，温柔地附和"),
    ("这样啊……", "轻声，慢慢地"),
    ("唔……", "轻声，思考"),
]


class FillerCache:
    def __init__(self, directory: Path, *, model: str, voice: str):
        self._dir = directory
        self._key = f"{model}|{voice}"
        self._clips: list[bytes] = []
        self._lock = asyncio.Lock()

    def _path(self, text: str, style: str) -> Path:
        digest = hashlib.sha1(f"{self._key}|{text}|{style}".encode()).hexdigest()[:16]
        return self._dir / f"{digest}.pcm"

    def load(self) -> int:
        self._clips = [p.read_bytes() for text, style in FILLERS if (p := self._path(text, style)).exists()]
        return len(self._clips)

    async def ensure(self, tts: TTS) -> int:
        """缺哪个补哪个；失败只记日志，不影响会话。"""
        async with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            for text, style in FILLERS:
                path = self._path(text, style)
                if path.exists():
                    continue
                try:
                    pcm = await synthesize_all(tts, text, style)
                except Exception as e:  # noqa: BLE001
                    log.warning("应声词合成失败 %s: %s", text, e)
                    continue
                if pcm:
                    path.write_bytes(pcm)
            return self.load()

    def pick(self) -> bytes | None:
        return random.choice(self._clips) if self._clips else None


async def _main() -> None:
    from .config import Settings
    from .tts import GeminiTTS

    logging.basicConfig(level=logging.INFO)
    s = Settings.from_env()
    tts = GeminiTTS(api_key=s.gemini_api_key, voice=s.tts_voice, model=s.tts_model, base_url=s.gemini_base_url)
    cache = FillerCache(s.data_dir / "fillers", model=s.tts_model, voice=s.tts_voice)
    n = await cache.ensure(tts)
    await tts.aclose()
    print(f"应声词缓存：{n}/{len(FILLERS)} 条")


if __name__ == "__main__":
    asyncio.run(_main())
