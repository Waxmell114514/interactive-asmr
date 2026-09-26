"""FastAPI 服务：静态前端 + /api/characters + /ws 会话通道。

启动：``python -m asmr.server``（或 ``asmr-server``），浏览器打开 http://127.0.0.1:8000 。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .character import Character, load_characters
from .config import Settings
from .fillers import FillerCache
from .llm import OpenAICompatibleLLM
from .session import Services, Session
from .stt import create_stt
from .tts import GeminiTTS
from .turn import create_turn_detector

log = logging.getLogger(__name__)


class WebSocketTransport:
    def __init__(self, ws: WebSocket):
        self._ws = ws

    async def send_json(self, data: dict) -> None:
        await self._ws.send_text(json.dumps(data, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send_bytes(data)

    async def receive(self) -> str | bytes | None:
        try:
            msg = await self._ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None
        if msg["type"] == "websocket.disconnect":
            return None
        if msg.get("bytes") is not None:
            return msg["bytes"]
        return msg.get("text")

    async def close(self) -> None:
        await self._ws.close()


class Runtime:
    """进程级共享资源：STT 模型只加载一次；TTS 客户端和应声词缓存按音色复用。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._stt = None
        self._llm = None
        self._tts: dict[str, GeminiTTS] = {}
        self._fillers: dict[str, FillerCache] = {}
        self._background: set[asyncio.Task] = set()

    def load(self) -> None:
        s = self.settings
        log.info("加载 STT（%s, %s）……", s.stt_backend, s.stt_device)
        self._stt = create_stt(s.stt_backend, model=s.stt_model, device=s.stt_device, language=s.stt_language)
        self._llm = OpenAICompatibleLLM(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            model=s.llm_model,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
        )

    def services_for(self, character: Character) -> Services:
        s = self.settings
        if self._stt is None:
            self.load()
        voice = character.voice or s.tts_voice
        if voice not in self._tts:
            self._tts[voice] = GeminiTTS(
                api_key=s.gemini_api_key,
                voice=voice,
                model=s.tts_model,
                style_mode=s.tts_style_mode,
                base_url=s.gemini_base_url,
            )
            cache = FillerCache(s.data_dir / "fillers", model=s.tts_model, voice=voice)
            cache.load()
            self._fillers[voice] = cache
        tts, fillers = self._tts[voice], self._fillers[voice]
        if s.fillers:
            task = asyncio.get_running_loop().create_task(fillers.ensure(tts))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        turn = create_turn_detector(
            confidence=s.vad_confidence,
            start_secs=s.vad_start_secs,
            stop_secs=s.vad_stop_secs,
            turn_stop_secs=s.turn_stop_secs,
            smart_turn=s.smart_turn,
        )
        return Services(stt=self._stt, llm=self._llm, tts=tts, turn_detector=turn, fillers=fillers)


def create_app(
    settings: Settings | None = None,
    services_factory: Callable[[Character], Services] | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    characters = load_characters(settings.characters_dir)
    runtime = Runtime(settings) if services_factory is None else None

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        if runtime is not None:
            await asyncio.to_thread(runtime.load)
        yield

    app = FastAPI(title="互动 ASMR", lifespan=lifespan)
    app.state.settings = settings
    app.state.characters = characters

    @app.get("/api/characters")
    async def list_characters() -> dict:
        sounds_dir = settings.web_dir / "sounds"
        sounds = sorted(p.relative_to(sounds_dir).as_posix() for p in sounds_dir.rglob("*") if p.is_file())
        return {"characters": [c.summary() for c in characters.values()], "sfx": list(settings.sfx), "sounds": sounds}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        transport = WebSocketTransport(ws)
        first = await transport.receive()
        try:
            start = json.loads(first) if isinstance(first, str) else {}
        except json.JSONDecodeError:
            start = {}
        character = characters.get(start.get("character", "")) or next(iter(characters.values()), None)
        if start.get("type") != "start" or character is None:
            await transport.send_json({"type": "error", "message": "需要先发送 start，且至少有一张角色卡"})
            await ws.close()
            return
        factory = services_factory or runtime.services_for
        session = Session(transport, factory(character), settings, character)
        log.info("会话开始：%s", character.name)
        try:
            await session.run()
        finally:
            log.info("会话结束：%s", character.name)
            with contextlib.suppress(Exception):
                await ws.close()

    app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    return app


def main() -> None:
    import sys

    import uvicorn
    from loguru import logger

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.remove()  # Pipecat 用 loguru，默认 DEBUG 太吵
    logger.add(sys.stderr, level="INFO")
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
