"""会话测试用的假服务和假浏览器。"""

from __future__ import annotations

import asyncio
import json
import struct

from asmr.character import Character
from asmr.config import Settings
from asmr.session import Services, Session
from asmr.state import Timing
from asmr.tags import Position
from asmr.turn import SpeechStarted, TurnEnded


class FakeSTT:
    def __init__(self):
        self.texts: dict[bytes, str] = {}

    async def transcribe(self, pcm16: bytes) -> str:
        return self.texts.get(pcm16, "")


class FakeLLM:
    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.calls: list[list[dict]] = []

    async def stream(self, messages):
        self.calls.append([dict(m) for m in messages])
        text = self.replies.pop(0) if self.replies else "嗯。"
        for i in range(0, len(text), 3):
            await asyncio.sleep(0)
            yield text[i : i + 3]


class FakeTTS:
    sample_rate = 24000

    def __init__(self, delay: float = 0.0, fail_on: str | None = None):
        self.delay = delay
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    async def synthesize(self, text, style=""):
        self.calls.append((text, style))
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("blocked")
        for _ in range(2):
            await asyncio.sleep(self.delay)
            yield b"\x10\x00" * 240


class FakeTurnDetector:
    """音频块 b"S" = 开口；b"E" + 内容 = 说完（内容交给 FakeSTT 查表）。"""

    async def process(self, chunk: bytes):
        if chunk == b"S":
            return [SpeechStarted()]
        if chunk.startswith(b"E"):
            return [TurnEnded(chunk[1:])]
        return []


class FakeBrowser:
    """实现 Transport；默认收到 segment_end 就立刻“播完”并回报。"""

    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.audio: dict[int, int] = {}
        self.autoplay = True
        self.closed = False

    # Transport
    async def send_json(self, data: dict) -> None:
        data = json.loads(json.dumps(data))
        self.sent.append(data)
        if data["type"] == "segment_end" and self.autoplay:
            await self.inbox.put(json.dumps({"type": "seg_start", "seg": data["seg"]}))
            await self.inbox.put(json.dumps({"type": "seg_end", "seg": data["seg"]}))

    async def send_bytes(self, data: bytes) -> None:
        (seg,) = struct.unpack("<I", data[:4])
        self.audio[seg] = self.audio.get(seg, 0) + len(data) - 4

    async def receive(self):
        return await self.inbox.get()

    async def close(self) -> None:
        self.closed = True
        await self.inbox.put(None)

    # 测试辅助
    async def say(self, audio_key: bytes) -> None:
        await self.inbox.put(b"S")
        await self.inbox.put(b"E" + audio_key)

    async def report(self, kind: str, seg: int) -> None:
        await self.inbox.put(json.dumps({"type": kind, "seg": seg}))

    def of_type(self, kind: str) -> list[dict]:
        return [m for m in self.sent if m["type"] == kind]

    def states(self) -> list[str]:
        return [m["state"] for m in self.of_type("state")]


def make_session(
    *,
    replies=None,
    first_mes="[pos: left, near]你来啦。今天辛苦了。",
    timing: Timing | None = None,
    tts: FakeTTS | None = None,
):
    settings = Settings(intro_delay=0.0, fillers=False, timing=timing or Timing(t1=1000, silence_to_sleep=10_000))
    character = Character(id="t", name="小雪", first_mes=first_mes, style="温柔", position=Position("front", "mid"))
    browser = FakeBrowser()
    stt, llm, tts = FakeSTT(), FakeLLM(replies), tts or FakeTTS()
    services = Services(stt=stt, llm=llm, tts=tts, turn_detector=FakeTurnDetector())
    session = Session(browser, services, settings, character, tick_interval=0.01)
    return session, browser, stt, llm, tts


async def wait_for(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("等待超时")
        await asyncio.sleep(0.005)
