"""一个 WebSocket 连接 = 一场会话：把轮次检测、STT、LLM、标签解析、TTS 和状态机串起来。

下行协议（后端 → 浏览器）
    文本 JSON：
        {"type": "state", "state": "waiting"}
        {"type": "segment", "seg": 7, "kind": "reply", "text": "字幕", "style": "...",
         "cues": [{"kind": "pos", "azimuth": "left", "distance": "near"}], "gain_db": 0}
        {"type": "segment_end", "seg": 7}          该段音频已全部下发
        {"type": "interrupt"}                      淡出并清空播放队列
        {"type": "transcript", "text": "..."}      用户说的话（可选显示）
        {"type": "end", "fade_secs": 60}           入睡定时到，环境音淡出后结束
    二进制：4 字节小端 uint32 段号 + 24kHz 单声道 16-bit PCM

上行协议（浏览器 → 后端）
    二进制：16kHz 单声道 16-bit PCM 麦克风数据
    文本 JSON：{"type": "seg_start", "seg": 7} / {"type": "seg_end", "seg": 7} 播放进度回报；
              {"type": "stop"}

对话历史只记录**实际播出**的段落：浏览器回报 seg_start 后才算说过，被打断时未播的部分丢弃。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from .character import INTRO_PROMPT, MONOLOGUE_PROMPT, WAKE_HINT, Character
from .config import Settings
from .fillers import FillerCache
from .llm import LLM
from .loudness import LoudnessLeveler
from .segments import Segment, Segmenter
from .state import (
    Action,
    EndSession,
    EnterSleep,
    Interrupt,
    Respond,
    SessionStateMachine,
    StartMonologue,
    State,
)
from .stt import STT
from .tags import Position, Sentence, TagParser
from .tts import TTS
from .turn import SpeechStarted, TurnEnded

log = logging.getLogger(__name__)

# 从 Sleep 唤醒后的首个回复强制使用的 style。
LIGHTEST_STYLE = "极轻的耳语，很慢，几乎只有气声"
# X6 兜底：独白里出现这类追问就整句丢掉。
CHECK_IN_RE = re.compile(r"(还在(吗|嘛|么)|睡着了(吗|嘛|么)|在听(吗|嘛|么)|听得到(吗|嘛|么)|听见了?(吗|嘛|么)|怎么不说话)")
END_FADE_SECS = 60.0


class Transport(Protocol):
    async def send_json(self, data: dict) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...
    async def receive(self) -> str | bytes | None: ...
    async def close(self) -> None: ...


class TurnDetectorLike(Protocol):
    async def process(self, chunk: bytes) -> list: ...


@dataclass
class Services:
    stt: STT
    llm: LLM
    tts: TTS
    turn_detector: TurnDetectorLike
    fillers: FillerCache | None = None


@dataclass
class Response:
    kind: str  # intro | reply | monologue
    user_parts: list[str] = field(default_factory=list)
    prompt: str | None = None  # 隐藏的 user 消息（独白 / 开场提示）
    gain_db: float = 0.0
    style_override: str | None = None
    gentle: bool = False
    order: list[int] = field(default_factory=list)
    segments: dict[int, Segment] = field(default_factory=dict)
    started: set[int] = field(default_factory=set)
    done: set[int] = field(default_factory=set)
    generation_done: bool = False
    audio_started: bool = False
    task: asyncio.Task | None = None

    @property
    def user_text(self) -> str:
        return " ".join(self.user_parts)


class Session:
    def __init__(
        self,
        transport: Transport,
        services: Services,
        settings: Settings,
        character: Character,
        *,
        clock: Callable[[], float] = time.monotonic,
        tick_interval: float = 0.25,
    ):
        self.transport = transport
        self.services = services
        self.settings = settings
        self.character = character
        self.clock = clock
        self.tick_interval = tick_interval
        self.sm = SessionStateMachine(settings.timing)
        self.history: list[dict] = []
        self.system_prompt = character.build_system_prompt(settings.sfx)
        self.style = character.style
        self.position: Position = character.position
        self._pending_user: list[str] = []
        self._response: Response | None = None
        self._seg_counter = 0
        self._turn_seq = 0
        self._sent_state: State | None = None
        self._send_lock = asyncio.Lock()
        self._leveler = LoudnessLeveler(sample_rate=services.tts.sample_rate)
        self._tasks: set[asyncio.Task] = set()
        self._closed = asyncio.Event()

    # ---- 主循环 -----------------------------------------------------------

    async def run(self) -> None:
        self.sm.start(self.clock())
        await self._sync_state()
        self._spawn(self._ticker())
        self._spawn(self._intro())
        try:
            while not self._closed.is_set():
                msg = await self.transport.receive()
                if msg is None:
                    break
                if isinstance(msg, (bytes, bytearray)):
                    await self._on_audio(bytes(msg))
                else:
                    await self._on_message(json.loads(msg))
        finally:
            self._closed.set()
            self.sm.stop()
            await self._cancel_response()
            for task in list(self._tasks):
                task.cancel()
            for task in list(self._tasks):
                with contextlib.suppress(BaseException):
                    await task

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception():
            log.error("会话任务异常", exc_info=task.exception())

    async def _ticker(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(self.tick_interval)
            await self._apply(self.sm.tick(self.clock()))

    async def _intro(self) -> None:
        await asyncio.sleep(self.settings.intro_delay)
        if self.sm.state != State.INTRO:
            return  # 开场前用户已经开口
        if self.character.first_mes:
            self._start_response(Response(kind="intro"))
        else:
            self._start_response(Response(kind="intro", prompt=INTRO_PROMPT))

    # ---- 上行消息 ---------------------------------------------------------

    async def _on_audio(self, chunk: bytes) -> None:
        for event in await self.services.turn_detector.process(chunk):
            if isinstance(event, SpeechStarted):
                self._turn_seq += 1
                await self._apply(self.sm.user_speech_start(self.clock()))
            elif isinstance(event, TurnEnded):
                self._spawn(self._transcribe(event.audio, self._turn_seq))

    async def _on_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        resp = self._response
        if kind == "seg_start" and resp is not None:
            seg_id = msg.get("seg")
            if seg_id in resp.segments:
                resp.started.add(seg_id)
                seg = resp.segments[seg_id]
                self.position = seg.position
                self.style = seg.style
        elif kind == "seg_end" and resp is not None:
            seg_id = msg.get("seg")
            if seg_id in resp.segments:
                resp.started.add(seg_id)
                resp.done.add(seg_id)
                await self._check_response_done(resp)
        elif kind == "stop":
            self._closed.set()

    async def _transcribe(self, audio: bytes, seq: int) -> None:
        try:
            text = (await self.services.stt.transcribe(audio)).strip()
        except Exception:  # noqa: BLE001
            log.exception("STT 失败")
            text = ""
        if text:
            self._pending_user.append(text)
            await self._send_json({"type": "transcript", "text": text})
        if seq != self._turn_seq:
            return  # 转写期间用户又开口了：这段并入下一轮
        await self._apply(self.sm.user_turn_end(self.clock(), " ".join(self._pending_user)))

    # ---- 动作执行 ---------------------------------------------------------

    async def _apply(self, actions: list[Action]) -> None:
        for action in actions:
            if isinstance(action, Interrupt):
                await self._interrupt()
            elif isinstance(action, Respond):
                await self._maybe_send_filler()
                self._start_response(
                    Response(
                        kind="reply",
                        user_parts=list(self._pending_user),
                        gentle=action.gentle,
                        style_override=LIGHTEST_STYLE if action.gentle else None,
                    )
                )
            elif isinstance(action, StartMonologue):
                self._start_response(
                    Response(
                        kind="monologue",
                        prompt=MONOLOGUE_PROMPT.format(n="1–2" if action.max_sentences > 1 else "1"),
                        gain_db=action.gain_db,
                    )
                )
            elif isinstance(action, EnterSleep):
                log.info("进入入睡模式")
            elif isinstance(action, EndSession):
                await self._send_json({"type": "end", "fade_secs": END_FADE_SECS})
                self._spawn(self._close_later(END_FADE_SECS + 2))
        await self._sync_state()

    async def _close_later(self, delay: float) -> None:
        await asyncio.sleep(delay)
        self._closed.set()
        with contextlib.suppress(Exception):
            await self.transport.close()

    async def _sync_state(self) -> None:
        if self.sm.state != self._sent_state:
            self._sent_state = self.sm.state
            await self._send_json({"type": "state", "state": self.sm.state.value})

    async def _interrupt(self) -> None:
        # 先取消（不让旧回复再发出任何段落），再通知前端淡出。
        if self._response is not None and self._response.task is not None:
            self._response.task.cancel()
        await self._send_json({"type": "interrupt"})
        await self._cancel_response()

    async def _cancel_response(self) -> None:
        resp, self._response = self._response, None
        if resp is None:
            return
        if resp.task is not None:
            resp.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await resp.task
        self._commit(resp, interrupted=True)

    async def _maybe_send_filler(self) -> None:
        fillers = self.services.fillers
        if not (self.settings.fillers and fillers) or random.random() > self.settings.filler_probability:
            return
        pcm = fillers.pick()
        if pcm is None:
            return
        seg_id = self._next_seg()
        await self._send_json({"type": "segment", "seg": seg_id, "kind": "filler", "text": "", "cues": [], "gain_db": 0})
        await self._send_bytes(struct.pack("<I", seg_id) + self._leveler.process(pcm))
        await self._send_json({"type": "segment_end", "seg": seg_id})

    # ---- 回复生成 ---------------------------------------------------------

    def _start_response(self, resp: Response) -> None:
        if self._response is not None:
            old = self._response
            self._response = None
            if old.task is not None:
                old.task.cancel()
            self._commit(old, interrupted=True)
        self._response = resp
        resp.task = self._spawn(self._run_response(resp))

    async def _run_response(self, resp: Response) -> None:
        parser = TagParser(style=self.style, position=self.position, sfx_whitelist=self.settings.sfx)
        segmenter = Segmenter(position=self.position, whole=resp.kind == "monologue")
        queue: asyncio.Queue[Segment | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                async for delta in self._text_source(resp):
                    for event in parser.feed(delta):
                        for seg in self._segment(segmenter, event, resp):
                            await queue.put(seg)
                for event in parser.flush():
                    for seg in self._segment(segmenter, event, resp):
                        await queue.put(seg)
                for seg in segmenter.flush():
                    await queue.put(seg)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while (seg := await queue.get()) is not None:
                await self._play_segment(resp, seg)
            await producer
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("生成回复失败")
        finally:
            producer.cancel()
            with contextlib.suppress(BaseException):
                await producer
        resp.generation_done = True
        await self._check_response_done(resp)

    def _segment(self, segmenter: Segmenter, event, resp: Response) -> list[Segment]:
        if isinstance(event, Sentence) and resp.kind == "monologue" and CHECK_IN_RE.search(event.plain):
            log.info("独白中丢弃追问句: %s", event.plain)
            return []
        return segmenter.push(event)

    async def _text_source(self, resp: Response) -> AsyncIterator[str]:
        if resp.kind == "intro" and resp.prompt is None:
            yield self.character.render(self.character.first_mes)
            return
        async for delta in self.services.llm.stream(self._build_messages(resp)):
            yield delta

    def _build_messages(self, resp: Response) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.history[-self.settings.history_messages :]
        user = self._user_message(resp)
        if user:
            if messages[-1]["role"] == "user":
                messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n" + user}
            else:
                messages.append({"role": "user", "content": user})
        return messages

    @staticmethod
    def _user_message(resp: Response) -> str:
        if resp.kind == "reply":
            return resp.user_text + (f"\n{WAKE_HINT}" if resp.gentle else "")
        return resp.prompt or ""

    async def _play_segment(self, resp: Response, seg: Segment) -> None:
        seg_id = self._next_seg()
        resp.order.append(seg_id)
        resp.segments[seg_id] = seg
        style = resp.style_override or seg.style
        await self._send_json(
            {
                "type": "segment",
                "seg": seg_id,
                "kind": resp.kind,
                "text": seg.subtitle,
                "style": style,
                "cues": [cue.to_json() for cue in seg.cues],
                "gain_db": resp.gain_db,
            }
        )
        if seg.text:
            try:
                async for pcm in self.services.tts.synthesize(seg.text, style):
                    await self._send_bytes(struct.pack("<I", seg_id) + self._leveler.process(pcm))
                    if not resp.audio_started:
                        resp.audio_started = True
                        await self._apply(self.sm.response_audio_started(self.clock()))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # 内容过滤、网络错误等：跳过这一段，会话继续。
                log.warning("TTS 失败，跳过该段: %s (%s)", seg.subtitle, e)
        await self._send_json({"type": "segment_end", "seg": seg_id})

    async def _check_response_done(self, resp: Response) -> None:
        if resp is not self._response or not resp.generation_done:
            return
        if any(seg_id not in resp.done for seg_id in resp.order):
            return
        self._response = None
        self._commit(resp, interrupted=False)
        await self._apply(self.sm.playback_finished(self.clock()))

    # ---- 历史 -------------------------------------------------------------

    def _commit(self, resp: Response, *, interrupted: bool) -> None:
        """只把实际播出的段落写进历史。一句都没播出时什么也不写，用户的话留到下一轮。"""
        started = [seg_id for seg_id in resp.order if seg_id in resp.started]
        if not started:
            return
        user = self._user_message(resp)
        if user:
            self._add_history("user", user)
        texts = [resp.segments[seg_id].render() for seg_id in started]
        if interrupted and started[-1] not in resp.done and resp.segments[started[-1]].text:
            texts[-1] += "——"
        assistant = "\n".join(t for t in texts if t)
        if assistant:
            self._add_history("assistant", assistant)
        if resp.kind == "reply":
            self._pending_user = self._pending_user[len(resp.user_parts) :]

    def _add_history(self, role: str, content: str) -> None:
        if self.history and self.history[-1]["role"] == role:
            self.history[-1] = {"role": role, "content": self.history[-1]["content"] + "\n" + content}
        else:
            self.history.append({"role": role, "content": content})
        limit = self.settings.history_messages * 2
        if len(self.history) > limit:
            del self.history[: len(self.history) - limit]

    # ---- 发送 -------------------------------------------------------------

    def _next_seg(self) -> int:
        self._seg_counter += 1
        return self._seg_counter

    async def _send_json(self, data: dict) -> None:
        if self._closed.is_set():
            return
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.transport.send_json(data)

    async def _send_bytes(self, data: bytes) -> None:
        if self._closed.is_set():
            return
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.transport.send_bytes(data)
