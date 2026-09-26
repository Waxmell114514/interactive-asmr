"""会话状态机（纯逻辑，不做 IO；时间由调用方传入，便于测试）。

用户不说话是正常状态：沉默 T1 后进入独白，每次独白后 T1 翻倍（上限 60s）；
连续 N 次独白无回应或累计沉默超过阈值，进入 Sleep；Sleep 满定时后淡出结束。

    Intro → Waiting → Listening → Thinking → Speaking → Waiting
    Waiting → Monologue → Waiting / Listening
    Waiting → Sleep → Listening / 结束
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class State(str, Enum):
    INTRO = "intro"
    WAITING = "waiting"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    MONOLOGUE = "monologue"
    SLEEP = "sleep"
    ENDED = "ended"


@dataclass
class Timing:
    t1: float = 12.0  # 沉默转独白
    t1_max: float = 60.0
    monologues_to_sleep: int = 5  # N
    silence_to_sleep: float = 600.0  # 累计沉默 10 分钟
    sleep_timer: float = 1200.0  # 入睡后 20 分钟淡出
    monologue_decay_db: float = 2.0  # 每轮独白 −2dB


# ---- 状态机发出的动作，由 Session 执行 ------------------------------------


@dataclass(frozen=True)
class Interrupt:
    """用户开口：前端淡出清空队列，后端取消未完成的 LLM / TTS。"""


@dataclass(frozen=True)
class Respond:
    text: str
    gentle: bool = False  # 从 Sleep 被唤醒：首个回复强制最轻的 style


@dataclass(frozen=True)
class StartMonologue:
    round: int
    gain_db: float
    max_sentences: int


@dataclass(frozen=True)
class EnterSleep:
    pass


@dataclass(frozen=True)
class EndSession:
    pass


Action = Interrupt | Respond | StartMonologue | EnterSleep | EndSession


class SessionStateMachine:
    def __init__(self, timing: Timing | None = None):
        self.timing = timing or Timing()
        self.state = State.INTRO
        self._t1 = self.timing.t1
        self._waiting_since = 0.0
        self._last_user_activity = 0.0
        self._sleep_since = 0.0
        self._resting = State.WAITING  # 进入 Listening 前的静息状态（Waiting / Sleep）
        self._woke_from_sleep = False
        self.monologue_count = 0

    # ---- 事件 -------------------------------------------------------------

    def start(self, now: float) -> list[Action]:
        self.state = State.INTRO
        self._last_user_activity = now
        return []

    def user_speech_start(self, now: float) -> list[Action]:
        if self.state == State.ENDED:
            return []
        actions: list[Action] = []
        if self.state in (State.INTRO, State.THINKING, State.SPEAKING, State.MONOLOGUE):
            actions.append(Interrupt())
        if self.state == State.SLEEP:
            self._woke_from_sleep = True
        if self.state in (State.WAITING, State.SLEEP):
            self._resting = self.state
        elif self.state != State.LISTENING:
            self._resting = State.WAITING
        self.state = State.LISTENING
        return actions

    def user_turn_end(self, now: float, text: str) -> list[Action]:
        if self.state != State.LISTENING:
            return []
        if not text.strip():
            # 咳嗽、翻身之类被 VAD 误判的声音：回到原来的静息状态，不回应，也不重置沉默计数。
            self._woke_from_sleep = False
            self._rest(now, self._resting)
            return []
        self._last_user_activity = now
        self.monologue_count = 0
        self._t1 = self.timing.t1
        self.state = State.THINKING
        gentle, self._woke_from_sleep = self._woke_from_sleep, False
        return [Respond(text, gentle=gentle)]

    def response_audio_started(self, now: float) -> list[Action]:
        if self.state == State.THINKING:
            self.state = State.SPEAKING
        return []

    def playback_finished(self, now: float) -> list[Action]:
        """当前回复（开场白 / 回复 / 独白）已经全部播完，或者根本没产生音频。"""
        if self.state in (State.INTRO, State.THINKING, State.SPEAKING, State.MONOLOGUE):
            self._rest(now, State.WAITING)
            return self._check_sleep(now)
        return []

    def tick(self, now: float) -> list[Action]:
        if self.state == State.WAITING:
            actions = self._check_sleep(now)
            if actions:
                return actions
            if now - self._waiting_since >= self._t1:
                self.monologue_count += 1
                self._t1 = min(self._t1 * 2, self.timing.t1_max)
                self.state = State.MONOLOGUE
                n = self.monologue_count
                return [
                    StartMonologue(
                        round=n,
                        gain_db=-self.timing.monologue_decay_db * n,
                        max_sentences=2 if n <= 2 else 1,
                    )
                ]
        elif self.state == State.SLEEP:
            if now - self._sleep_since >= self.timing.sleep_timer:
                self.state = State.ENDED
                return [EndSession()]
        return []

    def stop(self) -> None:
        self.state = State.ENDED

    # ---- 内部 -------------------------------------------------------------

    def _rest(self, now: float, state: State) -> None:
        self.state = state
        if state == State.WAITING:
            self._waiting_since = now

    def _check_sleep(self, now: float) -> list[Action]:
        if self.state != State.WAITING:
            return []
        if (
            self.monologue_count >= self.timing.monologues_to_sleep
            or now - self._last_user_activity >= self.timing.silence_to_sleep
        ):
            self.state = State.SLEEP
            self._sleep_since = now
            return [EnterSleep()]
        return []
