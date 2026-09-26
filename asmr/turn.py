"""轮次检测：Silero VAD 判断有没有声音，Smart Turn v3 判断是否说完。

两个模型都用 Pipecat 附带的实现（ONNX，CPU 几十毫秒）。VAD 静音 ``vad_stop_secs`` 后请 Smart Turn
判定；判定“没说完”就继续等，直到静音达到 ``turn_stop_secs`` 兜底结束——慢语速、句中长停顿不会被抢话。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechStarted:
    pass


@dataclass(frozen=True)
class TurnEnded:
    audio: bytes  # 整个轮次的 16kHz PCM（含开口前的预录）


TurnEvent = SpeechStarted | TurnEnded


class TurnDetector:
    def __init__(
        self,
        *,
        vad,
        turn_analyzer=None,
        stop_secs: float = 3.0,
        pre_roll_secs: float = 0.5,
        max_turn_secs: float = 30.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        from pipecat.audio.vad.vad_analyzer import VADState

        self._VADState = VADState
        self._vad = vad
        self._turn = turn_analyzer
        self._stop_secs = stop_secs
        self._max_bytes = int(max_turn_secs * sample_rate) * 2
        self._pre_roll_bytes = int(pre_roll_secs * sample_rate) * 2
        self._bytes_per_sec = sample_rate * 2
        self._pre_roll: deque[bytes] = deque()
        self._pre_roll_len = 0
        self._audio: list[bytes] = []
        self._audio_len = 0
        self._in_turn = False
        self._speaking = False
        self._silence_bytes = 0

    async def process(self, chunk: bytes) -> list[TurnEvent]:
        S = self._VADState
        state = await self._vad.analyze_audio(chunk)
        is_speech = state in (S.STARTING, S.SPEAKING)
        events: list[TurnEvent] = []

        if self._in_turn:
            self._audio.append(chunk)
            self._audio_len += len(chunk)
        else:
            self._pre_roll.append(chunk)
            self._pre_roll_len += len(chunk)
            while self._pre_roll_len - len(self._pre_roll[0]) >= self._pre_roll_bytes:
                self._pre_roll_len -= len(self._pre_roll.popleft())

        end = False
        if state == S.SPEAKING and not self._speaking:
            self._speaking = True
            self._silence_bytes = 0
            if not self._in_turn:
                self._in_turn = True
                self._audio = list(self._pre_roll)
                self._audio_len = self._pre_roll_len
                self._pre_roll.clear()
                self._pre_roll_len = 0
                events.append(SpeechStarted())

        if self._turn is not None:
            from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

            if self._turn.append_audio(chunk, is_speech) == EndOfTurnState.COMPLETE and self._in_turn:
                end = True  # 静音达到 stop_secs 兜底
            if state == S.QUIET and self._speaking:
                self._speaking = False
                result, _ = await self._turn.analyze_end_of_turn()
                if result == EndOfTurnState.COMPLETE:
                    end = True
        else:
            if state == S.QUIET and self._speaking:
                self._speaking = False
            if self._in_turn and not is_speech:
                self._silence_bytes += len(chunk)
                if self._silence_bytes >= self._stop_secs * self._bytes_per_sec:
                    end = True
            else:
                self._silence_bytes = 0

        if self._in_turn and self._audio_len >= self._max_bytes:
            log.info("轮次超过最大时长，强制结束")
            end = True

        if end and self._in_turn:
            events.append(TurnEnded(b"".join(self._audio)))
            self._reset_turn()
        return events

    def _reset_turn(self) -> None:
        self._in_turn = False
        self._speaking = False
        self._audio = []
        self._audio_len = 0
        self._silence_bytes = 0
        if self._turn is not None:
            self._turn.clear()


def create_turn_detector(
    *,
    confidence: float,
    start_secs: float,
    stop_secs: float,
    turn_stop_secs: float,
    smart_turn: bool,
) -> TurnDetector:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    vad = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=VADParams(confidence=confidence, start_secs=start_secs, stop_secs=stop_secs, min_volume=0.3),
    )
    vad.set_sample_rate(SAMPLE_RATE)
    analyzer = None
    if smart_turn:
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        analyzer = LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=turn_stop_secs))
        analyzer.set_sample_rate(SAMPLE_RATE)
        analyzer.update_vad_start_secs(start_secs)
    return TurnDetector(vad=vad, turn_analyzer=analyzer, stop_secs=turn_stop_secs)
