from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VADState

from asmr.turn import SpeechStarted, TurnDetector, TurnEnded, create_turn_detector

CHUNK = b"\x00\x00" * 320  # 20ms @16kHz


class ScriptedVAD:
    def __init__(self, states):
        self.states = list(states)

    async def analyze_audio(self, chunk):
        return self.states.pop(0) if self.states else VADState.QUIET


class ScriptedTurn:
    """analyze_end_of_turn 依次返回给定结果；append_audio 从不兜底。"""

    def __init__(self, results):
        self.results = list(results)
        self.cleared = 0

    def append_audio(self, chunk, is_speech):
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self):
        return self.results.pop(0), None

    def clear(self):
        self.cleared += 1


Q, ST, SP, STOP = VADState.QUIET, VADState.STARTING, VADState.SPEAKING, VADState.STOPPING


async def feed(detector, n):
    events = []
    for _ in range(n):
        events += await detector.process(CHUNK)
    return events


async def test_pause_judged_incomplete_does_not_end_turn():
    vad = ScriptedVAD([Q, ST, SP, SP, STOP, Q, Q, ST, SP, STOP, Q])
    turn = ScriptedTurn([EndOfTurnState.INCOMPLETE, EndOfTurnState.COMPLETE])
    det = TurnDetector(vad=vad, turn_analyzer=turn)
    events = await feed(det, 11)
    assert events[0] == SpeechStarted()
    assert len(events) == 2 and isinstance(events[1], TurnEnded)
    # 预录 + 整个轮次（含中间停顿）都交给 STT
    assert len(events[1].audio) == 11 * len(CHUNK)
    assert turn.cleared == 1


async def test_silence_fallback_without_smart_turn():
    vad = ScriptedVAD([SP, SP, STOP] + [Q] * 200)
    det = TurnDetector(vad=vad, turn_analyzer=None, stop_secs=1.0)
    events = await feed(det, 2 + 49)  # 第 3 块起算静音，共 49 块
    assert events == [SpeechStarted()]  # 静音还差一点才到 1 秒
    events = await feed(det, 1)
    assert len(events) == 1 and isinstance(events[0], TurnEnded)


async def test_pre_roll_is_bounded():
    vad = ScriptedVAD([Q] * 100 + [SP, STOP, Q])
    det = TurnDetector(vad=vad, turn_analyzer=ScriptedTurn([EndOfTurnState.COMPLETE]), pre_roll_secs=0.5)
    events = await feed(det, 103)
    audio = events[-1].audio
    assert len(audio) <= (25 + 3 + 1) * len(CHUNK)


async def test_real_models_load_and_ignore_silence():
    det = create_turn_detector(confidence=0.6, start_secs=0.2, stop_secs=0.2, turn_stop_secs=3.0, smart_turn=True)
    assert await feed(det, 50) == []
