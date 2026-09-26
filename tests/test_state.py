from asmr.state import (
    EndSession,
    EnterSleep,
    Interrupt,
    Respond,
    SessionStateMachine,
    StartMonologue,
    State,
    Timing,
)


def waiting_machine(now=0.0):
    sm = SessionStateMachine(Timing())
    sm.start(now)
    sm.playback_finished(now)  # 开场白播完
    assert sm.state == State.WAITING
    return sm


def test_normal_turn():
    sm = waiting_machine()
    assert sm.user_speech_start(1) == []
    assert sm.state == State.LISTENING
    assert sm.user_turn_end(2, "你好") == [Respond("你好")]
    assert sm.state == State.THINKING
    sm.response_audio_started(3)
    assert sm.state == State.SPEAKING
    sm.playback_finished(5)
    assert sm.state == State.WAITING


def test_barge_in_interrupts_speaking_intro_and_thinking():
    for setup in ("intro", "thinking", "speaking"):
        sm = SessionStateMachine()
        sm.start(0)
        if setup != "intro":
            sm.playback_finished(0)
            sm.user_speech_start(1)
            sm.user_turn_end(2, "嗯")
            if setup == "speaking":
                sm.response_audio_started(3)
        assert sm.user_speech_start(4) == [Interrupt()]
        assert sm.state == State.LISTENING


def test_empty_transcript_returns_to_rest_without_reply():
    sm = waiting_machine()
    sm.user_speech_start(1)
    assert sm.user_turn_end(2, "  ") == []
    assert sm.state == State.WAITING


def test_monologue_timing_doubles_and_caps():
    sm = waiting_machine(0)
    starts = []
    now = 0.0
    while len(starts) < 4:
        now += 0.5
        for action in sm.tick(now):
            assert isinstance(action, StartMonologue)
            starts.append((now, action))
            sm.playback_finished(now)  # 独白瞬间播完
    times = [t for t, _ in starts]
    gaps = [times[0]] + [b - a for a, b in zip(times, times[1:])]
    assert gaps == [12, 24, 48, 60]
    assert [a.gain_db for _, a in starts] == [-2, -4, -6, -8]
    assert [a.max_sentences for _, a in starts] == [2, 2, 1, 1]


def test_no_question_state_after_silence_goes_to_sleep_after_n_monologues():
    sm = waiting_machine(0)
    now, count = 0.0, 0
    actions = []
    while sm.state != State.SLEEP:
        now += 1
        actions = sm.tick(now)
        if any(isinstance(a, StartMonologue) for a in actions):
            count += 1
            sm.playback_finished(now)
            actions = sm.tick(now)
    assert count == 5
    assert sm.state == State.SLEEP


def test_accumulated_silence_triggers_sleep():
    sm = SessionStateMachine(Timing(t1=10_000, silence_to_sleep=600))
    sm.start(0)
    sm.playback_finished(0)
    assert sm.tick(599) == []
    assert sm.tick(600) == [EnterSleep()]


def test_sleep_timer_ends_session():
    sm = SessionStateMachine(Timing(t1=10_000, silence_to_sleep=10, sleep_timer=100))
    sm.start(0)
    sm.playback_finished(0)
    assert sm.tick(10) == [EnterSleep()]
    assert sm.tick(109) == []
    assert sm.tick(110) == [EndSession()]
    assert sm.state == State.ENDED


def test_wake_from_sleep_is_gentle_once():
    sm = SessionStateMachine(Timing(t1=10_000, silence_to_sleep=10))
    sm.start(0)
    sm.playback_finished(0)
    sm.tick(10)
    assert sm.state == State.SLEEP
    assert sm.user_speech_start(20) == []
    assert sm.user_turn_end(21, "还没睡") == [Respond("还没睡", gentle=True)]
    sm.playback_finished(25)
    sm.user_speech_start(30)
    assert sm.user_turn_end(31, "嗯") == [Respond("嗯", gentle=False)]


def test_noise_during_sleep_stays_asleep():
    sm = SessionStateMachine(Timing(t1=10_000, silence_to_sleep=10))
    sm.start(0)
    sm.playback_finished(0)
    sm.tick(10)
    sm.user_speech_start(20)
    assert sm.user_turn_end(21, "") == []
    assert sm.state == State.SLEEP


def test_user_reply_resets_monologue_backoff():
    sm = waiting_machine(0)
    assert isinstance(sm.tick(12)[0], StartMonologue)
    sm.playback_finished(13)
    sm.user_speech_start(14)
    sm.user_turn_end(15, "嗯")
    sm.playback_finished(16)
    assert sm.monologue_count == 0
    assert sm.tick(27.9) == []
    assert isinstance(sm.tick(28)[0], StartMonologue)
