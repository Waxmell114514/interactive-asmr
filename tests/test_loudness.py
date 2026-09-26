import numpy as np

from asmr.loudness import LoudnessLeveler


def tone(db, secs=3.0, sr=24000):
    t = np.arange(int(secs * sr)) / sr
    x = np.sin(2 * np.pi * 220 * t) * np.sqrt(2) * 10 ** (db / 20)
    return (x * 32767).astype(np.int16).tobytes()


def rms_db(pcm):
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    return 20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9)


def level(pcm, leveler, chunk=4800):
    return b"".join(leveler.process(pcm[i : i + chunk]) for i in range(0, len(pcm), chunk))


def test_quiet_whisper_and_loud_speech_end_up_close():
    lev = LoudnessLeveler()
    quiet = level(tone(-38), lev)
    lev = LoudnessLeveler()
    loud = level(tone(-14), lev)
    tail = 24000 * 2  # 最后 1 秒，已收敛
    assert abs(rms_db(quiet[-tail:]) - (-26)) < 1.5  # 最多提升 12dB
    assert abs(rms_db(loud[-tail:]) - (-24)) < 1.5


def test_never_clips():
    lev = LoudnessLeveler()
    x = np.frombuffer(tone(0), dtype=np.int16)
    out = np.frombuffer(level(x.tobytes(), lev), dtype=np.int16)
    assert np.max(np.abs(out)) < 32000  # 起始未收敛时也只是软削波，不会顶到满幅
    assert np.max(np.abs(out[-24000:])) <= 0.9 * 32767  # 收敛后峰值在 -1dBFS 以下


def test_silence_is_not_boosted():
    lev = LoudnessLeveler()
    silence = (np.random.default_rng(0).normal(0, 1e-4, 24000) * 32767).astype(np.int16).tobytes()
    assert rms_db(level(silence, lev)) < -70
