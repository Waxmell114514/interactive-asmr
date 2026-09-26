"""响度归一化（X7）：慢速自动增益，把 TTS 输出拉到统一电平，再由前端限幅器兜底。

耳语和正常说话的原始电平能差 15dB 以上；这里用较慢的时间常数跟踪 RMS，
只做温和的整体调平，不压扁句内动态。极静的片段（停顿、气声尾巴）不参与估计，
避免把底噪放大。
"""

from __future__ import annotations

import numpy as np


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


class LoudnessLeveler:
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        target_dbfs: float = -24.0,
        max_boost_db: float = 12.0,
        max_cut_db: float = 12.0,
        gate_dbfs: float = -50.0,
        time_constant: float = 1.5,
        block_ms: float = 20.0,
        ceiling: float = 0.89,  # -1 dBFS
    ):
        self._target = target_dbfs
        self._max_boost = max_boost_db
        self._max_cut = max_cut_db
        self._gate = gate_dbfs
        self._block = max(1, int(sample_rate * block_ms / 1000))
        self._alpha = 1 - np.exp(-(block_ms / 1000) / time_constant)
        self._ceiling = ceiling
        self._level_db: float | None = None
        self._gain = 1.0

    def process(self, pcm16: bytes) -> bytes:
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        out = np.empty_like(x)
        for start in range(0, len(x), self._block):
            block = x[start : start + self._block]
            rms = float(np.sqrt(np.mean(block * block))) if len(block) else 0.0
            block_db = 20 * np.log10(rms + 1e-9)
            if block_db > self._gate:
                if self._level_db is None:
                    self._level_db = block_db
                else:
                    self._level_db += self._alpha * (block_db - self._level_db)
            if self._level_db is not None:
                want = np.clip(self._target - self._level_db, -self._max_cut, self._max_boost)
                target_gain = db_to_gain(want)
            else:
                target_gain = self._gain
            ramp = np.linspace(self._gain, target_gain, len(block), endpoint=False, dtype=np.float32)
            self._gain = target_gain
            out[start : start + len(block)] = block * ramp
        # 软削波：超过天花板的部分用 tanh 收住，杜绝爆音。
        over = np.abs(out) > self._ceiling
        if over.any():
            head = 1 - self._ceiling
            out[over] = np.sign(out[over]) * (self._ceiling + head * np.tanh((np.abs(out[over]) - self._ceiling) / head))
        return (np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()
