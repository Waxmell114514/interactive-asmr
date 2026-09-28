"""运行配置：全部来自环境变量（可写在项目根目录的 .env 里）。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .state import Timing
from .tags import DEFAULT_SFX

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _float(name: str, default: float) -> float:
    value = _env(name)
    return float(value) if value else default


def _int(name: str, default: int) -> int:
    value = _env(name)
    return int(value) if value else default


def _bool(name: str, default: bool) -> bool:
    value = _env(name).lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000

    # LLM：任意 OpenAI 兼容 API
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 300
    history_messages: int = 40

    # TTS 后端：gemini | gptsovits
    tts_backend: str = "gemini"
    gptsovits_url: str = "http://127.0.0.1:9880"
    gptsovits_config: Path | None = None  # 角色卡没写 gptsovits 配置时用这个 JSON

    # TTS：Gemini 3.8 Flash-Lite TTS + Voice Design 音色
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    tts_model: str = "gemini-3.8-flash-lite-tts"
    tts_voice: str = "Achernar"  # voice_...（Voice Design）或预置音色名
    tts_style_mode: str = "annotation"  # annotation | prefix | none，见 M0 对比

    # STT
    stt_backend: str = "sensevoice"  # sensevoice | whisper
    stt_device: str = "cpu"  # cpu | cuda
    stt_model: str = ""  # 留空用各后端默认模型
    stt_language: str = "zh"

    # 轮次检测
    vad_confidence: float = 0.6
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2
    turn_stop_secs: float = 3.0  # Smart Turn 判定“没说完”时，静音多久兜底结束
    smart_turn: bool = True

    # 体验
    fillers: bool = True
    filler_probability: float = 0.7
    intro_delay: float = 3.0  # 等环境音淡入后再开场
    sfx: tuple[str, ...] = DEFAULT_SFX
    timing: Timing = field(default_factory=Timing)

    characters_dir: Path = ROOT / "characters"
    data_dir: Path = ROOT / "data"
    web_dir: Path = ROOT / "web"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or ROOT / ".env")
        timing = Timing(
            t1=_float("ASMR_T1", 12.0),
            t1_max=_float("ASMR_T1_MAX", 60.0),
            monologues_to_sleep=_int("ASMR_MONOLOGUES_TO_SLEEP", 5),
            silence_to_sleep=_float("ASMR_SILENCE_TO_SLEEP", 600.0),
            sleep_timer=_float("ASMR_SLEEP_TIMER", 1200.0),
            monologue_decay_db=_float("ASMR_MONOLOGUE_DECAY_DB", 2.0),
        )
        d = cls()
        return cls(
            host=_env("ASMR_HOST", d.host),
            port=_int("ASMR_PORT", d.port),
            llm_base_url=_env("LLM_BASE_URL", d.llm_base_url),
            llm_api_key=_env("LLM_API_KEY"),
            llm_model=_env("LLM_MODEL", d.llm_model),
            llm_temperature=_float("LLM_TEMPERATURE", d.llm_temperature),
            llm_max_tokens=_int("LLM_MAX_TOKENS", d.llm_max_tokens),
            history_messages=_int("LLM_HISTORY_MESSAGES", d.history_messages),
            tts_backend=_env("TTS_BACKEND", d.tts_backend),
            gptsovits_url=_env("GPTSOVITS_URL", d.gptsovits_url),
            gptsovits_config=Path(_env("GPTSOVITS_CONFIG")) if _env("GPTSOVITS_CONFIG") else None,
            gemini_api_key=_env("GEMINI_API_KEY"),
            gemini_base_url=_env("GEMINI_BASE_URL", d.gemini_base_url),
            tts_model=_env("TTS_MODEL", d.tts_model),
            tts_voice=_env("TTS_VOICE", d.tts_voice),
            tts_style_mode=_env("TTS_STYLE_MODE", d.tts_style_mode),
            stt_backend=_env("STT_BACKEND", d.stt_backend),
            stt_device=_env("STT_DEVICE", d.stt_device),
            stt_model=_env("STT_MODEL"),
            stt_language=_env("STT_LANGUAGE", d.stt_language),
            vad_confidence=_float("VAD_CONFIDENCE", d.vad_confidence),
            vad_start_secs=_float("VAD_START_SECS", d.vad_start_secs),
            vad_stop_secs=_float("VAD_STOP_SECS", d.vad_stop_secs),
            turn_stop_secs=_float("TURN_STOP_SECS", d.turn_stop_secs),
            smart_turn=_bool("SMART_TURN", d.smart_turn),
            fillers=_bool("ASMR_FILLERS", d.fillers),
            filler_probability=_float("ASMR_FILLER_PROBABILITY", d.filler_probability),
            intro_delay=_float("ASMR_INTRO_DELAY", d.intro_delay),
            timing=timing,
        )
