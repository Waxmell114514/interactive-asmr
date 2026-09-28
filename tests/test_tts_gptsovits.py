import io
import json
import wave

import httpx
import numpy as np
import pytest

from asmr.tts import TTSError, synthesize_all
from asmr.tts_gptsovits import GPTSoVITSConfig, GPTSoVITSTTS, prepare_text

CONFIG = {
    "refs": [
        {"match": ["耳语", "气声"], "audio": "refs/whisper.wav", "text": "悄悄告诉你。"},
        {"match": ["笑"], "audio": "refs/smile.wav", "text": "嘿嘿。"},
        {"audio": "refs/normal.wav", "text": "你好呀。", "default": True},
    ],
    "speed_factor": 0.9,
}


def wav_header(rate: int, frames: bytes = b"") -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return buf.getvalue()


def make_tts(handler, **overrides):
    cfg = GPTSoVITSConfig.from_dict({**CONFIG, **overrides})
    return GPTSoVITSTTS(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def chunked(*parts: bytes):
    async def gen():
        for part in parts:
            yield part

    return gen()


def test_prepare_text_removes_vocal_tags():
    assert prepare_text("嗯……<short pause> 今天辛苦了。<sigh> 闭上眼睛。") == "嗯……， 今天辛苦了。 闭上眼睛。"
    assert prepare_text("<long pause> 晚安") == "…… 晚安"
    assert prepare_text("<sigh>") == ""
    assert prepare_text("……") == ""


def test_pick_ref_by_style_keyword():
    cfg = GPTSoVITSConfig.from_dict(CONFIG)
    assert cfg.pick_ref("耳语，很轻").audio == "refs/whisper.wav"
    assert cfg.pick_ref("极轻的耳语，很慢，几乎只有气声").audio == "refs/whisper.wav"  # 入睡唤醒用的 style
    assert cfg.pick_ref("轻笑，温柔").audio == "refs/smile.wav"
    assert cfg.pick_ref("").audio == "refs/normal.wav"


def test_config_requires_refs():
    with pytest.raises(ValueError):
        GPTSoVITSConfig.from_dict({"refs": []})


async def test_streaming_wav_is_resampled_to_24k():
    seen = {}
    pcm = (np.sin(np.arange(32000) / 10) * 8000).astype(np.int16).tobytes()  # 1 秒 @32kHz（v2Pro）
    body = wav_header(32000) + pcm

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        # 故意把 WAV 头和奇数字节切碎
        return httpx.Response(200, content=chunked(body[:10], body[10:47], body[47:1001], body[1001:]))

    tts = make_tts(handler)
    out = await synthesize_all(tts, "嗯……<sigh> 睡吧。", "耳语")
    assert abs(len(out) / 2 - 24000) < 50
    assert seen["url"].endswith("/tts")
    assert seen["body"]["text"] == "嗯…… 睡吧。"
    assert seen["body"]["ref_audio_path"] == "refs/whisper.wav"
    assert seen["body"]["prompt_text"] == "悄悄告诉你。"
    assert seen["body"]["media_type"] == "wav"
    assert seen["body"]["speed_factor"] == 0.9


async def test_non_streaming_wav_at_24k_passes_through():
    pcm = b"\x01\x00" * 2400
    tts = make_tts(lambda r: httpx.Response(200, content=wav_header(24000, pcm)), streaming_mode=0)
    assert await synthesize_all(tts, "你好。") == pcm


async def test_only_tags_skips_request():
    def handler(request):
        raise AssertionError("不应该发请求")

    assert await synthesize_all(make_tts(handler), "<sigh>") == b""


async def test_http_error_raises():
    tts = make_tts(lambda r: httpx.Response(400, json={"message": "ref audio not found"}))
    with pytest.raises(TTSError, match="ref audio not found"):
        await synthesize_all(tts, "你好。")


async def test_switches_weights_once():
    calls = []

    def handler(request: httpx.Request):
        calls.append(request.url.path)
        if request.url.path == "/tts":
            return httpx.Response(200, content=wav_header(24000, b"\x00\x00" * 10))
        return httpx.Response(200, text="success")

    tts = make_tts(handler, gpt_weights="g.ckpt", sovits_weights="s.pth")
    await synthesize_all(tts, "一。")
    await synthesize_all(tts, "二。")
    assert calls == ["/set_gpt_weights", "/set_sovits_weights", "/tts", "/tts"]
