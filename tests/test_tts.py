import base64
import json

import httpx
import pytest

from asmr.tts import GeminiTTS, TTSError, build_request, synthesize_all


def sse(*events):
    lines = []
    for event in events:
        lines.append(f"event: {event.get('event_type', 'message')}")
        lines.append(f"data: {json.dumps(event)}")
        lines.append("")
    lines += ["event: done", "data: [DONE]", ""]
    return "\n".join(lines) + "\n"


def audio(data: bytes):
    return {"index": 0, "delta": {"type": "audio", "data": base64.b64encode(data).decode()}, "event_type": "step.delta"}


def make_tts(handler, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiTTS(api_key="k", voice="voice_abc", client=client, **kwargs)


def test_style_goes_to_speech_metadata_not_text():
    body = build_request("gemini-3.8-flash-lite-tts", "voice_abc", "你好。", "耳语", "annotation")
    content = body["input"][0]["content"][0]
    assert content["text"] == "你好。"
    assert content["annotations"] == [{"type": "speech_metadata", "style": "耳语"}]
    assert body["generation_config"]["speech_config"] == [{"voice": "voice_abc"}]
    assert body["response_format"]["mime_type"] == "audio/l16"
    assert body["stream"] is True


def test_prefix_and_empty_style():
    assert build_request("m", "v", "好。", "轻", "prefix")["input"][0]["content"][0]["text"] == "轻：好。"
    assert "annotations" not in build_request("m", "v", "好。", "", "annotation")["input"][0]["content"][0]


async def test_streams_pcm_and_keeps_sample_alignment():
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers["x-goog-api-key"]
        stream = sse(
            {"event_type": "interaction.created", "interaction": {"id": "x"}},
            {"index": 0, "step": {"type": "model_output"}, "event_type": "step.start"},
            audio(b"\x01\x02\x03"),
            audio(b"\x04\x05\x06"),
            {"index": 0, "event_type": "step.stop"},
            {"event_type": "interaction.completed", "interaction": {"status": "completed"}},
        )
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    tts = make_tts(handler)
    chunks = [c async for c in tts.synthesize("嗯。", "轻")]
    assert all(len(c) % 2 == 0 for c in chunks)
    assert b"".join(chunks) == b"\x01\x02\x03\x04\x05\x06"
    assert seen["key"] == "k"
    assert seen["body"]["input"][0]["content"][0]["annotations"][0]["style"] == "轻"


async def test_http_error_raises():
    tts = make_tts(lambda r: httpx.Response(400, json={"error": {"message": "blocked"}}))
    with pytest.raises(TTSError, match="400"):
        await synthesize_all(tts, "你好")


async def test_stream_error_event_raises():
    body = sse(audio(b"\x00\x00"), {"event_type": "error", "error": {"message": "safety", "code": "blocked"}})
    tts = make_tts(lambda r: httpx.Response(200, text=body))
    with pytest.raises(TTSError, match="safety"):
        await synthesize_all(tts, "你好")
