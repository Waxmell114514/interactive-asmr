"""TTS：Gemini 3.8 TTS（Interactions API，SSE 流式，24kHz 单声道 16-bit PCM）。

style 的传法（Spec 里的待验证项）：Gemini 3.8 TTS 把 ``text`` 严格当作逐字稿，
语气放进 ``speech_metadata.style`` 注解就不会被念出来。``prefix`` 模式把 style 写成文本前缀，
只保留给 M0 做对比试听。
"""

from __future__ import annotations

import base64
import json
import logging
from typing import AsyncIterator, Protocol

import httpx

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000


class TTSError(RuntimeError):
    pass


class TTS(Protocol):
    sample_rate: int

    def synthesize(self, text: str, style: str = "") -> AsyncIterator[bytes]: ...


def build_request(model: str, voice: str, text: str, style: str, style_mode: str, stream: bool = True) -> dict:
    content: dict = {"type": "text", "text": text}
    if style and style_mode == "annotation":
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    elif style and style_mode == "prefix":
        content["text"] = f"{style}：{text}"
    return {
        "model": model,
        "input": [{"type": "user_input", "content": [content]}],
        "response_format": {"type": "audio", "mime_type": "audio/l16", "sample_rate": SAMPLE_RATE},
        "generation_config": {"speech_config": [{"voice": voice}]},
        "stream": stream,
    }


class GeminiTTS:
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        *,
        api_key: str,
        voice: str,
        model: str = "gemini-3.8-flash-lite-tts",
        style_mode: str = "annotation",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        client: httpx.AsyncClient | None = None,
    ):
        self.voice = voice
        self.model = model
        self.style_mode = style_mode
        self._url = f"{base_url.rstrip('/')}/interactions"
        self._headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def synthesize(self, text: str, style: str = "") -> AsyncIterator[bytes]:
        body = build_request(self.model, self.voice, text, style, self.style_mode)
        carry = b""  # 保证下发的 PCM 块按 2 字节对齐
        async with self._client.stream("POST", self._url, headers=self._headers, json=body) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise TTSError(f"Gemini TTS HTTP {resp.status_code}: {detail}")
            async for event in _sse_events(resp.aiter_lines()):
                kind = event.get("event_type")
                if kind == "step.delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "audio" and delta.get("data"):
                        pcm = carry + base64.b64decode(delta["data"])
                        cut = len(pcm) - len(pcm) % 2
                        carry = pcm[cut:]
                        if cut:
                            yield pcm[:cut]
                elif kind == "error":
                    raise TTSError(f"Gemini TTS 错误: {event.get('error')}")
                elif kind == "interaction.completed":
                    status = (event.get("interaction") or {}).get("status")
                    if status not in (None, "completed"):
                        raise TTSError(f"Gemini TTS 未完成: {status}")

    async def aclose(self) -> None:
        await self._client.aclose()


async def _sse_events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    data: list[str] = []
    async for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line == "" and data:
            payload = "\n".join(data)
            data = []
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                log.warning("无法解析的 SSE 数据: %s", payload[:200])
    if data and data != ["[DONE]"]:
        try:
            yield json.loads("\n".join(data))
        except json.JSONDecodeError:
            pass


async def synthesize_all(tts: TTS, text: str, style: str = "") -> bytes:
    return b"".join([chunk async for chunk in tts.synthesize(text, style)])
