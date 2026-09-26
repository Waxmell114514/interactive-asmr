"""OpenAI 兼容的流式 LLM 客户端（DeepSeek / Gemini / Claude 等均可）。"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

from openai import AsyncOpenAI


class LLM(Protocol):
    def stream(self, messages: list[dict]) -> AsyncIterator[str]: ...


class OpenAICompatibleLLM:
    def __init__(self, *, base_url: str, api_key: str, model: str, temperature: float = 0.8, max_tokens: int = 300):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or "none")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        try:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            # 被打断时任务取消，这里关闭连接，服务端停止生成。
            await stream.close()
