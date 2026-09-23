"""Groq adapter (OpenAI-compatible chat completions)."""

from __future__ import annotations

from gridwise.llm.providers._http import post_json
from gridwise.llm.types import LLMRequest, LLMResponse, NotConfigured, ServerError

URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str | None, model: str, name: str | None = None) -> None:
        self._api_key = api_key
        self._model = model
        if name:
            self.name = name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise NotConfigured("GROQ_API_KEY is not set")

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        body = await post_json(
            URL,
            payload,
            timeout=request.timeout,
            provider=self.name,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

        choices = body.get("choices") or []
        if not choices:
            raise ServerError("groq returned no choices")

        text = (choices[0].get("message") or {}).get("content") or ""
        if not text.strip():
            raise ServerError(
                f"groq returned empty content (finish_reason={choices[0].get('finish_reason')})"
            )

        return LLMResponse(text=text, model=self._model)
