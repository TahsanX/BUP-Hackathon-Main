"""Ollama adapter — the terminal link in the chain.

Runs inside our own container against a model baked into the image at build
time, so it has no quota, no key and no dependency on anyone else's uptime.
That is what lets the chain always end with an answer instead of a no_op.
"""

from __future__ import annotations

from gridwise.llm.providers._http import post_json
from gridwise.llm.types import LLMRequest, LLMResponse, ServerError


class OllamaProvider:
    name = "ollama"

    def __init__(self, host: str, model: str) -> None:
        self._host = host.rstrip("/")
        self._model = model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }

        body = await post_json(
            f"{self._host}/api/chat",
            payload,
            timeout=request.timeout,
            provider=self.name,
        )

        text = (body.get("message") or {}).get("content") or ""
        if not text.strip():
            raise ServerError("ollama returned empty content")

        return LLMResponse(text=text, model=self._model)
