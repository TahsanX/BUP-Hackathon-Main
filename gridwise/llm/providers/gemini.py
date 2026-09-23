"""Google Gemini adapter (generativelanguage REST API)."""

from __future__ import annotations

from gridwise.llm.providers._http import post_json
from gridwise.llm.types import LLMRequest, LLMResponse, NotConfigured, ServerError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str) -> None:
        self._api_key = api_key
        self._model = model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise NotConfigured("GEMINI_API_KEY is not set")

        payload = {
            "system_instruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
                # Native JSON mode. The schema itself stays in the prompt: pydantic
                # emits $ref/$defs, which Gemini's responseSchema subset rejects.
                "responseMimeType": "application/json",
            },
        }

        body = await post_json(
            f"{BASE_URL}/{self._model}:generateContent",
            payload,
            timeout=request.timeout,
            provider=self.name,
            headers={"x-goog-api-key": self._api_key},
        )

        return LLMResponse(text=_extract_text(body), model=self._model)


def _extract_text(body: dict) -> str:
    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback", {})
        raise ServerError(f"gemini returned no candidates (feedback={feedback})")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        reason = candidates[0].get("finishReason", "unknown")
        raise ServerError(f"gemini returned an empty candidate (finishReason={reason})")
    return text
