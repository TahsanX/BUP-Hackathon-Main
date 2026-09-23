"""Google Gemini adapter (generativelanguage REST API)."""

from __future__ import annotations

from gridwise.llm.providers._http import post_json
from gridwise.llm.types import LLMRequest, LLMResponse, NotConfigured, ServerError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str, name: str | None = None) -> None:
        self._api_key = api_key
        self._model = model
        if name:
            # Distinguishes key pool members (e.g. "gemini#2") so a rate-limited
            # key cools down alone instead of taking every key for this
            # provider down with it.
            self.name = name

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise NotConfigured("GEMINI_API_KEY is not set")

        generation_config = {
            "temperature": 0,
            "topP": 0,
            "topK": 1,
            "seed": 7,
            # No maxOutputTokens: 3.x models are thinking-only and thinking
            # tokens count against that cap, truncating the answer to nothing.
            "responseMimeType": "application/json",
        }
        if request.json_schema:
            generation_config["responseSchema"] = to_gemini_schema(request.json_schema)

        payload = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.user}]}],
            "generationConfig": generation_config,
        }

        body = await post_json(
            f"{BASE_URL}/{self._model}:generateContent",
            payload,
            timeout=request.timeout,
            provider=self.name,
            headers={"x-goog-api-key": self._api_key},
        )

        return LLMResponse(text=_extract_text(body), model=self._model)


_DROP = {"title", "default", "additionalProperties", "$defs", "examples"}


def to_gemini_schema(schema: dict) -> dict:
    """Pydantic JSON schema -> Gemini's OpenAPI subset: inline $refs, turn
    `anyOf: [X, null]` into a nullable X, and drop keys Gemini rejects."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        if "anyOf" in node:
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            merged = walk(options[0]) if len(options) == 1 else {"type": "string"}
            if len(options) < len(node["anyOf"]):
                merged = {**merged, "nullable": True}
            extra = {k: walk(v) for k, v in node.items() if k not in _DROP and k != "anyOf"}
            return {**merged, **extra}
        return {k: walk(v) for k, v in node.items() if k not in _DROP}

    return walk(schema)


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
