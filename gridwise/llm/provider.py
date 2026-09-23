"""The only contract a provider has to satisfy.

Adapters return raw text and raise a typed ProviderError. They do not parse
JSON, do not retry, and do not know what the text is going to be used for —
that keeps this layer reusable across problems unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gridwise.llm.types import LLMRequest, LLMResponse


@runtime_checkable
class Provider(Protocol):
    name: str

    async def generate(self, request: LLMRequest) -> LLMResponse: ...
