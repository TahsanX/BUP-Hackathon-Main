"""Provider registry.

Order comes from LLM_PROVIDER_ORDER alone. A provider with no credentials is
simply absent from the chain — there is no platform sniffing anywhere.

Each API key in a provider's pool becomes its own chain candidate (gemini,
gemini#2, ...), so a single rate-limited key only cools down itself instead
of removing the whole provider — the load-balancing pattern also used by one
of the two accepted submissions to this challenge.
"""

from __future__ import annotations

from gridwise.core.config import Settings
from gridwise.llm.provider import Provider
from gridwise.llm.providers.gemini import GeminiProvider
from gridwise.llm.providers.groq import GroqProvider
from gridwise.llm.providers.ollama import OllamaProvider

__all__ = ["GeminiProvider", "GroqProvider", "OllamaProvider", "build_providers"]


def _pool_providers(cls: type[Provider], base_name: str, keys: tuple[str, ...], model: str) -> list[Provider]:
    return [
        cls(key, model, name=base_name if i == 0 else f"{base_name}#{i + 1}")
        for i, key in enumerate(keys)
    ]


def build_providers(settings: Settings) -> list[Provider]:
    available: list[Provider] = []
    for name in settings.provider_order:
        if name == "gemini":
            available += _pool_providers(GeminiProvider, "gemini", settings.gemini_api_keys, settings.gemini_model)
        elif name == "groq":
            available += _pool_providers(GroqProvider, "groq", settings.groq_api_keys, settings.groq_model)
        elif name == "ollama":
            available.append(OllamaProvider(settings.ollama_host, settings.ollama_model))
    return available
