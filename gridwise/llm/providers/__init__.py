"""Provider registry.

Order comes from LLM_PROVIDER_ORDER alone. A provider with no credentials is
simply absent from the chain — there is no platform sniffing anywhere.
"""

from __future__ import annotations

from gridwise.core.config import Settings
from gridwise.llm.provider import Provider
from gridwise.llm.providers.gemini import GeminiProvider
from gridwise.llm.providers.groq import GroqProvider
from gridwise.llm.providers.ollama import OllamaProvider

__all__ = ["GeminiProvider", "GroqProvider", "OllamaProvider", "build_providers"]


def build_providers(settings: Settings) -> list[Provider]:
    available: list[Provider] = []
    for name in settings.provider_order:
        if name == "gemini" and settings.gemini_api_key:
            available.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        elif name == "groq" and settings.groq_api_key:
            available.append(GroqProvider(settings.groq_api_key, settings.groq_model))
        elif name == "ollama":
            available.append(OllamaProvider(settings.ollama_host, settings.ollama_model))
    return available
