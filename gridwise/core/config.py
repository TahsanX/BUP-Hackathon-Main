"""Environment-driven settings.

Every knob is an environment variable and nothing branches on the hosting
platform. The previous build carried a `if os.environ.get("VERCEL"): raise`
short-circuit in its LLM path, so the deployed service silently never called a
model and answered every scenario with all-no_op. Provider availability is
expressed by which keys are present and by LLM_PROVIDER_ORDER — never by
detecting where the process happens to be running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_ORDER = ("gemini", "groq", "ollama")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    provider_order: tuple[str, ...] = DEFAULT_ORDER

    gemini_api_key: str | None = None
    # 2.0-flash shut down 2026-06-01 and 2.5-* is closed to new projects. Start on
    # the cheap/fast tier and move up to gemini-3.8-flash if the interpretation
    # eval shows misses — a semantically wrong directive is the one failure the
    # guardrails cannot catch, so first-provider accuracy is worth paying for.
    gemini_model: str = "gemini-3.5-flash-lite"

    groq_api_key: str | None = None
    # llama-3.3-70b-versatile was decommissioned for free/developer-tier keys on
    # 2026-08-16; gpt-oss-120b is Groq's recommended replacement at that tier.
    groq_model: str = "openai/gpt-oss-120b"

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b-instruct"

    total_budget_seconds: float = 20.0
    cooldown_seconds: float = 30.0
    timeouts: dict[str, float] = field(
        default_factory=lambda: {"gemini": 6.0, "groq": 6.0, "ollama": 9.0}
    )

    @classmethod
    def from_env(cls) -> Settings:
        raw_order = os.environ.get("LLM_PROVIDER_ORDER", "")
        order = tuple(p.strip() for p in raw_order.split(",") if p.strip()) or DEFAULT_ORDER
        return cls(
            provider_order=order,
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or None,
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            groq_api_key=os.environ.get("GROQ_API_KEY") or None,
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            total_budget_seconds=_env_float("LLM_TOTAL_BUDGET_SECONDS", 20.0),
            cooldown_seconds=_env_float("LLM_COOLDOWN_SECONDS", 30.0),
            timeouts={
                "gemini": _env_float("GEMINI_TIMEOUT_SECONDS", 6.0),
                "groq": _env_float("GROQ_TIMEOUT_SECONDS", 6.0),
                "ollama": _env_float("OLLAMA_TIMEOUT_SECONDS", 9.0),
            },
        )
