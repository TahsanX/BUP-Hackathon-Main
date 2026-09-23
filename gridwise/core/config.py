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

DEFAULT_ORDER = ("gemini", "groq", "nvidia", "ollama")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _split_keys(*env_names: str) -> tuple[str, ...]:
    """Merge a comma-separated *_API_KEYS var and a singular *_API_KEY var into
    one ordered, de-duplicated key pool. A pool of size N > 1 gives the
    provider chain N independent candidates instead of one: a key that gets
    rate-limited cools down alone, and its siblings keep answering requests.
    Never logs the values."""
    keys: list[str] = []
    for name in env_names:
        raw = os.environ.get(name, "")
        for part in raw.split(","):
            part = part.strip()
            if part and part not in keys:
                keys.append(part)
    return tuple(keys)


@dataclass(frozen=True)
class Settings:
    provider_order: tuple[str, ...] = DEFAULT_ORDER

    # A pool, not a single key: GEMINI_API_KEYS="k1,k2" (or one GEMINI_API_KEY)
    # gives the chain multiple independent candidates for the same provider,
    # so one key's rate limit doesn't take Gemini out of the chain entirely.
    gemini_api_keys: tuple[str, ...] = ()
    # 2.0-flash shut down 2026-06-01 and 2.5-* is closed to new projects. Start on
    # the cheap/fast tier and move up to gemini-3.8-flash if the interpretation
    # eval shows misses — a semantically wrong directive is the one failure the
    # guardrails cannot catch, so first-provider accuracy is worth paying for.
    gemini_model: str = "gemini-3.5-flash-lite"

    groq_api_keys: tuple[str, ...] = ()
    # llama-3.3-70b-versatile was decommissioned for free/developer-tier keys on
    # 2026-08-16; gpt-oss-120b is Groq's recommended replacement at that tier.
    groq_model: str = "openai/gpt-oss-120b"

    # build.nvidia.com free endpoints. Unlike gemini/groq, the pool here is
    # usually one key fanned out across several *models* — build_providers()
    # takes the cross product of keys x models, so one key + three model ids
    # still yields three independent chain candidates.
    nvidia_api_keys: tuple[str, ...] = ()
    nvidia_models: tuple[str, ...] = ()

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:3b-instruct"

    total_budget_seconds: float = 20.0
    cooldown_seconds: float = 30.0
    timeouts: dict[str, float] = field(
        default_factory=lambda: {"gemini": 6.0, "groq": 6.0, "nvidia": 8.0, "ollama": 9.0}
    )

    @property
    def gemini_api_key(self) -> str | None:
        """First key in the pool, for callers that only want a single one."""
        return self.gemini_api_keys[0] if self.gemini_api_keys else None

    @property
    def groq_api_key(self) -> str | None:
        return self.groq_api_keys[0] if self.groq_api_keys else None

    @classmethod
    def from_env(cls) -> Settings:
        raw_order = os.environ.get("LLM_PROVIDER_ORDER", "")
        order = tuple(p.strip() for p in raw_order.split(",") if p.strip()) or DEFAULT_ORDER
        return cls(
            provider_order=order,
            gemini_api_keys=_split_keys("GEMINI_API_KEYS", "GEMINI_API_KEY"),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            groq_api_keys=_split_keys("GROQ_API_KEYS", "GROQ_API_KEY"),
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            nvidia_api_keys=_split_keys("NVIDIA_API_KEYS", "NVIDIA_API_KEY"),
            nvidia_models=tuple(
                m.strip() for m in os.environ.get("NVIDIA_MODELS", "").split(",") if m.strip()
            ),
            ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b-instruct"),
            total_budget_seconds=_env_float("LLM_TOTAL_BUDGET_SECONDS", 20.0),
            cooldown_seconds=_env_float("LLM_COOLDOWN_SECONDS", 30.0),
            timeouts={
                "gemini": _env_float("GEMINI_TIMEOUT_SECONDS", 6.0),
                "groq": _env_float("GROQ_TIMEOUT_SECONDS", 6.0),
                "nvidia": _env_float("NVIDIA_TIMEOUT_SECONDS", 8.0),
                "ollama": _env_float("OLLAMA_TIMEOUT_SECONDS", 9.0),
            },
        )
