"""Transport-level types shared by every provider adapter.

The error taxonomy is the whole point: the chain routes on the *kind* of
failure, not on a generic exception. A dead key must not be retried, a rate
limit must cool the provider down, and malformed JSON deserves one corrective
re-prompt before we give up on that provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_SECRET = re.compile(
    r"(AIza[0-9A-Za-z_\-]{10,}|gsk_[0-9A-Za-z]{10,}|sk-[0-9A-Za-z]{10,}|Bearer\s+\S+)"
)


def redact(text: str) -> str:
    """Strip anything key-shaped before a message reaches a log or a client."""
    return _SECRET.sub("[redacted]", text)


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    json_schema: dict[str, Any] | None = None
    timeout: float = 8.0
    max_output_tokens: int = 2048
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str


class ProviderError(Exception):
    """Base class for every transport failure. Message is redacted on the way in."""

    def __init__(self, message: str) -> None:
        super().__init__(redact(message))


class AuthFailure(ProviderError):
    """Credentials are rejected. Retrying this key is pointless."""


class RateLimited(ProviderError):
    """Quota exhausted or throttled. Cool this provider down and move on."""


class ProviderTimeout(ProviderError):
    """The provider did not answer inside its slice of the budget."""


class ServerError(ProviderError):
    """Provider-side 5xx or transport failure."""


class MalformedOutput(ProviderError):
    """A response arrived but was not usable as the requested structure."""


class NotConfigured(ProviderError):
    """No credentials/host for this provider; skip it without counting a failure."""


@dataclass
class Attempt:
    provider: str
    outcome: str
    detail: str = ""
    seconds: float = 0.0


@dataclass
class ProviderTrace:
    """Diagnostics for one interpretation. Logged, never returned to the client."""

    attempts: list[Attempt] = field(default_factory=list)
    winner: str | None = None
    repaired: bool = False

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    def summary(self) -> str:
        trail = " -> ".join(
            f"{a.provider}:{a.outcome}" + (f"[{redact(str(a.detail))[:160]}]" if a.outcome not in ("ok", "cooldown") and a.detail else "")
            for a in self.attempts
        ) or "none"
        return f"winner={self.winner or 'none'} repaired={self.repaired} trail={trail}"
