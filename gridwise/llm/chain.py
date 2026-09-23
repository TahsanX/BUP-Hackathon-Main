"""Failover orchestration: the reusable core of this whole codebase.

Problem-agnostic by construction — it takes a prompt pair plus a pydantic model
and returns a validated instance. Nothing under gridwise/llm/ imports anything
from gridwise/problem/, which is what lets a future project keep this layer
byte-identical and swap only the prompt and the schema.

Providers are tried in order rather than raced. Racing every provider on every
request would burn all three quotas at once, which is precisely the failure we
are trying to avoid; a short cooldown on a provider that just rate-limited
gets the latency benefit without the quota cost.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from gridwise.core.config import Settings
from gridwise.llm.json_guard import extract_json_object
from gridwise.llm.provider import Provider
from gridwise.llm.types import (
    Attempt,
    AuthFailure,
    LLMRequest,
    NotConfigured,
    ProviderError,
    ProviderTimeout,
    ProviderTrace,
    RateLimited,
    ServerError,
    redact,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Provider name -> monotonic timestamp before which we skip it.
_cooldowns: dict[str, float] = {}

MIN_SLICE_SECONDS = 1.5


def reset_cooldowns() -> None:
    _cooldowns.clear()


def _cooled_down(name: str) -> bool:
    return time.monotonic() < _cooldowns.get(name, 0.0)


def _cool(name: str, seconds: float) -> None:
    _cooldowns[name] = time.monotonic() + seconds


async def complete_structured(
    *,
    system: str,
    user: str,
    schema: type[T],
    providers: Sequence[Provider],
    settings: Settings | None = None,
    repair_attempts: int = 1,
    json_schema: dict | None = None,
) -> tuple[T | None, ProviderTrace]:
    settings = settings or Settings.from_env()
    trace = ProviderTrace()
    deadline = time.monotonic() + settings.total_budget_seconds

    for provider in providers:
        remaining = deadline - time.monotonic()
        if remaining < MIN_SLICE_SECONDS:
            trace.record(Attempt(provider.name, "budget_exhausted"))
            break

        if _cooled_down(provider.name):
            trace.record(Attempt(provider.name, "cooldown"))
            continue

        base_name = provider.name.split("#", 1)[0]
        timeout = min(settings.timeouts.get(base_name, 8.0), remaining)
        parsed = await _ask(
            provider,
            system=system,
            user=user,
            schema=schema,
            timeout=timeout,
            repair_attempts=repair_attempts,
            cooldown_seconds=settings.cooldown_seconds,
            json_schema=json_schema or schema.model_json_schema(),
            trace=trace,
        )
        if parsed is not None:
            trace.winner = provider.name
            return parsed, trace

    logger.warning("interpretation: no provider succeeded (%s)", trace.summary())
    return None, trace


async def _ask(
    provider: Provider,
    *,
    system: str,
    user: str,
    schema: type[T],
    timeout: float,
    repair_attempts: int,
    cooldown_seconds: float,
    trace: ProviderTrace,
    json_schema: dict,
) -> T | None:
    prompt = user

    for attempt_number in range(repair_attempts + 1):
        started = time.monotonic()
        try:
            response = await provider.generate(
                LLMRequest(
                    system=system,
                    user=prompt,
                    json_schema=json_schema,
                    timeout=timeout,
                )
            )
        except NotConfigured as exc:
            trace.record(Attempt(provider.name, "not_configured", str(exc)))
            return None
        except AuthFailure as exc:
            # A rejected key will stay rejected; do not spend another attempt on it.
            trace.record(Attempt(provider.name, "auth_failure", str(exc)))
            _cool(provider.name, cooldown_seconds)
            return None
        except (RateLimited, ProviderTimeout, ServerError, asyncio.TimeoutError) as exc:
            outcome = type(exc).__name__.lower()
            trace.record(
                Attempt(provider.name, outcome, str(exc), time.monotonic() - started)
            )
            _cool(provider.name, cooldown_seconds)
            return None
        except ProviderError as exc:
            trace.record(Attempt(provider.name, "error", str(exc), time.monotonic() - started))
            return None
        except Exception as exc:  # adapter bug or unexpected transport failure
            trace.record(
                Attempt(provider.name, "unexpected", redact(str(exc)), time.monotonic() - started)
            )
            _cool(provider.name, cooldown_seconds)
            return None

        elapsed = time.monotonic() - started
        payload = extract_json_object(response.text)
        if payload is not None:
            try:
                parsed = schema.model_validate(payload)
            except ValidationError as exc:
                problem = _first_error(exc)
            else:
                trace.record(Attempt(provider.name, "ok", response.model, elapsed))
                return parsed
        else:
            problem = "the reply contained no JSON object"

        trace.record(Attempt(provider.name, "malformed", problem, elapsed))

        if attempt_number >= repair_attempts:
            return None

        # One corrective re-prompt to the same provider before failing over: a
        # near-miss is usually cheaper to fix here than to re-ask elsewhere.
        trace.repaired = True
        prompt = (
            f"{user}\n\nYour previous reply could not be used: {problem}. "
            "Reply with ONLY a valid JSON object matching the required schema, "
            "with no commentary and no code fences."
        )

    return None


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "the reply did not match the required schema"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "payload"
    return f"{location}: {first.get('msg', 'invalid')}"
