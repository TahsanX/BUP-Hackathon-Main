"""Verify each configured provider actually answers, one at a time.

    python scripts/check_providers.py

Model IDs churn — Groq retired llama-3.3-70b-versatile for free keys in
Aug 2026 and Gemini shut down 2.0-flash in Jun 2026 — so a "bad key" is often
really a dead model id. This separates the two: an auth failure names the key,
a 404 names the model.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridwise.core.config import Settings  # noqa: E402
from gridwise.llm.chain import complete_structured, reset_cooldowns
from gridwise.llm.providers import build_providers
from gridwise.llm.types import ProviderError
from gridwise.problem.directives.prompt import render_notes, system_prompt
from gridwise.problem.directives.schema import RawDirectiveBatch
from gridwise.problem.directives.validate import validate_batch
from gridwise.problem.domain import Battery

BATTERY = Battery(
    capacity_kwh=200.0,
    initial_energy_kwh=100.0,
    minimum_energy_kwh=40.0,
    max_charge_kwh_per_hour=50.0,
    max_discharge_kwh_per_hour=50.0,
)

NOTES = [
    "Crews wash the array from 10 AM until 1 PM; treat usable solar as about a "
    "quarter of forecast while they work.",
    "The library extends its opening hours next semester.",
]

EXPECTED = [("solar_reduction", (10, 11, 12), 0.25), ("no_op", (), None)]


async def check(settings: Settings) -> int:
    providers = build_providers(settings)
    if not providers:
        print("No provider is configured. Set GEMINI_API_KEY and/or GROQ_API_KEY.")
        return 1

    model_of = {
        "gemini": settings.gemini_model,
        "groq": settings.groq_model,
        "ollama": settings.ollama_model,
    }

    failures = 0
    for provider in providers:
        reset_cooldowns()
        label = f"{provider.name} ({model_of.get(provider.name, '?')})"
        started = time.monotonic()

        try:
            batch, trace = await complete_structured(
                system=system_prompt(),
                user=render_notes(NOTES, BATTERY),
                schema=RawDirectiveBatch,
                providers=[provider],
                settings=settings,
            )
        except ProviderError as exc:
            print(f"  FAIL  {label}: {exc}")
            failures += 1
            continue

        elapsed = time.monotonic() - started

        if batch is None:
            detail = "; ".join(f"{a.outcome}: {a.detail}" for a in trace.attempts)
            print(f"  FAIL  {label}  {elapsed:.2f}s  {detail}")
            failures += 1
            continue

        result = validate_batch(batch, n_notes=len(NOTES), battery=BATTERY)
        got = [(d.directive_type, d.hours, d.value) for d in result.directives]

        correct = all(
            actual[0] == want[0] and actual[1] == want[1]
            and (want[2] is None or abs((actual[2] or 0) - want[2]) < 0.06)
            for actual, want in zip(got, EXPECTED)
        )

        verdict = "OK  " if correct else "WARN"
        print(f"  {verdict}  {label}  {elapsed:.2f}s  {'clean' if result.clean else 'downgraded'}")
        for directive in result.directives:
            print(
                f"          note {directive.note_index}: {directive.directive_type} "
                f"hours={list(directive.hours)} value={directive.value}"
            )
        if not correct:
            print(f"          expected: {EXPECTED}")

    return failures


def main() -> int:
    settings = Settings.from_env()
    print(f"chain order: {' -> '.join(settings.provider_order)}\n")
    failures = asyncio.run(check(settings))
    print()
    if failures:
        print(f"{failures} provider(s) unreachable — see the reason above.")
    else:
        print("All configured providers answered.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
