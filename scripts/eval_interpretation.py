"""Score a provider's interpretation accuracy on deliberately awkward paraphrases.

    python scripts/eval_interpretation.py --provider ollama --model gemma3:4b
    python scripts/eval_interpretation.py --provider groq --compare gemma3:4b,qwen2.5:3b-instruct

Hidden judge notes paraphrase; the public pack's wording teaches nothing about
that. Every case below is phrased unlike the sample pack and targets a specific
way the extraction goes wrong: the BY/TO inversion, windows that wrap midnight,
percentages of capacity, and maintenance notes that look like small talk.

Accuracy is what matters here, not latency. The local model sits third in the
chain, so by the time it runs the upstream timeouts already dominate the
response time — but it is also the last thing standing between a paraphrased
note and an all-no_op answer, which costs both interpretation and application
marks. A wrong-but-well-formed directive is the one failure the guardrails
cannot catch.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridwise.core.config import Settings  # noqa: E402
from gridwise.llm.chain import complete_structured, reset_cooldowns  # noqa: E402
from gridwise.llm.providers import GeminiProvider, GroqProvider, OllamaProvider  # noqa: E402
from gridwise.problem.directives.prompt import render_notes, system_prompt  # noqa: E402
from gridwise.problem.directives.schema import RawDirectiveBatch  # noqa: E402
from gridwise.problem.directives.validate import validate_batch  # noqa: E402
from gridwise.problem.domain import Battery  # noqa: E402

BATTERY = Battery(
    capacity_kwh=300.0,
    initial_energy_kwh=150.0,
    minimum_energy_kwh=50.0,
    max_charge_kwh_per_hour=60.0,
    max_discharge_kwh_per_hour=60.0,
)


@dataclass
class Case:
    label: str
    note: str
    directive_type: str
    hours: tuple[int, ...]
    value: float | None


CASES = [
    Case(
        "solar: TO phrasing",
        "Haze is forecast midday - expect output to fall to roughly 30% of normal "
        "between 11 in the morning and 2 in the afternoon.",
        "solar_reduction", (11, 12, 13), 0.30,
    ),
    Case(
        "solar: BY inversion",
        "Panel washing will cut rooftop generation by about 70% from 9 AM to 11 AM.",
        "solar_reduction", (9, 10), 0.30,
    ),
    Case(
        "solar: fraction in words",
        "With the array half covered by scaffolding from 1 PM to 4 PM we will only "
        "see about a quarter of the usual yield.",
        "solar_reduction", (13, 14, 15), 0.25,
    ),
    Case(
        "solar: total loss",
        "The PV feeder is isolated entirely between 07:00 and 09:00.",
        "solar_reduction", (7, 8), 0.0,
    ),
    Case(
        "reserve: percent of capacity",
        "Please keep no less than 40% of the pack in hand from 6 PM to 9 PM in case "
        "the ward needs backup.",
        "minimum_battery_reserve", (18, 19, 20), 120.0,
    ),
    Case(
        "reserve: fraction in words",
        "Hold back at least a third of the battery through the 8 PM to 11 PM window.",
        "minimum_battery_reserve", (20, 21, 22), 100.0,
    ),
    Case(
        "no_charge: indirect wording",
        "Electricians have the charger cabinet open from 2 AM until 5 AM, so nothing "
        "can go into the pack.",
        "no_charge_window", (2, 3, 4), None,
    ),
    Case(
        "no_charge: lockout jargon",
        "Rectifier lockout applies 13:00-16:00.",
        "no_charge_window", (13, 14, 15), None,
    ),
    Case(
        "no_discharge: indirect wording",
        "During the protection trials from 5 to 7 in the morning the battery must not "
        "feed any load.",
        "no_discharge_window", (5, 6), None,
    ),
    Case(
        "no_discharge: 'keep it offline'",
        "Keep the pack off the bus between 4 PM and 6 PM while the relay is swapped.",
        "no_discharge_window", (16, 17), None,
    ),
    Case(
        "grid cap: derate wording",
        "The utility has derated our feeder - hold import under 140 units in any hour "
        "from 6 PM to 8 PM.",
        "max_grid_window", (18, 19), 140.0,
    ),
    Case(
        "grid cap: wraps midnight",
        "Overnight the substation limits us to 90 kWh an hour from 11 PM through 2 AM.",
        # Ascending order is what the spec requires, so a wrapped window sorts to 0,1,23.
        "max_grid_window", (0, 1, 23), 90.0,
    ),
    Case(
        "no_op: administrative",
        "Reminder that the semester fee deadline moves to the 14th.",
        "no_op", (), None,
    ),
    Case(
        "no_op: sounds technical but is not",
        "IT will migrate the campus wifi controller to the new rack tonight.",
        "no_op", (), None,
    ),
    Case(
        "no_op: future-dated",
        "Next month's shutdown drill will be scheduled once the vendor confirms.",
        "no_op", (), None,
    ),
]


def build_provider(name: str, model: str | None, settings: Settings):
    if name == "gemini":
        return GeminiProvider(settings.gemini_api_key, model or settings.gemini_model)
    if name == "groq":
        return GroqProvider(settings.groq_api_key, model or settings.groq_model)
    if name == "ollama":
        return OllamaProvider(settings.ollama_host, model or settings.ollama_model)
    raise SystemExit(f"unknown provider {name!r}")


def grade(case: Case, directive) -> tuple[bool, str]:
    if directive.directive_type != case.directive_type:
        return False, f"type {directive.directive_type} != {case.directive_type}"
    if case.directive_type == "no_op":
        return True, ""
    if tuple(directive.hours) != case.hours:
        return False, f"hours {list(directive.hours)} != {list(case.hours)}"
    if case.value is not None:
        got = directive.value if directive.value is not None else -1
        # 0.02 absolute on factors, 1% on kWh quantities.
        tolerance = 0.02 if case.directive_type == "solar_reduction" else max(1.0, case.value * 0.01)
        if abs(got - case.value) > tolerance:
            return False, f"value {got} != {case.value}"
    return True, ""


async def evaluate(provider, settings: Settings) -> tuple[int, list[float], list[str]]:
    correct = 0
    latencies: list[float] = []
    misses: list[str] = []

    for case in CASES:
        reset_cooldowns()
        started = time.monotonic()
        batch, _ = await complete_structured(
            system=system_prompt(),
            user=render_notes([case.note], BATTERY),
            schema=RawDirectiveBatch,
            providers=[provider],
            settings=settings,
        )
        latencies.append(time.monotonic() - started)

        if batch is None:
            misses.append(f"{case.label}: no answer")
            continue

        result = validate_batch(batch, n_notes=1, battery=BATTERY)
        ok, reason = grade(case, result.directives[0])
        if ok:
            correct += 1
        else:
            misses.append(f"{case.label}: {reason}")

    return correct, latencies, misses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--compare", default=None, help="comma-separated model ids to benchmark head to head"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    settings = Settings.from_env()
    settings = Settings(
        **{
            **settings.__dict__,
            "total_budget_seconds": max(settings.total_budget_seconds, args.timeout + 5),
            "timeouts": {args.provider: args.timeout},
        }
    )

    models = args.compare.split(",") if args.compare else [args.model]

    print(f"{len(CASES)} paraphrase cases | provider={args.provider}\n")
    rows = []
    for model in models:
        provider = build_provider(args.provider, model, settings)
        label = model or "default"
        correct, latencies, misses = asyncio.run(evaluate(provider, settings))

        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        rows.append((label, correct, statistics.mean(latencies), p95))

        print(f"--- {label} ---")
        print(f"  accuracy : {correct}/{len(CASES)} ({100 * correct / len(CASES):.0f}%)")
        print(f"  latency  : mean {statistics.mean(latencies):.2f}s | p95 {p95:.2f}s")
        for miss in misses:
            print(f"  MISS     : {miss}")
        print()

    if len(rows) > 1:
        print(f"{'model':<28}{'accuracy':>10}{'mean':>10}{'p95':>10}")
        for label, correct, mean, p95 in rows:
            print(f"{label:<28}{correct:>7}/{len(CASES):<2}{mean:>9.2f}s{p95:>9.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
