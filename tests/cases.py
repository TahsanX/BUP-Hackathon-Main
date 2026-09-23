"""Loader for the organizer's public sample pack.

The pack is vendored at tests/fixtures/public_sample_cases.json and committed.
The previous build kept it out of the repo via .gitignore, which is why its
end-to-end test silently failed to collect and the real bug shipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gridwise.problem.domain import Battery, Directive, Scenario, VALUE_FIELD

FIXTURE = Path(__file__).parent / "fixtures" / "public_sample_cases.json"

CASES: list[dict[str, Any]] = json.loads(FIXTURE.read_text())["cases"]
CASE_IDS: list[str] = [case["id"] for case in CASES]


def scenario_from_input(payload: dict[str, Any]) -> Scenario:
    hours = sorted(payload["hours"], key=lambda h: h["hour"])
    battery = payload["battery"]
    return Scenario(
        scenario_id=payload["scenario_id"],
        demand=tuple(float(h["demand_kwh"]) for h in hours),
        solar=tuple(float(h["solar_kwh"]) for h in hours),
        tariff=tuple(float(h["tariff_bdt_per_kwh"]) for h in hours),
        battery=Battery(
            capacity_kwh=float(battery["capacity_kwh"]),
            initial_energy_kwh=float(battery["initial_energy_kwh"]),
            minimum_energy_kwh=float(battery["minimum_energy_kwh"]),
            max_charge_kwh_per_hour=float(battery["max_charge_kwh_per_hour"]),
            max_discharge_kwh_per_hour=float(battery["max_discharge_kwh_per_hour"]),
        ),
    )


def directives_from_expected(entries: list[dict[str, Any]]) -> list[Directive]:
    """Turn the pack's reference interpretation into ground-truth directives."""
    directives: list[Directive] = []
    for entry in entries:
        kind = entry["directive_type"]
        adjustment = entry.get("structured_adjustment") or {}
        field = VALUE_FIELD[kind]
        directives.append(
            Directive(
                note_index=entry["note_index"],
                directive_type=kind,
                hours=tuple(int(h) for h in adjustment.get("hours", ())),
                value=float(adjustment[field]) if field is not None else None,
                explanation=entry.get("explanation", ""),
            )
        )
    return directives
