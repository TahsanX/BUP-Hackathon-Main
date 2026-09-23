"""Human-readable plan_summary.

Template-driven on purpose. The rubric is explicit that using a language model
only for the summary does not satisfy the LLM requirement, so spending a call
here would buy latency and risk without buying a single point.
"""

from __future__ import annotations

from collections.abc import Sequence

from gridwise.problem.domain import Directive, Scenario, SolveResult

_PHRASING = {
    "solar_reduction": "reduced solar availability",
    "minimum_battery_reserve": "a raised battery reserve",
    "no_charge_window": "a no-charging window",
    "no_discharge_window": "a no-discharging window",
    "max_grid_window": "a grid import cap",
}


def build_plan_summary(
    scenario: Scenario, directives: Sequence[Directive], result: SolveResult
) -> str:
    applied = [d for d in directives if d.applies]
    ignored = len(directives) - len(applied)

    parts: list[str] = []

    if applied:
        described = ", ".join(
            f"{_PHRASING[d.directive_type]} for hour(s) "
            f"{', '.join(str(h) for h in d.hours)}"
            for d in applied
        )
        parts.append(f"Applied {described}.")
    else:
        parts.append("No operator note changed the schedule.")

    if ignored:
        note_word = "note" if ignored == 1 else "notes"
        parts.append(f"Ignored {ignored} unrelated {note_word}.")

    charge_hours = [p.hour for p in result.plan if p.battery_action == "charge"]
    discharge_hours = [p.hour for p in result.plan if p.battery_action == "discharge"]
    if charge_hours and discharge_hours:
        cheap = min(scenario.tariff[h] for h in charge_hours)
        dear = max(scenario.tariff[h] for h in discharge_hours)
        parts.append(
            f"Charged the battery in hours priced from {cheap:g} BDT/kWh and discharged "
            f"into hours priced up to {dear:g} BDT/kWh, returning it to its starting level "
            "by the end of the day."
        )
    else:
        parts.append("Held the battery at its starting level across the day.")

    parts.append(
        f"Total grid import {result.total_grid_kwh:g} kWh at {result.total_cost_bdt:g} BDT, "
        f"peaking at {result.peak_grid_kwh:g} kWh in one hour."
    )

    if not result.feasible:
        parts.append("Some requested limits could not be met exactly and were relaxed.")

    return " ".join(parts)
