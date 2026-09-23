"""Independent replay of a finished schedule.

Written from the problem statement's rules rather than from the solver, on
purpose: if it shared code with the optimizer it could only confirm the
optimizer's own mistakes. This is the last gate before a response goes out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from gridwise.problem.domain import N_HOURS, Overlay, PlanHour, Scenario

TOL = 0.01


def replay(
    scenario: Scenario,
    overlay: Overlay,
    plan: Sequence[PlanHour],
    *,
    total_grid_kwh: float,
    total_cost_bdt: float,
    peak_grid_kwh: float,
) -> list[str]:
    """Return a list of rule violations. Empty means the schedule is valid."""
    problems: list[str] = []
    battery = scenario.battery

    if len(plan) != N_HOURS or {p.hour for p in plan} != set(range(N_HOURS)):
        return ["hourly_plan must contain exactly one entry for each hour 0..23"]

    ordered = sorted(plan, key=lambda p: p.hour)
    level = battery.initial_energy_kwh

    for entry in ordered:
        h = entry.hour
        values = {
            "grid_kwh": entry.grid_kwh,
            "solar_used_kwh": entry.solar_used_kwh,
            "battery_kwh": entry.battery_kwh,
            "battery_energy_after_kwh": entry.battery_energy_after_kwh,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                problems.append(f"hour {h}: {name} is not finite")
            elif value < -TOL:
                problems.append(f"hour {h}: {name} is negative ({value})")

        if entry.battery_action not in {"charge", "discharge", "idle"}:
            problems.append(f"hour {h}: unknown battery_action {entry.battery_action!r}")
            continue

        if entry.battery_action == "idle" and abs(entry.battery_kwh) > TOL:
            problems.append(f"hour {h}: idle action must carry battery_kwh = 0")

        if entry.solar_used_kwh > overlay.effective_solar[h] + TOL:
            problems.append(
                f"hour {h}: solar_used_kwh {entry.solar_used_kwh} exceeds effective solar "
                f"{overlay.effective_solar[h]}"
            )

        charge = entry.battery_kwh if entry.battery_action == "charge" else 0.0
        discharge = entry.battery_kwh if entry.battery_action == "discharge" else 0.0

        if charge > 0:
            if not overlay.charge_ok[h]:
                problems.append(f"hour {h}: charging during a no_charge_window")
            if charge > battery.max_charge_kwh_per_hour + TOL:
                problems.append(f"hour {h}: charge {charge} exceeds hourly limit")
        if discharge > 0:
            if not overlay.discharge_ok[h]:
                problems.append(f"hour {h}: discharging during a no_discharge_window")
            if discharge > battery.max_discharge_kwh_per_hour + TOL:
                problems.append(f"hour {h}: discharge {discharge} exceeds hourly limit")

        if entry.grid_kwh > overlay.grid_cap[h] + TOL:
            problems.append(
                f"hour {h}: grid_kwh {entry.grid_kwh} exceeds cap {overlay.grid_cap[h]}"
            )

        balance = entry.grid_kwh + entry.solar_used_kwh + discharge - (
            scenario.demand[h] + charge
        )
        if abs(balance) > TOL:
            problems.append(f"hour {h}: energy balance off by {balance:.4f}")

        level = level + charge - discharge
        if abs(level - entry.battery_energy_after_kwh) > TOL:
            problems.append(
                f"hour {h}: battery_energy_after_kwh {entry.battery_energy_after_kwh} "
                f"does not follow from the action (expected {level:.4f})"
            )
        level = entry.battery_energy_after_kwh

        if level < overlay.min_floor[h] - TOL:
            problems.append(f"hour {h}: battery {level} below required reserve {overlay.min_floor[h]}")
        if level > battery.capacity_kwh + TOL:
            problems.append(f"hour {h}: battery {level} above capacity {battery.capacity_kwh}")

    if abs(level - battery.initial_energy_kwh) > TOL:
        problems.append(
            f"end-of-day battery {level} does not return to initial "
            f"{battery.initial_energy_kwh}"
        )

    recomputed_grid = sum(p.grid_kwh for p in ordered)
    recomputed_cost = sum(p.grid_kwh * scenario.tariff[p.hour] for p in ordered)
    recomputed_peak = max(p.grid_kwh for p in ordered)

    for name, reported, expected in (
        ("total_grid_kwh", total_grid_kwh, recomputed_grid),
        ("total_cost_bdt", total_cost_bdt, recomputed_cost),
        ("peak_grid_kwh", peak_grid_kwh, recomputed_peak),
    ):
        if abs(reported - expected) > TOL:
            problems.append(f"{name} {reported} disagrees with recalculated {expected:.4f}")

    return problems
