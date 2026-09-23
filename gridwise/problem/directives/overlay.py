"""Fold validated directives into per-hour constraint arrays.

Overlap rules come straight from the problem statement: solar reductions stack
multiplicatively, reserves take the strictest floor, grid caps take the
strictest ceiling, and window bans are absorbing.
"""

from __future__ import annotations

from collections.abc import Iterable

from gridwise.problem.domain import Directive, Overlay, Scenario


def build_overlay(scenario: Scenario, directives: Iterable[Directive]) -> Overlay:
    overlay = Overlay.base(scenario)

    for directive in directives:
        kind = directive.directive_type
        if kind == "no_op":
            continue

        for hour in directive.hours:
            if kind == "solar_reduction":
                overlay.effective_solar[hour] *= directive.value
            elif kind == "minimum_battery_reserve":
                overlay.min_floor[hour] = max(overlay.min_floor[hour], directive.value)
            elif kind == "max_grid_window":
                overlay.grid_cap[hour] = min(overlay.grid_cap[hour], directive.value)
            elif kind == "no_charge_window":
                overlay.charge_ok[hour] = False
            elif kind == "no_discharge_window":
                overlay.discharge_ok[hour] = False

    return overlay
