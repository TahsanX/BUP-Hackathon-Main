"""Pure domain types shared by the guardrail, overlay, solver and replay layers.

Deliberately dataclasses rather than pydantic models: everything downstream of
the guardrail runs with zero I/O and zero framework dependency, so the whole
optimizer is unit-testable without an API key or a network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

N_HOURS = 24

# Per directive type: the key its numeric payload uses inside
# `structured_adjustment`. Window types carry no number.
VALUE_FIELD: dict[str, str | None] = {
    "solar_reduction": "factor",
    "minimum_battery_reserve": "minimum_energy_kwh",
    "max_grid_window": "max_grid_kwh",
    "no_charge_window": None,
    "no_discharge_window": None,
    "no_op": None,
}

DIRECTIVE_TYPES = frozenset(VALUE_FIELD)


@dataclass(frozen=True)
class Battery:
    capacity_kwh: float
    initial_energy_kwh: float
    minimum_energy_kwh: float
    max_charge_kwh_per_hour: float
    max_discharge_kwh_per_hour: float


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    demand: tuple[float, ...]
    solar: tuple[float, ...]
    tariff: tuple[float, ...]
    battery: Battery


@dataclass(frozen=True)
class Directive:
    """A guardrail-approved directive, ready to be folded into the LP."""

    note_index: int
    directive_type: str
    hours: tuple[int, ...] = ()
    value: float | None = None
    explanation: str = ""

    @property
    def applies(self) -> bool:
        return self.directive_type != "no_op"

    def structured_adjustment(self) -> dict[str, Any] | None:
        if self.directive_type == "no_op":
            return None
        adjustment: dict[str, Any] = {"hours": list(self.hours)}
        field = VALUE_FIELD[self.directive_type]
        if field is not None:
            adjustment[field] = self.value
        return adjustment


def no_op(note_index: int, explanation: str) -> Directive:
    return Directive(note_index=note_index, directive_type="no_op", explanation=explanation)


@dataclass
class Overlay:
    """Per-hour constraint arrays — the single place directives touch the math."""

    effective_solar: list[float]
    min_floor: list[float]
    grid_cap: list[float]
    charge_ok: list[bool]
    discharge_ok: list[bool]

    @classmethod
    def base(cls, scenario: Scenario) -> Overlay:
        battery = scenario.battery
        return cls(
            effective_solar=list(scenario.solar),
            min_floor=[battery.minimum_energy_kwh] * N_HOURS,
            grid_cap=[math.inf] * N_HOURS,
            charge_ok=[True] * N_HOURS,
            discharge_ok=[True] * N_HOURS,
        )

    def has_raised_floor(self, battery: Battery) -> bool:
        return any(f > battery.minimum_energy_kwh + 1e-12 for f in self.min_floor)

    def has_grid_cap(self) -> bool:
        return any(math.isfinite(c) for c in self.grid_cap)


@dataclass(frozen=True)
class PlanHour:
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: str
    battery_kwh: float
    battery_energy_after_kwh: float


@dataclass
class SolveResult:
    plan: list[PlanHour]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    feasible: bool = True
    relaxations: tuple[str, ...] = ()
