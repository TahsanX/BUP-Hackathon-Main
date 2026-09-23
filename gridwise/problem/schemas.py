"""API contract for POST /optimize-energy.

Field names and shapes are fixed by the official problem statement; the judge
harness matches them exactly, so nothing here may be renamed.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

N_HOURS = 24

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


class HourInput(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    capacity_kwh: float = Field(ge=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    def inconsistency(self) -> str | None:
        """Well-formed but physically impossible levels -> 422, not 400."""
        if self.initial_energy_kwh > self.capacity_kwh:
            return "initial_energy_kwh must not exceed capacity_kwh"
        if self.minimum_energy_kwh > self.capacity_kwh:
            return "minimum_energy_kwh must not exceed capacity_kwh"
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            return "initial_energy_kwh must not be below minimum_energy_kwh"
        return None


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=N_HOURS, max_length=N_HOURS)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, notes: list[str]) -> list[str]:
        if any(not note.strip() for note in notes):
            raise ValueError("operator_notes entries must be non-empty")
        return notes

    @field_validator("hours")
    @classmethod
    def _hours_cover_full_day(cls, hours: list[HourInput]) -> list[HourInput]:
        if {h.hour for h in hours} != set(range(N_HOURS)):
            raise ValueError("hours must contain each hour 0..23 exactly once")
        return hours

    def hours_sorted(self) -> list[HourInput]:
        return sorted(self.hours, key=lambda h: h.hour)


class DirectiveInterpretationOut(BaseModel):
    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretationOut]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
