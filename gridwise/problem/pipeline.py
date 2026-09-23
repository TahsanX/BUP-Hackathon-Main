"""Request -> interpretation -> guardrail -> optimisation -> verified response."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from gridwise.core.config import Settings
from gridwise.llm.provider import Provider
from gridwise.problem.directives.interpreter import InterpretationOutcome, interpret_notes
from gridwise.problem.directives.overlay import build_overlay
from gridwise.problem.domain import Battery, Scenario
from gridwise.problem.schemas import (
    DirectiveInterpretationOut,
    HourlyPlanEntry,
    OptimizeRequest,
    OptimizeResponse,
)
from gridwise.problem.solver.replay import replay
from gridwise.problem.solver.solve import grid_only_plan, solve
from gridwise.problem.summary import build_plan_summary

logger = logging.getLogger(__name__)


def scenario_from_request(request: OptimizeRequest) -> Scenario:
    hours = request.hours_sorted()
    return Scenario(
        scenario_id=request.scenario_id,
        demand=tuple(h.demand_kwh for h in hours),
        solar=tuple(h.solar_kwh for h in hours),
        tariff=tuple(h.tariff_bdt_per_kwh for h in hours),
        battery=Battery(
            capacity_kwh=request.battery.capacity_kwh,
            initial_energy_kwh=request.battery.initial_energy_kwh,
            minimum_energy_kwh=request.battery.minimum_energy_kwh,
            max_charge_kwh_per_hour=request.battery.max_charge_kwh_per_hour,
            max_discharge_kwh_per_hour=request.battery.max_discharge_kwh_per_hour,
        ),
    )


async def run_optimization(
    request: OptimizeRequest,
    *,
    providers: Sequence[Provider],
    settings: Settings | None = None,
) -> tuple[OptimizeResponse, InterpretationOutcome]:
    scenario = scenario_from_request(request)

    outcome = await interpret_notes(
        request.operator_notes,
        scenario.battery,
        providers=providers,
        settings=settings,
    )

    overlay = build_overlay(scenario, outcome.directives)
    result = solve(scenario, overlay)

    violations = replay(
        scenario,
        overlay,
        result.plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )
    if violations:
        # The optimiser contradicted the rules it was built from. Ship a plan that
        # is provably valid instead of a cheaper one we cannot stand behind.
        logger.error("solver output failed replay for %s: %s", scenario.scenario_id, violations)
        result = grid_only_plan(scenario, overlay)

    response = OptimizeResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=[
            DirectiveInterpretationOut(
                note_index=d.note_index,
                applies=d.applies,
                directive_type=d.directive_type,
                structured_adjustment=d.structured_adjustment(),
                explanation=d.explanation,
            )
            for d in outcome.directives
        ],
        hourly_plan=[
            HourlyPlanEntry(
                hour=p.hour,
                grid_kwh=p.grid_kwh,
                solar_used_kwh=p.solar_used_kwh,
                battery_action=p.battery_action,
                battery_kwh=p.battery_kwh,
                battery_energy_after_kwh=p.battery_energy_after_kwh,
            )
            for p in result.plan
        ],
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
        plan_summary=build_plan_summary(scenario, outcome.directives, result),
    )

    return response, outcome
