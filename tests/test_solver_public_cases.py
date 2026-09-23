"""Phase-2 gate: the solver must reproduce every public reference cost.

Runs entirely offline with ground-truth directives, so a broken optimizer is
caught without spending a single LLM call.
"""

from __future__ import annotations

import pytest

from gridwise.problem.directives.overlay import build_overlay
from gridwise.problem.solver.replay import replay
from gridwise.problem.solver.solve import solve
from tests.cases import CASE_IDS, CASES, directives_from_expected, scenario_from_input

TOL = 0.01


@pytest.fixture(params=CASES, ids=CASE_IDS)
def solved(request):
    case = request.param
    scenario = scenario_from_input(case["input"])
    directives = directives_from_expected(case["expected_output"]["directive_interpretation"])
    overlay = build_overlay(scenario, directives)
    return case, scenario, overlay, solve(scenario, overlay)


def test_matches_reference_cost(solved):
    case, _, _, result = solved
    expected = float(case["expected_output"]["total_cost_bdt"])
    assert result.total_cost_bdt == pytest.approx(expected, abs=TOL), (
        f"{case['id']}: got {result.total_cost_bdt}, reference {expected}"
    )


def test_schedule_is_feasible(solved):
    case, _, _, result = solved
    assert result.feasible, f"{case['id']} needed a relaxation: {result.relaxations}"


def test_schedule_passes_independent_replay(solved):
    case, scenario, overlay, result = solved
    violations = replay(
        scenario,
        overlay,
        result.plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )
    assert violations == [], f"{case['id']}: {violations}"


def test_reported_totals_match_plan(solved):
    _, scenario, _, result = solved
    assert result.total_grid_kwh == pytest.approx(
        sum(p.grid_kwh for p in result.plan), abs=TOL
    )
    assert result.total_cost_bdt == pytest.approx(
        sum(p.grid_kwh * scenario.tariff[p.hour] for p in result.plan), abs=TOL
    )
    assert result.peak_grid_kwh == pytest.approx(max(p.grid_kwh for p in result.plan), abs=TOL)


def test_no_hour_both_charges_and_discharges(solved):
    _, _, _, result = solved
    for entry in result.plan:
        assert entry.battery_action in {"charge", "discharge", "idle"}
        if entry.battery_action == "idle":
            assert entry.battery_kwh == 0.0
        else:
            assert entry.battery_kwh > 0.0
