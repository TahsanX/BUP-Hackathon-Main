"""Each directive must bind in the schedule, not merely be echoed back."""

from __future__ import annotations

import random

import pytest

from gridwise.problem.directives.overlay import build_overlay
from gridwise.problem.domain import Battery, Directive, N_HOURS, Overlay, Scenario
from gridwise.problem.solver.replay import replay
from gridwise.problem.solver.solve import solve

TOL = 0.01

BATTERY = Battery(
    capacity_kwh=200.0,
    initial_energy_kwh=120.0,
    minimum_energy_kwh=40.0,
    max_charge_kwh_per_hour=50.0,
    max_discharge_kwh_per_hour=50.0,
)

# Cheap overnight, expensive evening: the shape that makes arbitrage worthwhile.
TARIFF = tuple([5.0] * 6 + [10.0] * 10 + [28.0] * 5 + [8.0] * 3)
DEMAND = tuple([100.0] * N_HOURS)
SOLAR = tuple([0.0] * 7 + [40.0] * 10 + [0.0] * 7)

SCENARIO = Scenario(
    scenario_id="EDGE", demand=DEMAND, solar=SOLAR, tariff=TARIFF, battery=BATTERY
)


def run(*directives: Directive):
    overlay = build_overlay(SCENARIO, list(directives))
    result = solve(SCENARIO, overlay)
    return overlay, result


def assert_valid(overlay: Overlay, result):
    violations = replay(
        SCENARIO,
        overlay,
        result.plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )
    assert violations == []


def directive(kind: str, hours, value=None) -> Directive:
    return Directive(note_index=0, directive_type=kind, hours=tuple(hours), value=value)


def test_baseline_uses_the_battery_for_arbitrage():
    overlay, result = run()
    assert_valid(overlay, result)
    assert any(p.battery_action == "discharge" for p in result.plan)
    assert result.plan[-1].battery_energy_after_kwh == pytest.approx(
        BATTERY.initial_energy_kwh, abs=TOL
    )


def test_no_charge_window_is_enforced():
    hours = (2, 3, 4)
    overlay, result = run(directive("no_charge_window", hours))
    assert_valid(overlay, result)
    for hour in hours:
        assert result.plan[hour].battery_action != "charge"


def test_no_discharge_window_is_enforced():
    hours = (16, 17, 18)
    overlay, result = run(directive("no_discharge_window", hours))
    assert_valid(overlay, result)
    for hour in hours:
        assert result.plan[hour].battery_action != "discharge"


def test_max_grid_window_caps_import():
    hours, cap = (16, 17, 18), 60.0
    overlay, result = run(directive("max_grid_window", hours, cap))
    assert_valid(overlay, result)
    for hour in hours:
        assert result.plan[hour].grid_kwh <= cap + TOL


def test_minimum_battery_reserve_holds_the_floor():
    hours, reserve = (16, 17, 18), 150.0
    overlay, result = run(directive("minimum_battery_reserve", hours, reserve))
    assert_valid(overlay, result)
    for hour in hours:
        assert result.plan[hour].battery_energy_after_kwh >= reserve - TOL


def test_solar_reduction_limits_usable_solar():
    hours, factor = (10, 11, 12), 0.25
    overlay, result = run(directive("solar_reduction", hours, factor))
    assert_valid(overlay, result)
    for hour in hours:
        assert result.plan[hour].solar_used_kwh <= SOLAR[hour] * factor + TOL


def test_overlapping_solar_reductions_stack_multiplicatively():
    overlay = build_overlay(
        SCENARIO,
        [
            Directive(0, "solar_reduction", (10, 11), 0.5),
            Directive(1, "solar_reduction", (11, 12), 0.5),
        ],
    )
    assert overlay.effective_solar[10] == pytest.approx(SOLAR[10] * 0.5)
    assert overlay.effective_solar[11] == pytest.approx(SOLAR[11] * 0.25)
    assert overlay.effective_solar[12] == pytest.approx(SOLAR[12] * 0.5)


def test_overlapping_reserves_take_the_strictest_floor():
    overlay = build_overlay(
        SCENARIO,
        [
            Directive(0, "minimum_battery_reserve", (18,), 90.0),
            Directive(1, "minimum_battery_reserve", (18,), 150.0),
        ],
    )
    assert overlay.min_floor[18] == 150.0


def test_overlapping_grid_caps_take_the_strictest_ceiling():
    overlay = build_overlay(
        SCENARIO,
        [
            Directive(0, "max_grid_window", (18,), 120.0),
            Directive(1, "max_grid_window", (18,), 70.0),
        ],
    )
    assert overlay.grid_cap[18] == 70.0


def test_several_directives_apply_together():
    overlay, result = run(
        Directive(0, "solar_reduction", (10, 11), 0.3),
        Directive(1, "no_charge_window", (2, 3), None),
        Directive(2, "max_grid_window", (18, 19), 80.0),
    )
    assert_valid(overlay, result)
    assert result.feasible
    for hour in (2, 3):
        assert result.plan[hour].battery_action != "charge"
    for hour in (18, 19):
        assert result.plan[hour].grid_kwh <= 80.0 + TOL


def test_contradictory_reserve_relaxes_instead_of_crashing():
    # A reserve above the starting level on the final hour cannot coexist with
    # end-of-day neutrality. Physics must survive; the soft directive gives way.
    overlay, result = run(directive("minimum_battery_reserve", (23,), 190.0))

    assert result.feasible is False
    assert result.relaxations

    physics_only = replay(
        SCENARIO,
        Overlay.base(SCENARIO),
        result.plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )
    assert physics_only == [], f"relaxation broke the physics: {physics_only}"


def test_a_binding_directive_never_makes_the_plan_cheaper():
    _, baseline = run()
    _, constrained = run(directive("no_discharge_window", tuple(range(16, 21))))
    assert constrained.total_cost_bdt >= baseline.total_cost_bdt - TOL


@pytest.mark.parametrize("seed", range(40))
def test_random_scenarios_produce_replayable_schedules(seed):
    rng = random.Random(seed)
    capacity = rng.uniform(120, 400)
    battery = Battery(
        capacity_kwh=capacity,
        initial_energy_kwh=rng.uniform(0.3, 0.7) * capacity,
        minimum_energy_kwh=rng.uniform(0.05, 0.2) * capacity,
        max_charge_kwh_per_hour=rng.uniform(20, 80),
        max_discharge_kwh_per_hour=rng.uniform(20, 80),
    )
    scenario = Scenario(
        scenario_id=f"RAND-{seed}",
        demand=tuple(rng.uniform(60, 220) for _ in range(N_HOURS)),
        solar=tuple(rng.choice([0.0, rng.uniform(0, 180)]) for _ in range(N_HOURS)),
        tariff=tuple(rng.uniform(4, 34) for _ in range(N_HOURS)),
        battery=battery,
    )

    directives = []
    if rng.random() < 0.5:
        start = rng.randrange(0, 20)
        directives.append(
            Directive(0, "solar_reduction", tuple(range(start, start + 3)), rng.uniform(0, 1))
        )
    if rng.random() < 0.5:
        start = rng.randrange(0, 20)
        kind = rng.choice(["no_charge_window", "no_discharge_window"])
        directives.append(Directive(1, kind, tuple(range(start, start + 3)), None))

    overlay = build_overlay(scenario, directives)
    result = solve(scenario, overlay)

    violations = replay(
        scenario,
        overlay,
        result.plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    )
    assert violations == [], f"seed {seed}: {violations}"
