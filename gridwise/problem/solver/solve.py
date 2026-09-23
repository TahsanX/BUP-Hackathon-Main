"""Solve the LP and turn the raw solution into a schedule the judge can replay.

Nothing here reports an LP variable directly. `grid` is re-derived from the
energy-balance equation and `battery_energy_after_kwh` from a forward replay of
the battery, so the totals the judge recalculates from `hourly_plan` agree with
the ones we report by construction rather than by luck.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from gridwise.problem.domain import N_HOURS, Overlay, PlanHour, Scenario, SolveResult
from gridwise.problem.solver.model import (
    CHARGE,
    DISCHARGE,
    ENERGY,
    SOLAR,
    build_lp,
)

ROUND_DP = 6
ACTION_EPS = 1e-9
TOL = 0.01


def solve(scenario: Scenario, overlay: Overlay) -> SolveResult:
    parts = build_lp(scenario, overlay, slack=False)
    res = linprog(
        c=parts.c,
        A_ub=parts.A_ub,
        b_ub=parts.b_ub,
        A_eq=parts.A_eq,
        b_eq=parts.b_eq,
        bounds=parts.bounds,
        method="highs",
    )
    if res.success:
        return _materialize(scenario, overlay, res.x, feasible=True, relaxations=())

    # A genuine infeasibility means a directive was misread: organizer scoring
    # scenarios are guaranteed solvable. Relax only the soft directives (grid cap
    # and the raised part of the reserve) and flag it — never the physics, since
    # a plan that breaks physics fails the judge's replay outright.
    relaxed = build_lp(scenario, overlay, slack=True)
    res = linprog(
        c=relaxed.c,
        A_ub=relaxed.A_ub,
        b_ub=relaxed.b_ub,
        A_eq=relaxed.A_eq,
        b_eq=relaxed.b_eq,
        bounds=relaxed.bounds,
        method="highs",
    )
    if res.success:
        return _materialize(
            scenario,
            overlay,
            res.x,
            feasible=False,
            relaxations=("max_grid_window", "minimum_battery_reserve"),
        )

    return grid_only_plan(scenario, overlay)


def _materialize(
    scenario: Scenario,
    overlay: Overlay,
    x: np.ndarray,
    *,
    feasible: bool,
    relaxations: tuple[str, ...],
) -> SolveResult:
    effective_solar = np.asarray(overlay.effective_solar, dtype=float)

    solar = np.round(np.clip(x[SOLAR], 0.0, None), ROUND_DP)
    solar = np.minimum(solar, effective_solar)

    charge = np.clip(x[CHARGE], 0.0, None)
    discharge = np.clip(x[DISCHARGE], 0.0, None)

    # Netting. Both the balance row and the state row depend only on
    # (discharge - charge), so collapsing an hour onto a single direction changes
    # neither, keeps |net| inside the rate limit it came from, and can never
    # create a banned action (a banned charge is already 0, so net <= 0).
    net = np.round(charge - discharge, ROUND_DP)

    solar, grid, energy = _derive(scenario, solar, net)
    net, solar, grid, energy = _repair_end_of_day(scenario, overlay, net, solar, grid, energy)

    plan: list[PlanHour] = []
    for h in range(N_HOURS):
        if net[h] > ACTION_EPS:
            action, magnitude = "charge", float(net[h])
        elif net[h] < -ACTION_EPS:
            action, magnitude = "discharge", float(-net[h])
        else:
            action, magnitude = "idle", 0.0
        plan.append(
            PlanHour(
                hour=h,
                grid_kwh=float(grid[h]),
                solar_used_kwh=float(solar[h]),
                battery_action=action,
                battery_kwh=round(magnitude, ROUND_DP),
                battery_energy_after_kwh=float(energy[h]),
            )
        )

    return SolveResult(
        plan=plan,
        total_grid_kwh=round(float(sum(p.grid_kwh for p in plan)), ROUND_DP),
        total_cost_bdt=round(
            float(sum(p.grid_kwh * scenario.tariff[p.hour] for p in plan)), ROUND_DP
        ),
        peak_grid_kwh=round(max(p.grid_kwh for p in plan), ROUND_DP),
        feasible=feasible,
        relaxations=relaxations,
    )


def _derive(
    scenario: Scenario, solar: np.ndarray, net: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover solar/grid/energy from the battery plan so balance holds exactly."""
    demand = np.asarray(scenario.demand, dtype=float)

    solar = solar.copy()
    residual = demand + net - solar
    # Negative residual would mean free energy with nowhere to go; curtail solar
    # instead of clamping grid, which would silently break the balance equation.
    surplus = np.minimum(residual, 0.0)
    solar = np.round(np.clip(solar + surplus, 0.0, None), ROUND_DP)

    grid = np.round(np.clip(demand + net - solar, 0.0, None), ROUND_DP)

    energy = np.empty(N_HOURS)
    level = scenario.battery.initial_energy_kwh
    for h in range(N_HOURS):
        level = round(level + float(net[h]), ROUND_DP)
        energy[h] = level

    return solar, grid, energy


def _repair_end_of_day(
    scenario: Scenario,
    overlay: Overlay,
    net: np.ndarray,
    solar: np.ndarray,
    grid: np.ndarray,
    energy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Absorb any floating-point residue on the end-of-day neutrality rule.

    Should be a no-op in practice; it exists so a rounding artifact can never
    fail the judge's strictest equality check.
    """
    initial = scenario.battery.initial_energy_kwh
    gap = round(float(energy[-1]) - initial, ROUND_DP)
    if abs(gap) <= 1e-9:
        return net, solar, grid, energy

    # Cheapest hour first: the correction is absorbed where grid energy costs least.
    for h in sorted(range(N_HOURS), key=lambda i: scenario.tariff[i]):
        candidate = net.copy()
        candidate[h] = round(float(candidate[h]) - gap, ROUND_DP)
        cand_solar, cand_grid, cand_energy = _derive(scenario, solar, candidate)
        if abs(float(cand_energy[-1]) - initial) <= 1e-9 and _is_legal(
            scenario, overlay, candidate, cand_solar, cand_grid, cand_energy
        ):
            return candidate, cand_solar, cand_grid, cand_energy

    return net, solar, grid, energy


def _is_legal(
    scenario: Scenario,
    overlay: Overlay,
    net: np.ndarray,
    solar: np.ndarray,
    grid: np.ndarray,
    energy: np.ndarray,
) -> bool:
    battery = scenario.battery
    for h in range(N_HOURS):
        flow = float(net[h])
        if flow > ACTION_EPS:
            if not overlay.charge_ok[h] or flow > battery.max_charge_kwh_per_hour + TOL:
                return False
        elif flow < -ACTION_EPS:
            if not overlay.discharge_ok[h] or -flow > battery.max_discharge_kwh_per_hour + TOL:
                return False
        if grid[h] < -TOL or grid[h] > overlay.grid_cap[h] + TOL:
            return False
        if solar[h] < -TOL or solar[h] > overlay.effective_solar[h] + TOL:
            return False
        if energy[h] < overlay.min_floor[h] - TOL or energy[h] > battery.capacity_kwh + TOL:
            return False
    return True


def grid_only_plan(scenario: Scenario, overlay: Overlay) -> SolveResult:
    """Last resort: buy everything from the grid, battery idle all day.

    Always satisfies the physics and end-of-day neutrality, so it is a valid (if
    expensive) answer — strictly better than returning an error.
    """
    plan: list[PlanHour] = []
    level = scenario.battery.initial_energy_kwh
    for h in range(N_HOURS):
        used = round(min(overlay.effective_solar[h], scenario.demand[h]), ROUND_DP)
        plan.append(
            PlanHour(
                hour=h,
                grid_kwh=round(scenario.demand[h] - used, ROUND_DP),
                solar_used_kwh=used,
                battery_action="idle",
                battery_kwh=0.0,
                battery_energy_after_kwh=level,
            )
        )

    return SolveResult(
        plan=plan,
        total_grid_kwh=round(sum(p.grid_kwh for p in plan), ROUND_DP),
        total_cost_bdt=round(sum(p.grid_kwh * scenario.tariff[p.hour] for p in plan), ROUND_DP),
        peak_grid_kwh=round(max(p.grid_kwh for p in plan), ROUND_DP),
        feasible=False,
        relaxations=("grid_only_fallback",),
    )
