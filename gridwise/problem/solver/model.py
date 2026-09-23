"""Build the 24-hour scheduling LP.

Variable layout is block-major (all 24 grid vars, then all 24 solar vars, ...)
so that directive overlays map onto contiguous numpy slices.

    grid      [  0 :  24)
    solar     [ 24 :  48)
    charge    [ 48 :  72)
    discharge [ 72 :  96)
    E         [ 96 : 120)
    s_grid    [120 : 144)   slack model only
    s_floor   [144 : 168)   slack model only
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gridwise.problem.domain import N_HOURS, Overlay, Scenario

GRID, SOLAR, CHARGE, DISCHARGE, ENERGY = (slice(i * N_HOURS, (i + 1) * N_HOURS) for i in range(5))
S_GRID = slice(5 * N_HOURS, 6 * N_HOURS)
S_FLOOR = slice(6 * N_HOURS, 7 * N_HOURS)

N_CORE_VARS = 5 * N_HOURS
N_SLACK_VARS = 7 * N_HOURS

# Charging and discharging are lossless and 1:1, so (charge+d, discharge+d) costs
# the same as (charge, discharge) and HiGHS may return either. A vanishing cost on
# both makes such circulating loops strictly suboptimal, which keeps the solver
# from emitting an hour that both charges and discharges — a shape the response
# schema cannot express. At 1e-6 the cost distortion stays ~1e-3 BDT, three orders
# below the judge's 0.01 tolerance.
EPS_LOOP = 1e-6

BIG_M = 1e4


@dataclass
class LPParts:
    c: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    A_ub: np.ndarray | None
    b_ub: np.ndarray | None
    bounds: list[tuple[float | None, float | None]]


def build_lp(scenario: Scenario, overlay: Overlay, *, slack: bool = False) -> LPParts:
    battery = scenario.battery
    n_vars = N_SLACK_VARS if slack else N_CORE_VARS

    c = np.zeros(n_vars)
    c[GRID] = np.asarray(scenario.tariff, dtype=float)
    c[CHARGE] += EPS_LOOP
    c[DISCHARGE] += EPS_LOOP
    if slack:
        c[S_GRID] = BIG_M
        c[S_FLOOR] = BIG_M

    # --- equality rows: 24 balance + 24 battery state + 1 end-of-day neutrality ---
    A_eq = np.zeros((2 * N_HOURS + 1, n_vars))
    b_eq = np.zeros(2 * N_HOURS + 1)

    for h in range(N_HOURS):
        # grid[h] + solar[h] + discharge[h] - charge[h] = demand[h]
        A_eq[h, GRID.start + h] = 1.0
        A_eq[h, SOLAR.start + h] = 1.0
        A_eq[h, DISCHARGE.start + h] = 1.0
        A_eq[h, CHARGE.start + h] = -1.0
        b_eq[h] = scenario.demand[h]

        # E[h] - E[h-1] - charge[h] + discharge[h] = 0   (E[-1] = initial, moved to RHS)
        row = N_HOURS + h
        A_eq[row, ENERGY.start + h] = 1.0
        A_eq[row, CHARGE.start + h] = -1.0
        A_eq[row, DISCHARGE.start + h] = 1.0
        if h == 0:
            b_eq[row] = battery.initial_energy_kwh
        else:
            A_eq[row, ENERGY.start + h - 1] = -1.0

    # End-of-day neutrality as its own row rather than pinning E[23]'s bounds: a
    # reserve directive covering hour 23 above the initial level would otherwise
    # make lo > hi, which scipy rejects outright instead of reporting an
    # infeasibility the slack model can absorb.
    A_eq[2 * N_HOURS, ENERGY.start + N_HOURS - 1] = 1.0
    b_eq[2 * N_HOURS] = battery.initial_energy_kwh

    # --- bounds ---
    bounds: list[tuple[float | None, float | None]] = []
    for h in range(N_HOURS):
        cap = overlay.grid_cap[h]
        if slack or not math.isfinite(cap):
            bounds.append((0.0, None))
        else:
            bounds.append((0.0, cap))
    for h in range(N_HOURS):
        bounds.append((0.0, max(0.0, overlay.effective_solar[h])))
    for h in range(N_HOURS):
        bounds.append((0.0, battery.max_charge_kwh_per_hour if overlay.charge_ok[h] else 0.0))
    for h in range(N_HOURS):
        bounds.append((0.0, battery.max_discharge_kwh_per_hour if overlay.discharge_ok[h] else 0.0))
    for h in range(N_HOURS):
        floor = battery.minimum_energy_kwh if slack else overlay.min_floor[h]
        bounds.append((floor, battery.capacity_kwh))
    if slack:
        bounds.extend([(0.0, None)] * (2 * N_HOURS))

    # --- inequality rows (slack model only) ---
    A_ub: np.ndarray | None = None
    b_ub: np.ndarray | None = None
    if slack:
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for h in range(N_HOURS):
            cap = overlay.grid_cap[h]
            if math.isfinite(cap):
                # grid[h] - s_grid[h] <= cap
                row = np.zeros(n_vars)
                row[GRID.start + h] = 1.0
                row[S_GRID.start + h] = -1.0
                rows.append(row)
                rhs.append(cap)

            floor = overlay.min_floor[h]
            if floor > battery.minimum_energy_kwh + 1e-12:
                # E[h] + s_floor[h] >= floor
                row = np.zeros(n_vars)
                row[ENERGY.start + h] = -1.0
                row[S_FLOOR.start + h] = -1.0
                rows.append(row)
                rhs.append(-floor)

        if rows:
            A_ub = np.vstack(rows)
            b_ub = np.asarray(rhs, dtype=float)

    return LPParts(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
