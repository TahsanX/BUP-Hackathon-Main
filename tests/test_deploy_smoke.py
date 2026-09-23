"""Proof that the DEPLOYED service actually interprets notes with a model.

This is the test the previous build did not have. Its whole suite passed while
the shipped service answered every scenario with all-no_op, because a
platform-conditional branch disabled the LLM in production and nothing ever
exercised the deployed URL. A green unit suite says nothing about what the
judge will hit.

    GRIDWISE_BASE_URL=https://your-deployment pytest tests/test_deploy_smoke.py -v
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

BASE_URL = os.environ.get("GRIDWISE_BASE_URL", "").rstrip("/")

pytestmark = pytest.mark.skipif(
    not BASE_URL, reason="set GRIDWISE_BASE_URL to run the deployment smoke test"
)

TIMEOUT = 30.0

# One unmistakable directive plus one unmistakable distractor. Worded unlike the
# public pack so a lookup table cannot pass this.
SCENARIO = {
    "scenario_id": "SMOKE-01",
    "operator_notes": [
        "Crews are washing the array from 10 AM until 1 PM; treat usable solar as "
        "about a quarter of forecast while they work.",
        "The library will extend its opening hours next semester.",
    ],
    "hours": [
        {
            "hour": h,
            "demand_kwh": 150.0,
            "solar_kwh": 120.0 if 8 <= h <= 16 else 0.0,
            "tariff_bdt_per_kwh": 26.0 if 18 <= h <= 20 else 6.0,
        }
        for h in range(24)
    ],
    "battery": {
        "capacity_kwh": 200.0,
        "initial_energy_kwh": 100.0,
        "minimum_energy_kwh": 40.0,
        "max_charge_kwh_per_hour": 50.0,
        "max_discharge_kwh_per_hour": 50.0,
    },
}


@pytest.fixture(scope="module")
def response_body():
    started = time.monotonic()
    response = httpx.post(f"{BASE_URL}/optimize-energy", json=SCENARIO, timeout=TIMEOUT)
    elapsed = time.monotonic() - started
    assert response.status_code == 200, response.text
    body = response.json()
    body["_elapsed"] = elapsed
    return body


def test_health_is_reachable_and_ready():
    response = httpx.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_the_deployed_service_really_interpreted_the_notes(response_body):
    entries = response_body["directive_interpretation"]
    assert len(entries) == 2

    kinds = [entry["directive_type"] for entry in entries]
    assert kinds != ["no_op", "no_op"], (
        "the deployment answered with all-no_op: the model is not reachable from "
        "production, or something is short-circuiting the interpretation path"
    )
    assert kinds[0] == "solar_reduction"
    assert entries[0]["applies"] is True


def test_the_distractor_is_still_ignored(response_body):
    second = response_body["directive_interpretation"][1]
    assert second["directive_type"] == "no_op"
    assert second["applies"] is False
    assert second["structured_adjustment"] is None


def test_the_interpreted_directive_is_applied_to_the_plan(response_body):
    adjustment = response_body["directive_interpretation"][0]["structured_adjustment"]
    hours = set(adjustment["hours"])
    factor = adjustment["factor"]
    assert hours == {10, 11, 12}, f"expected a half-open 10:00-13:00 window, got {sorted(hours)}"
    assert factor == pytest.approx(0.25, abs=0.05)

    solar_by_hour = {e["hour"]: e["solar_used_kwh"] for e in response_body["hourly_plan"]}
    for hour in hours:
        assert solar_by_hour[hour] <= 120.0 * factor + 0.01, (
            "the directive was reported but the schedule ignored it"
        )


def test_totals_agree_with_the_returned_plan(response_body):
    plan = response_body["hourly_plan"]
    tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in SCENARIO["hours"]}

    assert len(plan) == 24
    assert response_body["total_grid_kwh"] == pytest.approx(
        sum(e["grid_kwh"] for e in plan), abs=0.01
    )
    assert response_body["total_cost_bdt"] == pytest.approx(
        sum(e["grid_kwh"] * tariff[e["hour"]] for e in plan), abs=0.01
    )
    assert response_body["peak_grid_kwh"] == pytest.approx(
        max(e["grid_kwh"] for e in plan), abs=0.01
    )


def test_end_of_day_battery_returns_to_its_starting_level(response_body):
    final = response_body["hourly_plan"][-1]["battery_energy_after_kwh"]
    assert final == pytest.approx(SCENARIO["battery"]["initial_energy_kwh"], abs=0.01)


def test_response_lands_inside_the_judge_timeout(response_body):
    assert response_body["_elapsed"] < TIMEOUT
