"""End-to-end HTTP behaviour with a stand-in provider — no network, no key."""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from gridwise.api import app as app_module
from gridwise.llm.chain import reset_cooldowns
from gridwise.llm.types import LLMRequest, LLMResponse, ServerError
from tests.cases import CASES

SAMPLE = CASES[0]  # SAMPLE-01: one solar_reduction + one distractor

INTERPRETATION = {
    "interpretations": [
        {
            "note_index": 0,
            "directive_type": "solar_reduction",
            "hours": [12, 13],
            "value": 0.25,
            "explanation": "panel cleaning",
        },
        {
            "note_index": 1,
            "directive_type": "no_op",
            "hours": [],
            "value": None,
            "explanation": "unrelated",
        },
    ]
}


class FakeProvider:
    name = "fake"

    def __init__(self, payload: dict | None = None, fail: bool = False) -> None:
        self._payload = payload if payload is not None else INTERPRETATION
        self._fail = fail

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if self._fail:
            raise ServerError("fake provider is down")
        return LLMResponse(text=json.dumps(self._payload), model="fake-model")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    reset_cooldowns()
    monkeypatch.setattr(app_module, "_providers", [FakeProvider()])
    yield
    reset_cooldowns()


@pytest.fixture
def client():
    with TestClient(app_module.app) as test_client:
        yield test_client


def payload() -> dict:
    return copy.deepcopy(SAMPLE["input"])


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_successful_response_matches_the_contract(client):
    response = client.post("/optimize-energy", json=payload())
    assert response.status_code == 200
    body = response.json()

    assert body["scenario_id"] == SAMPLE["input"]["scenario_id"]
    assert set(body) == {
        "scenario_id",
        "directive_interpretation",
        "hourly_plan",
        "total_grid_kwh",
        "total_cost_bdt",
        "peak_grid_kwh",
        "plan_summary",
    }
    assert len(body["hourly_plan"]) == 24
    assert [entry["hour"] for entry in body["hourly_plan"]] == list(range(24))
    assert isinstance(body["plan_summary"], str) and body["plan_summary"]


def test_interpretation_has_one_entry_per_note_in_order(client):
    body = client.post("/optimize-energy", json=payload()).json()
    entries = body["directive_interpretation"]

    assert [e["note_index"] for e in entries] == [0, 1]
    assert entries[0]["applies"] is True
    assert entries[0]["directive_type"] == "solar_reduction"
    assert entries[0]["structured_adjustment"] == {"hours": [12, 13], "factor": 0.25}
    assert entries[1]["applies"] is False
    assert entries[1]["directive_type"] == "no_op"
    assert entries[1]["structured_adjustment"] is None


def test_directive_is_actually_applied_to_the_schedule(client):
    body = client.post("/optimize-energy", json=payload()).json()
    solar_by_hour = {e["hour"]: e["solar_used_kwh"] for e in body["hourly_plan"]}
    base = {h["hour"]: h["solar_kwh"] for h in SAMPLE["input"]["hours"]}

    for hour in (12, 13):
        assert solar_by_hour[hour] <= base[hour] * 0.25 + 0.01, (
            "solar_reduction was reported but not honoured in the plan"
        )


def test_totals_agree_with_the_returned_plan(client):
    body = client.post("/optimize-energy", json=payload()).json()
    tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in SAMPLE["input"]["hours"]}
    plan = body["hourly_plan"]

    assert body["total_grid_kwh"] == pytest.approx(sum(e["grid_kwh"] for e in plan), abs=0.01)
    assert body["total_cost_bdt"] == pytest.approx(
        sum(e["grid_kwh"] * tariff[e["hour"]] for e in plan), abs=0.01
    )
    assert body["peak_grid_kwh"] == pytest.approx(max(e["grid_kwh"] for e in plan), abs=0.01)


def test_malformed_json_is_rejected_with_400(client):
    response = client.post(
        "/optimize-energy",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "mutate, label",
    [
        (lambda p: p.pop("battery"), "missing-battery"),
        (lambda p: p.__setitem__("operator_notes", []), "no-notes"),
        (lambda p: p.__setitem__("operator_notes", ["a", "b", "c", "d"]), "too-many-notes"),
        (lambda p: p.__setitem__("hours", p["hours"][:23]), "short-day"),
        (lambda p: p.__setitem__("operator_notes", ["   "]), "blank-note"),
        (
            lambda p: p["battery"].__setitem__(
                "initial_energy_kwh", p["battery"]["capacity_kwh"] + 100
            ),
            "initial-energy-exceeds-capacity",
        ),
        (
            lambda p: p["battery"].__setitem__(
                "minimum_energy_kwh", p["battery"]["capacity_kwh"] + 100
            ),
            "minimum-reserve-exceeds-capacity",
        ),
    ],
)
def test_structurally_invalid_requests_are_rejected(client, mutate, label):
    body = payload()
    mutate(body)
    response = client.post("/optimize-energy", json=body)
    assert response.status_code in (400, 422), label


def test_duplicate_hours_are_rejected(client):
    body = payload()
    body["hours"][5] = copy.deepcopy(body["hours"][4])
    assert client.post("/optimize-energy", json=body).status_code == 400


@pytest.mark.parametrize(
    "mutate, label",
    [
        (lambda p: p.pop("scenario_id"), "missing-field"),
        (lambda p: p.__setitem__("scenario_id", 123), "wrong-type"),
    ],
)
def test_structural_errors_are_400_per_problem_statement(client, mutate, label):
    body = payload()
    mutate(body)
    assert client.post("/optimize-energy", json=body).status_code == 400, label


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_400_not_a_crash(client, token):
    raw = json.dumps(payload())
    first_tariff = f'"tariff_bdt_per_kwh": {payload()["hours"][0]["tariff_bdt_per_kwh"]}'
    raw = raw.replace(first_tariff, f'"tariff_bdt_per_kwh": {token}', 1)
    assert token in raw
    response = client.post(
        "/optimize-energy", content=raw, headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


def test_initial_energy_below_minimum_is_422(client):
    body = payload()
    body["battery"]["initial_energy_kwh"] = body["battery"]["minimum_energy_kwh"] - 1
    assert client.post("/optimize-energy", json=body).status_code == 422


def test_zero_capacity_battery_runs_idle(client):
    body = payload()
    body["battery"].update(capacity_kwh=0, initial_energy_kwh=0, minimum_energy_kwh=0)
    response = client.post("/optimize-energy", json=body)
    assert response.status_code == 200
    assert all(h["battery_action"] == "idle" for h in response.json()["hourly_plan"])


def test_total_provider_failure_still_returns_a_valid_plan(client, monkeypatch):
    monkeypatch.setattr(app_module, "_providers", [FakeProvider(fail=True)])

    response = client.post("/optimize-energy", json=payload())
    assert response.status_code == 200

    body = response.json()
    entries = body["directive_interpretation"]
    assert [e["directive_type"] for e in entries] == ["no_op", "no_op"]
    assert len(body["hourly_plan"]) == 24
    assert app_module._status_counts["failed"] >= 1


def test_diagnostics_exposes_interpretation_status(client):
    client.post("/optimize-energy", json=payload())
    body = client.get("/diagnostics").json()

    assert body["providers"] == ["fake"]
    assert body["interpretation_status"].get("interpreted", 0) >= 1
