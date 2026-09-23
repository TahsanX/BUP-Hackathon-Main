"""Guardrails must isolate damage: one bad note never costs the others."""

from __future__ import annotations

import pytest

from gridwise.problem.directives.schema import RawDirectiveBatch
from gridwise.problem.directives.validate import validate_batch
from gridwise.problem.domain import Battery

BATTERY = Battery(
    capacity_kwh=200.0,
    initial_energy_kwh=120.0,
    minimum_energy_kwh=40.0,
    max_charge_kwh_per_hour=50.0,
    max_discharge_kwh_per_hour=50.0,
)

GOOD = {
    "note_index": 0,
    "directive_type": "solar_reduction",
    "hours": [12, 13],
    "value": 0.25,
    "explanation": "cleaning",
}


def run(entries: list[dict], n_notes: int = 1):
    return validate_batch(
        RawDirectiveBatch.model_validate({"interpretations": entries}),
        n_notes=n_notes,
        battery=BATTERY,
    )


def test_accepts_a_well_formed_directive():
    result = run([GOOD])
    assert result.clean
    directive = result.directives[0]
    assert directive.directive_type == "solar_reduction"
    assert directive.hours == (12, 13)
    assert directive.value == 0.25
    assert directive.applies is True
    assert directive.structured_adjustment() == {"hours": [12, 13], "factor": 0.25}


def test_no_op_carries_null_adjustment_and_applies_false():
    result = run([{"note_index": 0, "directive_type": "no_op", "explanation": "unrelated"}])
    directive = result.directives[0]
    assert directive.applies is False
    assert directive.structured_adjustment() is None


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"directive_type": "shed_load"}, id="unknown-type"),
        pytest.param({"hours": [12, 24]}, id="hour-out-of-range"),
        pytest.param({"hours": [12, -1]}, id="negative-hour"),
        pytest.param({"hours": []}, id="empty-hours"),
        pytest.param({"hours": "12,13"}, id="hours-not-a-list"),
        pytest.param({"hours": [True, False]}, id="booleans-as-hours"),
        pytest.param({"value": 1.4}, id="factor-above-one"),
        pytest.param({"value": -0.2}, id="negative-factor"),
        pytest.param({"value": None}, id="missing-value"),
        pytest.param({"value": "abc"}, id="non-numeric-value"),
    ],
)
def test_invalid_field_downgrades_only_that_note(override):
    bad = {**GOOD, "note_index": 1, **override}
    result = run([GOOD, bad], n_notes=2)

    assert result.downgraded is True
    assert result.directives[0].directive_type == "solar_reduction"  # untouched
    assert result.directives[1].directive_type == "no_op"
    assert result.directives[1].structured_adjustment() is None


def test_reserve_above_capacity_is_rejected():
    entry = {
        "note_index": 0,
        "directive_type": "minimum_battery_reserve",
        "hours": [18],
        "value": BATTERY.capacity_kwh + 1,
        "explanation": "",
    }
    assert run([entry]).directives[0].directive_type == "no_op"


def test_hours_are_deduplicated_and_sorted():
    entry = {**GOOD, "hours": [13, 12, 13]}
    assert run([entry]).directives[0].hours == (12, 13)


def test_missing_note_is_filled_with_no_op():
    result = run([GOOD], n_notes=3)
    assert [d.directive_type for d in result.directives] == ["solar_reduction", "no_op", "no_op"]
    assert result.downgraded is True


def test_duplicate_and_out_of_range_indices_are_dropped():
    entries = [GOOD, {**GOOD, "value": 0.9}, {**GOOD, "note_index": 7}]
    result = run(entries, n_notes=1)
    assert len(result.directives) == 1
    assert result.directives[0].value == 0.25  # first wins, later duplicate ignored


def test_none_batch_yields_one_no_op_per_note():
    result = validate_batch(None, n_notes=2, battery=BATTERY)
    assert [d.directive_type for d in result.directives] == ["no_op", "no_op"]
    assert result.downgraded is True


def test_entries_are_always_in_note_index_order():
    entries = [{**GOOD, "note_index": 2}, {**GOOD, "note_index": 0}]
    result = run(entries, n_notes=3)
    assert [d.note_index for d in result.directives] == [0, 1, 2]


@pytest.mark.parametrize(
    "start, end, expected",
    [
        pytest.param(13, 15, (13, 14), id="1pm-3pm"),
        pytest.param(9, 11, (9, 10), id="9am-11am"),
        pytest.param(18, 21, (18, 19, 20), id="6pm-9pm"),
        pytest.param(2, 5, (2, 3, 4), id="02-05"),
        pytest.param(23, 2, (0, 1, 23), id="wraps-midnight-sorts-ascending"),
        pytest.param(22, 24, (22, 23), id="end-24-means-midnight"),
        pytest.param(0, 1, (0,), id="single-hour"),
    ],
)
def test_window_is_expanded_half_open(start, end, expected):
    entry = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "start_hour": start,
        "end_hour": end,
        "explanation": "",
    }
    assert run([entry]).directives[0].hours == expected


@pytest.mark.parametrize(
    "start, end",
    [
        pytest.param(5, 5, id="empty-window"),
        pytest.param(-1, 4, id="negative-start"),
        pytest.param(4, 25, id="end-past-midnight-boundary"),
        pytest.param(24, 2, id="start-out-of-range"),
    ],
)
def test_unusable_window_downgrades_the_note(start, end):
    entry = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "start_hour": start,
        "end_hour": end,
        "explanation": "",
    }
    assert run([entry]).directives[0].directive_type == "no_op"


def test_explicit_hours_still_accepted_when_no_window_given():
    entry = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "hours": [14, 15],
        "explanation": "",
    }
    assert run([entry]).directives[0].hours == (14, 15)


def test_window_wins_over_a_contradictory_hours_list():
    # The model's own expansion is the part we do not trust; ours is derived.
    entry = {
        "note_index": 0,
        "directive_type": "no_charge_window",
        "start_hour": 9,
        "end_hour": 11,
        "hours": [9, 10, 11],
        "explanation": "",
    }
    assert run([entry]).directives[0].hours == (9, 10)


def test_string_numerics_are_coerced():
    entry = {**GOOD, "hours": ["12", "13"], "value": "0.25"}
    directive = run([entry]).directives[0]
    assert directive.hours == (12, 13)
    assert directive.value == 0.25
