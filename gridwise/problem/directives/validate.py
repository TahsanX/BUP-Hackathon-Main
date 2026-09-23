"""Deterministic guardrails over model output.

Two invariants drive everything here:

1. Exactly one entry per operator note, in note_index order. Missing, duplicate
   or out-of-range indices are repaired rather than raised.
2. A bad field costs only its own note. Downgrading one note to no_op keeps the
   other notes' directives — the previous build discarded the whole batch on any
   failure, which turned one bad field into a zero for the entire case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gridwise.problem.domain import (
    DIRECTIVE_TYPES,
    N_HOURS,
    Battery,
    Directive,
    no_op,
)
from gridwise.problem.directives.schema import RawDirective, RawDirectiveBatch

UNINTERPRETED = "Could not interpret this note reliably; it was not applied to the schedule."
MISSING = "The model returned no interpretation for this note."


@dataclass
class GuardrailResult:
    directives: list[Directive]
    downgraded: bool

    @property
    def clean(self) -> bool:
        return not self.downgraded


def validate_batch(
    batch: RawDirectiveBatch | None, *, n_notes: int, battery: Battery
) -> GuardrailResult:
    by_index: dict[int, RawDirective] = {}
    if batch is not None:
        for raw in batch.interpretations:
            index = _as_int(raw.note_index)
            if index is None or not 0 <= index < n_notes or index in by_index:
                continue  # unusable mapping; the note falls through to MISSING below
            by_index[index] = raw

    directives: list[Directive] = []
    downgraded = False

    for index in range(n_notes):
        raw = by_index.get(index)
        if raw is None:
            directives.append(no_op(index, MISSING))
            downgraded = True
            continue

        directive = _validate_one(raw, index, battery)
        if directive is None:
            directives.append(no_op(index, UNINTERPRETED))
            downgraded = True
        else:
            directives.append(directive)

    return GuardrailResult(directives=directives, downgraded=downgraded)


def _validate_one(raw: RawDirective, index: int, battery: Battery) -> Directive | None:
    kind = raw.directive_type
    if not isinstance(kind, str) or kind not in DIRECTIVE_TYPES:
        return None

    explanation = raw.explanation if isinstance(raw.explanation, str) else ""

    if kind == "no_op":
        return no_op(index, explanation or "This note does not affect today's energy schedule.")

    hours = _window_hours(raw)
    if not hours:
        return None

    value = _as_value(kind, raw.value, raw.value_basis, battery)
    if value is None and kind in {"solar_reduction", "minimum_battery_reserve", "max_grid_window"}:
        return None

    return Directive(
        note_index=index,
        directive_type=kind,
        hours=hours,
        value=value,
        explanation=explanation,
    )


def _as_int(value: object) -> int | None:
    # bool is an int subclass in Python; True must not slip through as hour 1.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _window_hours(raw: RawDirective) -> tuple[int, ...]:
    """Prefer expanding start/end ourselves; fall back to an explicit list."""
    start = _as_int(raw.start_hour)
    end = _as_int(raw.end_hour)
    if start is not None and end is not None:
        expanded = _expand_window(start, end)
        if expanded:
            return expanded
    return _as_hours(raw.hours)


def _expand_window(start: int, end: int) -> tuple[int, ...]:
    """Half-open [start, end), wrapping past midnight. Start-inclusive, end-exclusive."""
    if not (0 <= start < N_HOURS) or not (0 <= end <= N_HOURS):
        return ()
    if start == end:
        return ()  # an empty window is a misread, not an instruction

    hours: list[int] = []
    hour = start
    while hour != end % N_HOURS:
        hours.append(hour)
        hour = (hour + 1) % N_HOURS
        if len(hours) >= N_HOURS:
            break
    return tuple(sorted(hours))


def _as_hours(value: object) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    hours: set[int] = set()
    for item in value:
        hour = _as_int(item)
        if hour is None or not 0 <= hour < N_HOURS:
            return ()  # one bad hour invalidates the window; do not silently drop it
        hours.add(hour)
    return tuple(sorted(hours))


def _as_value(kind: str, value: object, basis: object, battery: Battery) -> float | None:
    if kind in {"no_charge_window", "no_discharge_window"}:
        return None

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().rstrip("%"))
        except ValueError:
            return None
    else:
        return None

    if not math.isfinite(number):
        return None

    if kind == "solar_reduction":
        # "70" means 70 percent; "1.4" is simply not a fraction. Only whole
        # numbers are percent-shaped, so a fractional out-of-range value stays
        # invalid and downgrades the note.
        if 1.0 < number <= 100.0 and number.is_integer():
            number /= 100.0
        if not 0.0 <= number <= 1.0:
            return None
        if isinstance(basis, str) and basis.strip().lower() in {"removed", "reduction", "lost"}:
            number = 1.0 - number
        return number
    if kind == "minimum_battery_reserve":
        if isinstance(basis, str) and basis.strip().lower() in {
            "percent_of_capacity",
            "fraction_of_capacity",
        }:
            if number > 1.0:
                number /= 100.0  # "40" percent rather than 0.4
            if not 0.0 <= number <= 1.0:
                return None
            number *= battery.capacity_kwh
        return number if 0.0 <= number <= battery.capacity_kwh else None
    if kind == "max_grid_window":
        return number if number >= 0.0 else None
    return None
