"""The model-facing shape of a directive interpretation.

Deliberately permissive. Strict pydantic parsing here would reject a whole batch
over one bad field and cost us every note in the request; instead we accept
loosely and let the deterministic guardrail downgrade offending notes one by one.

A single generic `value` field is used for every directive type rather than
type-specific keys — one uniform shape is markedly easier for small models (the
local fallback is a 3B) to emit correctly. `Directive.structured_adjustment()`
maps it back to the official per-type key on the way out.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RawDirective(BaseModel):
    model_config = ConfigDict(extra="ignore")

    note_index: Any = None
    directive_type: Any = None
    # Models are asked for the two clock hours and the half-open expansion is
    # done in code. Small models reliably read "9 AM to 11 AM" but just as
    # reliably expand it to [9, 10, 11]; taking the arithmetic away from them
    # removed most of the interpretation errors. `hours` stays accepted because
    # a model that volunteers the list anyway should not be thrown away.
    start_hour: Any = None
    end_hour: Any = None
    hours: Any = None
    value: Any = None
    # For solar_reduction: whether `value` is the share that REMAINS ("to 20%")
    # or the share that is LOST ("by 80%"). Same reasoning as start/end — asking
    # the model to do 1-x inverts it often enough to matter, so it reports what
    # the text said and the subtraction happens here.
    value_basis: Any = None
    explanation: Any = ""


class RawDirectiveBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    interpretations: list[RawDirective] = Field(default_factory=list)


JSON_SCHEMA_HINT = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "directive_type": {
                        "type": "string",
                        "enum": [
                            "solar_reduction",
                            "minimum_battery_reserve",
                            "no_charge_window",
                            "no_discharge_window",
                            "max_grid_window",
                            "no_op",
                        ],
                    },
                    "start_hour": {"type": ["integer", "null"]},
                    "end_hour": {"type": ["integer", "null"]},
                    "value": {"type": ["number", "null"]},
                    "value_basis": {
                        "type": ["string", "null"],
                        "enum": ["remaining", "removed", None],
                    },
                    "explanation": {"type": "string"},
                },
                "required": [
                    "note_index",
                    "directive_type",
                    "start_hour",
                    "end_hour",
                    "explanation",
                ],
            },
        }
    },
    "required": ["interpretations"],
}


# Sent as Gemini's responseSchema (OpenAPI subset: `nullable`, no type unions).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "directive_type": JSON_SCHEMA_HINT["properties"]["interpretations"]["items"][
                        "properties"
                    ]["directive_type"],
                    "start_hour": {"type": "integer", "nullable": True},
                    "end_hour": {"type": "integer", "nullable": True},
                    "value": {"type": "number", "nullable": True},
                    "value_basis": {
                        "type": "string",
                        "nullable": True,
                        "enum": ["remaining", "removed", "percent_of_capacity"],
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["note_index", "directive_type", "explanation"],
            },
        }
    },
    "required": ["interpretations"],
}
