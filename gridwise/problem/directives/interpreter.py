"""Operator notes -> guardrail-approved directives.

All notes go out in one batched call (quota and latency both matter), but the
guardrail works per note, so one unusable field costs one note rather than the
whole case.

The `status` field is the fix for the defect that sank the previous build: a
genuine "this note is irrelevant" and "the model never answered" both used to
collapse into an indistinguishable all-no_op response with an HTTP 200 on top.
They are now different states, and the failed one is loud in the logs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from gridwise.core.config import Settings
from gridwise.llm.chain import complete_structured
from gridwise.llm.provider import Provider
from gridwise.llm.types import ProviderTrace
from gridwise.problem.directives.prompt import render_notes, system_prompt
from gridwise.problem.directives.schema import RawDirectiveBatch
from gridwise.problem.directives.validate import validate_batch
from gridwise.problem.domain import Battery, Directive, no_op

logger = logging.getLogger(__name__)

Status = Literal["interpreted", "degraded", "failed"]

NO_MODEL = "No language model was reachable; this note was not applied to the schedule."


@dataclass
class InterpretationOutcome:
    directives: list[Directive]
    status: Status
    provider: str | None
    trace: ProviderTrace


async def interpret_notes(
    notes: Sequence[str],
    battery: Battery,
    *,
    providers: Sequence[Provider],
    settings: Settings | None = None,
) -> InterpretationOutcome:
    batch, trace = await complete_structured(
        system=system_prompt(),
        user=render_notes(list(notes), battery),
        schema=RawDirectiveBatch,
        providers=providers,
        settings=settings,
    )

    if batch is None:
        logger.error("interpretation failed for %d note(s): %s", len(notes), trace.summary())
        return InterpretationOutcome(
            directives=[no_op(i, NO_MODEL) for i in range(len(notes))],
            status="failed",
            provider=None,
            trace=trace,
        )

    result = validate_batch(batch, n_notes=len(notes), battery=battery)
    status: Status = "interpreted" if result.clean else "degraded"
    if result.downgraded:
        logger.warning("interpretation degraded: %s", trace.summary())

    return InterpretationOutcome(
        directives=result.directives,
        status=status,
        provider=trace.winner,
        trace=trace,
    )
