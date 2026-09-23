"""Every failure mode the chain is supposed to survive, without a network."""

from __future__ import annotations

import asyncio

import pytest

from gridwise.core.config import Settings
from gridwise.llm.chain import complete_structured, reset_cooldowns
from gridwise.llm.types import (
    AuthFailure,
    LLMRequest,
    LLMResponse,
    ProviderTimeout,
    RateLimited,
    ServerError,
)
from gridwise.problem.directives.schema import RawDirectiveBatch

GOOD_JSON = (
    '{"interpretations": [{"note_index": 0, "directive_type": "no_charge_window", '
    '"hours": [2, 3], "value": null, "explanation": "maintenance"}]}'
)

SETTINGS = Settings(
    provider_order=("a", "b", "c"),
    total_budget_seconds=20.0,
    cooldown_seconds=30.0,
    timeouts={"a": 5.0, "b": 5.0, "c": 5.0},
)


class ScriptedProvider:
    """Replays a fixed script of outcomes; raises if asked more often than scripted."""

    def __init__(self, name: str, script: list) -> None:
        self.name = name
        self._script = list(script)
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self._script:
            raise ServerError(f"{self.name}: script exhausted")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return LLMResponse(text=step, model=f"{self.name}-model")


@pytest.fixture(autouse=True)
def _clean_cooldowns():
    reset_cooldowns()
    yield
    reset_cooldowns()


def run(providers, repair_attempts: int = 1):
    return asyncio.run(
        complete_structured(
            system="s",
            user="u",
            schema=RawDirectiveBatch,
            providers=providers,
            settings=SETTINGS,
            repair_attempts=repair_attempts,
        )
    )


def test_first_healthy_provider_wins_and_others_are_untouched():
    first = ScriptedProvider("a", [GOOD_JSON])
    second = ScriptedProvider("b", [GOOD_JSON])

    parsed, trace = run([first, second])

    assert parsed is not None
    assert trace.winner == "a"
    assert second.calls == []


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ProviderTimeout("slow"), id="timeout"),
        pytest.param(RateLimited("429"), id="rate-limited"),
        pytest.param(ServerError("502"), id="server-error"),
        pytest.param(AuthFailure("bad key"), id="auth-failure"),
        pytest.param(RuntimeError("adapter bug"), id="unexpected"),
    ],
)
def test_failure_moves_to_the_next_provider(failure):
    first = ScriptedProvider("a", [failure])
    second = ScriptedProvider("b", [GOOD_JSON])

    parsed, trace = run([first, second])

    assert parsed is not None
    assert trace.winner == "b"
    assert len(first.calls) == 1, "a failed provider must not be retried on the same key"


def test_malformed_output_gets_one_repair_on_the_same_provider():
    first = ScriptedProvider("a", ["not json at all", GOOD_JSON])
    second = ScriptedProvider("b", [GOOD_JSON])

    parsed, trace = run([first, second])

    assert parsed is not None
    assert trace.winner == "a"
    assert trace.repaired is True
    assert len(first.calls) == 2
    assert "could not be used" in first.calls[1].user
    assert second.calls == []


def test_persistently_malformed_provider_fails_over():
    first = ScriptedProvider("a", ["garbage", "still garbage"])
    second = ScriptedProvider("b", [GOOD_JSON])

    parsed, trace = run([first, second])

    assert parsed is not None
    assert trace.winner == "b"
    assert len(first.calls) == 2


def test_all_providers_down_returns_none_not_an_exception():
    providers = [
        ScriptedProvider("a", [RateLimited("429")]),
        ScriptedProvider("b", [ProviderTimeout("slow")]),
        ScriptedProvider("c", [ServerError("500")]),
    ]

    parsed, trace = run(providers)

    assert parsed is None
    assert trace.winner is None
    assert len(trace.attempts) == 3


def test_rate_limited_provider_is_skipped_on_the_next_request():
    first = ScriptedProvider("a", [RateLimited("429"), GOOD_JSON])
    second = ScriptedProvider("b", [GOOD_JSON, GOOD_JSON])

    run([first, second])
    _, trace = run([first, second])

    assert trace.winner == "b"
    assert [a.outcome for a in trace.attempts] == ["cooldown", "ok"]
    assert len(first.calls) == 1, "cooled-down provider must not be called again"


def test_exhausted_budget_stops_the_chain():
    settings = Settings(
        provider_order=("a", "b"),
        total_budget_seconds=0.0,
        timeouts={"a": 5.0, "b": 5.0},
    )
    first = ScriptedProvider("a", [GOOD_JSON])

    parsed, trace = asyncio.run(
        complete_structured(
            system="s",
            user="u",
            schema=RawDirectiveBatch,
            providers=[first],
            settings=settings,
        )
    )

    assert parsed is None
    assert trace.attempts[0].outcome == "budget_exhausted"
    assert first.calls == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(f"```json\n{GOOD_JSON}\n```", id="fenced"),
        pytest.param(f"Sure! Here you go:\n{GOOD_JSON}\nHope that helps.", id="wrapped-in-prose"),
        pytest.param(
            '{"interpretations": [{"note_index": 0, "directive_type": "no_op", '
            '"hours": [], "value": null, "explanation": "x"},]}',
            id="trailing-comma",
        ),
    ],
)
def test_json_is_recovered_from_untidy_output(text):
    parsed, trace = run([ScriptedProvider("a", [text])])

    assert parsed is not None
    assert trace.winner == "a"
    assert trace.repaired is False


def test_empty_reply_is_treated_as_malformed():
    first = ScriptedProvider("a", ["", ""])
    second = ScriptedProvider("b", [GOOD_JSON])

    parsed, trace = run([first, second])

    assert parsed is not None
    assert trace.winner == "b"
