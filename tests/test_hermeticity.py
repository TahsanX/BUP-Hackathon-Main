"""Structural guarantees that keep the LLM core reusable and the tests offline.

If these fail, the building block has quietly grown a dependency on this
particular problem — which is exactly what makes it non-reusable next time.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gridwise.problem.directives.overlay import build_overlay
from gridwise.problem.directives.schema import RawDirectiveBatch
from gridwise.problem.directives.validate import validate_batch
from gridwise.problem.solver.replay import replay
from gridwise.problem.solver.solve import solve
from tests.cases import CASES, directives_from_expected, scenario_from_input

PACKAGE = Path(__file__).resolve().parent.parent / "gridwise"

REUSABLE_LAYERS = ["llm", "core"]
PURE_MODULES = [
    "problem/solver/model.py",
    "problem/solver/solve.py",
    "problem/solver/replay.py",
    "problem/directives/overlay.py",
    "problem/directives/validate.py",
    "problem/domain.py",
]


def _sources(relative: str):
    root = PACKAGE / relative
    return sorted(root.rglob("*.py")) if root.is_dir() else [root]


@pytest.mark.parametrize("layer", REUSABLE_LAYERS)
def test_reusable_layers_never_import_the_problem_layer(layer):
    offenders = [
        path.relative_to(PACKAGE).as_posix()
        for path in _sources(layer)
        if "gridwise.problem" in path.read_text()
    ]
    assert offenders == [], (
        f"gridwise/{layer}/ must stay problem-agnostic so it can be lifted into "
        f"another project unchanged; these files reach into the problem layer: {offenders}"
    )


@pytest.mark.parametrize("relative", PURE_MODULES)
def test_optimisation_path_does_no_io(relative):
    source = (PACKAGE / relative).read_text()
    for forbidden in ("import httpx", "import requests", "import os", "open("):
        assert forbidden not in source, f"{relative} should stay pure, found {forbidden!r}"


def test_whole_pipeline_runs_without_any_api_key(monkeypatch):
    for variable in (
        "GEMINI_API_KEY", "GEMINI_API_KEYS",
        "GROQ_API_KEY", "GROQ_API_KEYS",
        "NVIDIA_API_KEY", "NVIDIA_API_KEYS",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)

    case = CASES[0]
    scenario = scenario_from_input(case["input"])
    directives = directives_from_expected(case["expected_output"]["directive_interpretation"])
    overlay = build_overlay(scenario, directives)
    result = solve(scenario, overlay)

    assert result.feasible
    assert replay(
        scenario,
        overlay,
        result.plan,
        total_grid_kwh=result.total_grid_kwh,
        total_cost_bdt=result.total_cost_bdt,
        peak_grid_kwh=result.peak_grid_kwh,
    ) == []


def test_no_api_key_is_present_in_the_test_environment():
    leaked = [
        name
        for name in (
            "GEMINI_API_KEY", "GEMINI_API_KEYS",
            "GROQ_API_KEY", "GROQ_API_KEYS",
            "NVIDIA_API_KEY", "NVIDIA_API_KEYS",
            "OPENAI_API_KEY",
        )
        if os.environ.get(name)
    ]
    if leaked:
        pytest.skip(f"real credentials present in this shell: {leaked}")


def test_guardrail_needs_no_network():
    result = validate_batch(
        RawDirectiveBatch.model_validate({"interpretations": []}),
        n_notes=1,
        battery=scenario_from_input(CASES[0]["input"]).battery,
    )
    assert result.directives[0].directive_type == "no_op"
