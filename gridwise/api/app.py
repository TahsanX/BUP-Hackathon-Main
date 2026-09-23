"""FastAPI surface: GET /health and POST /optimize-energy."""

from __future__ import annotations

import logging
import os
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from gridwise.core.config import Settings
from gridwise.llm.providers import build_providers
from gridwise.llm.types import redact
from gridwise.problem.pipeline import run_optimization
from gridwise.problem.schemas import HealthResponse, OptimizeRequest, OptimizeResponse

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gridwise")

_settings = Settings.from_env()
_providers = build_providers(_settings)

# Interpretation outcomes since boot. Exposed on /diagnostics so a silent
# collapse to all-no_op is observable instead of invisible.
_status_counts: Counter[str] = Counter()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    logger.info(
        "provider chain: %s | budget %.1fs",
        " -> ".join(p.name for p in _providers) or "NONE",
        _settings.total_budget_seconds,
    )
    if not _providers:
        logger.error("no LLM provider is configured; interpretation cannot run")
    yield


app = FastAPI(title="GridWise", version="2.0.0", lifespan=_lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    # Never probes a model: readiness must not depend on a third party, and the
    # judge expects this inside 60s of start.
    return HealthResponse()


@app.get("/diagnostics")
async def diagnostics() -> dict:
    return {
        "providers": [p.name for p in _providers],
        "interpretation_status": dict(_status_counts),
    }


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(request: OptimizeRequest) -> OptimizeResponse:
    response, outcome = await run_optimization(
        request, providers=_providers, settings=_settings
    )

    _status_counts[outcome.status] += 1
    logger.info(
        "%s notes=%d status=%s provider=%s cost=%.2f %s",
        request.scenario_id,
        len(request.operator_notes),
        outcome.status,
        outcome.provider or "none",
        response.total_cost_bdt,
        outcome.trace.summary(),
    )

    return response


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    # 400 for a body that isn't well-formed JSON, 422 for well-formed but
    # semantically wrong — the split the problem statement asks for.
    unparseable = any(e.get("type") == "json_invalid" for e in errors)
    return JSONResponse(
        status_code=400 if unparseable else 422,
        content={
            "detail": "malformed request" if unparseable else "invalid request",
            "errors": [
                {
                    "field": ".".join(str(p) for p in e.get("loc", ())),
                    "message": str(e.get("msg", "")),
                }
                for e in errors
            ],
        },
    )


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": f"internal error: {redact(type(exc).__name__)}"},
    )
