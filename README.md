# GridWise

LLM-assisted 24-hour campus energy scheduling API.

Given a day of demand, solar and tariff data, a battery spec, and 1–3 free-text
operator notes, the service interprets the notes into structured directives,
validates them deterministically, and returns a cost-minimal 24-hour schedule
that obeys them.

```
notes ──► LLM chain ──► guardrails ──► overlay ──► LP optimiser ──► replay ──► response
          gemini          per-note      per-hour     scipy/HiGHS     independent
          → groq          downgrade     arrays                       re-check
          → ollama
```

---

## Quickstart (local, from a clean checkout)

```bash
git clone <this-repo> && cd GridWise-LLM
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add GEMINI_API_KEY and/or GROQ_API_KEY
set -a && . ./.env && set +a

uvicorn gridwise.api.app:app --port 8000
```

Then, in another shell:

```bash
curl -s localhost:8000/health
# {"status":"ok"}

curl -s -X POST localhost:8000/optimize-energy \
  -H 'content-type: application/json' \
  -d "$(python3 -c "import json;print(json.dumps(json.load(open('tests/fixtures/public_sample_cases.json'))['cases'][0]['input']))")" \
  | python3 -m json.tool | head -30
```

Expected for sample case 1: note 0 → `solar_reduction`, hours `[12, 13]`,
factor `0.25`; note 1 → `no_op`; 24 plan entries; `total_cost_bdt` `38365.00`.

## Quickstart (Docker)

```bash
docker build -t gridwise:2.0.0 .        # ~3 GB: the local fallback model is baked in
docker run -p 8000:8000 --env-file .env gridwise:2.0.0
curl -s localhost:8000/health
```

or `docker compose up --build`. To build a small image without the local model:

```bash
docker build --build-arg LOCAL_MODEL=none -t gridwise:slim .
docker run -e LLM_PROVIDER_ORDER=gemini,groq -p 8000:8000 --env-file .env gridwise:slim
```

That image has no local fallback, so the chain ends at Groq — keep `ollama` out
of `LLM_PROVIDER_ORDER` there.

---

## API

| Endpoint | Behaviour |
|---|---|
| `GET /health` | `{"status":"ok"}`. Never probes a model, so readiness cannot be held hostage by a third party. |
| `POST /optimize-energy` | Interpretation + 24-hour plan. Schema exactly as specified in the problem statement. |
| `GET /diagnostics` | Operational only: configured providers and a count of interpretation outcomes. |

Status codes: `200` success · `400` malformed JSON or structurally invalid
(missing/mistyped field, NaN/Infinity) · `422` well-formed but physically
impossible battery levels · `500` controlled internal error (never a stack
trace, never a credential).

## Directive types

| Type | `structured_adjustment` | Effect on the model |
|---|---|---|
| `solar_reduction` | `{hours, factor}` | `effective_solar[h] *= factor` (factor is what REMAINS) |
| `minimum_battery_reserve` | `{hours, minimum_energy_kwh}` | raises the battery floor in those hours |
| `no_charge_window` | `{hours}` | charge bound forced to 0 |
| `no_discharge_window` | `{hours}` | discharge bound forced to 0 |
| `max_grid_window` | `{hours, max_grid_kwh}` | caps hourly grid import |
| `no_op` | `null` | nothing |

Windows are whole hours, start-inclusive and end-exclusive: "1 PM to 3 PM" is
`[13, 14]`.

---

## Architecture

**`gridwise/llm/` and `gridwise/core/` are problem-agnostic.** They take a
prompt pair plus a pydantic model and return a validated instance; they contain
no reference to energy, batteries or schedules. A test asserts that neither
layer imports `gridwise/problem/`, so the block can be lifted into another
project by swapping only the prompt and the schema.

```
gridwise/
  core/config.py            env-driven settings; no platform detection anywhere
  llm/
    types.py                typed error taxonomy (auth / rate-limit / timeout / malformed)
    chain.py                failover, circuit breaker, time budget, one repair retry
    json_guard.py           recovers JSON from fences, prose and trailing commas
    providers/              gemini, groq, ollama adapters (async httpx)
  problem/
    directives/prompt.py    system prompt + few-shot   <- the swap point
    directives/validate.py  deterministic guardrails, per-note downgrade
    directives/overlay.py   directives -> per-hour constraint arrays
    solver/model.py         LP construction (120 vars, 49 equality rows)
    solver/solve.py         solve, net, reconcile, end-of-day repair
    solver/replay.py        independent rule checker, written from the spec
    pipeline.py             request -> response
  api/app.py                FastAPI surface
```

### How the LLM is used

All notes go out in **one batched call**. The model returns a uniform
`{note_index, directive_type, hours, value, explanation}` per note; a single
generic `value` field is markedly easier for small models to emit correctly than
type-specific keys, and it is mapped back to the official per-type key on the
way out.

Model output is treated as untrusted. Nothing reaches the optimiser until
`directives/validate.py` has checked the type against the allowed set, the hours
for uniqueness/range/ordering, and the numeric payload against its bounds.
`applies` is *derived* from the directive type rather than trusted, so an
`applies=true` on a `no_op` is structurally impossible.

### Degradation ladder

Every layer degrades rather than failing:

| Situation | Result |
|---|---|
| Gemini answers | `status=interpreted` |
| Gemini down → Groq answers | `status=interpreted` |
| Both down → local Ollama answers | `status=interpreted` (no quota, no network) |
| One note's field is unusable | that note → `no_op`, others still applied, `status=degraded` |
| No provider reachable | all `no_op` + unconstrained solve, `status=failed`, logged as an error |
| LP infeasible | soft directives relaxed, `feasible=False`; physics never relaxed |
| Relaxed LP also fails | grid-only plan — expensive but valid |

Providers are tried **in order with a 30s cooldown**, not raced. Racing would
burn every quota on every request, which is the failure being defended against.
A hard 20s wall-clock budget keeps the whole step inside the 30s judge ceiling.

### Multi-key pooling

A free-tier key's rate limit is per-key, not per-provider, so a single Gemini
key and a single Groq key means the *entire chain* is one request away from
"all candidates failed" the moment either key's RPM limit trips — a burst of
5 requests was enough to reproduce this in testing. `GEMINI_API_KEYS`/
`GROQ_API_KEYS` accept a comma-separated pool; `build_providers()` expands
each key into its own named candidate (`gemini`, `gemini#2`, ...), so the
existing per-candidate cooldown in `chain.py` isolates a rate-limited key
instead of removing the provider. This mirrors the load-balancing approach
one of the two accepted submissions to this challenge used in production.

NVIDIA NIM's free endpoints add a second axis: one key, several model ids.
`NVIDIA_API_KEYS` x `NVIDIA_MODELS` is a cross product, so a single key with
three model ids still becomes three independent candidates (`nvidia`,
`nvidia#2`, `nvidia#3`) — each with its own cooldown, so a 429 or a
temporarily-overloaded model doesn't take the other two down with it.

### Choosing the local model

`scripts/eval_interpretation.py` scores a provider on 15 paraphrased notes
written unlike the public pack. Measured over three runs on CPU:

| model | image cost | accuracy | mean latency |
|---|---|---|---|
| `gemma3:4b` | 3.3 GB | 14/15 | 1.8s |
| **`qwen2.5:3b-instruct`** (default) | **1.9 GB** | **14/15** | **1.3s** |
| `qwen2.5:1.5b-instruct` | 1.0 GB | 14/15 | 0.9s |

Accuracy is flat across the three, so the 3B wins on image size. The 1.5B is
faster still, but it misread "2 in the afternoon" as hour 2 — an AM/PM error
the larger models did not make, and the kind of mistake the guardrails cannot
catch. Its remaining 0.4s is invisible anyway: the local model runs third, so
reaching it already means ~12s of upstream timeouts have elapsed.

Those numbers only converged *after* the calculations moved out of the prompt.
Before that, the same eval scored the 4B at 60% and the 1.5B at 80%; the gap
between model sizes was almost entirely arithmetic the models should never have
been asked to do. Re-run the eval before changing `OLLAMA_MODEL`.

### Optimiser

A linear program (`scipy.optimize.linprog`, HiGHS): 5 continuous variables per
hour — grid, solar used, charge, discharge, stored energy — minimising
`Σ grid[h] × tariff[h]`. Equality rows carry the hourly energy balance, the
battery recursion, and end-of-day neutrality.

Two details that matter:

- **No hour both charges and discharges.** Lossless 1:1 storage makes
  `(charge+δ, discharge+δ)` cost-identical, and the response schema cannot
  express it. A 1e-6 tie-break makes such loops strictly suboptimal, and a
  netting pass guarantees it regardless — netting preserves `discharge − charge`,
  so balance, energy path and cost are all unchanged.
- **Reported totals cannot drift.** `grid_kwh` is re-derived from the balance
  equation and `battery_energy_after_kwh` from a forward replay, then the totals
  are summed from the plan itself — so the judge's recalculation agrees by
  construction.

---

## Tests

```bash
pytest                                   # 161 tests, no API key, no network
pytest tests/test_solver_public_cases.py # the gate: all 10 public reference costs
```

| Suite | Covers |
|---|---|
| `test_solver_public_cases.py` | all 10 official cases reproduce the reference cost exactly (0.00 BDT deviation) and pass independent replay |
| `test_solver_edges.py` | every directive binds; overlaps stack correctly; contradictory input relaxes instead of crashing; 40 random scenarios replay clean |
| `test_guardrails.py` | each malformed field downgrades only its own note |
| `test_llm_failover.py` | timeout, 429, 401, malformed, empty, all-down, cooldown, budget exhaustion |
| `test_api_contract.py` | exact response schema, 400/422 split, directive actually applied, totals agree |
| `test_hermeticity.py` | the LLM core never imports the problem layer; the whole pipeline runs with no key |
| `test_deploy_smoke.py` | **runs against the deployed URL** and asserts the service really interpreted the notes |

The deployment smoke test is the important one:

```bash
GRIDWISE_BASE_URL=https://your-deployment pytest tests/test_deploy_smoke.py -v
python scripts/load_check.py --url https://your-deployment --n 20
```

It fails loudly if production answers with all-`no_op`. A previous iteration of
this service shipped with a platform-conditional branch that disabled the LLM in
production; the unit suite stayed green while the deployed endpoint silently
ignored every operator note. No configuration in this codebase branches on the
hosting platform, and this test exists so that class of bug cannot ship again.

---

## Configuration

All settings are environment variables — see `.env.example`. The ones that
matter:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER_ORDER` | `gemini,groq,nvidia,ollama` | chain order; providers without credentials are skipped |
| `GEMINI_API_KEYS` / `GROQ_API_KEYS` | – | comma-separated key pool per provider (a single `GEMINI_API_KEY`/`GROQ_API_KEY` also works). Each key is its own chain candidate, so one key's rate limit cools down alone instead of taking the whole provider out — see "Multi-key pooling" below |
| `NVIDIA_API_KEYS` / `NVIDIA_MODELS` | – | build.nvidia.com free endpoints; one key fanned out across several model ids, each becoming its own chain candidate |
| `OLLAMA_MODEL` | `gemma3:4b` | local fallback, baked into the image at build time |
| `LLM_TOTAL_BUDGET_SECONDS` | `20` | wall-clock ceiling for interpretation |
| `LLM_COOLDOWN_SECONDS` | `30` | how long a failing provider is skipped |

**Secrets.** No key is baked into the image or committed to the repo. Error
messages pass through a redactor that strips key-shaped strings before they
reach a log or a response body.

## Dependencies

`fastapi` · `uvicorn` · `pydantic` · `numpy` · `scipy` · `httpx`. No LLM vendor
SDK: each provider is a thin async HTTP adapter, which keeps the dependency
surface small and makes adding a provider a single file.

## Known limitations

- No response caching. Repeated identical notes re-query the model.
- The local fallback needs ~55s to cold-load its weights (a warm call is ~1.3s),
  so the container warms it in the background at startup. It is last in the
  chain because it is CPU-bound, not because it is inaccurate.
- Wraparound windows ("11 PM to 2 AM") rely on the model emitting the hour list
  directly; they are validated but not independently re-derived from the text.
- `/diagnostics` counters are per-process and reset on restart.

## Reference material

`parallel-solutions/` holds two other teams' submissions to the same challenge,
kept for comparison. They are independent git clones and are excluded from this
repository and from the test run.
