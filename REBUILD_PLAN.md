# GridWise v2 — Scratch Rebuild Plan

> **Status:** Approved · এখনো implement করা হয়নি। Phase 1 থেকে শুরু (§7)।

---

## 1. Context — কেন এই rebuild

আগের prototype (`app/`) হ্যাকাথনে accept হয়নি। তিনটা সাবমিশন পাশাপাশি বিশ্লেষণ করে (তোমারটা, `Naim/bup-energy-optimizer`, `Tonmoy/bup-cse-fest-gridwise`) মূল কারণ পাওয়া গেছে — এবং সেটা প্রম্পট বা অপটিমাইজারের সমস্যা নয়:

```python
# app/llm.py:245-246  ← এটাই সব শেষ করেছে
if os.environ.get("VERCEL"):
    raise RuntimeError("External LLM calls disabled on Vercel")
```

Deploy target ছিল Vercel। ফলে **deployed URL-এ কোনো দিন কোনো LLM call হয়নি** — প্রতিটা request সরাসরি `safe_fallback` → সব note `no_op` → তবু HTTP 200 আর স্বাভাবিক দেখতে schedule। Rubric অনুযায়ী এটা সবচেয়ে কঠোর penalty ("Required LLM absent from interpretation path → not eligible for shortlist"), সাথে প্রতি কেসে Interpretation (25) + Application (25) = **50 পয়েন্ট**।

দ্বিতীয় কারণ: `tests/test_public_cases.py` একটা missing fixture ফাইলের কারণে **collect-ই হয়নি** — অর্থাৎ সবচেয়ে গুরুত্বপূর্ণ end-to-end টেস্ট কখনো চলেনি, তাই bug-টা submission-এর আগে ধরাও পড়েনি। `test_optimizer.py` (15) আর `test_validators.py` (18) পাস করেছে, কিন্তু সেগুলো unit test — **optimizer কখনো ১০টা public reference cost-এর বিপরীতে যাচাই হয়নি**।

**Outcome চাওয়া হচ্ছে:** এমন একটা base যেটা (ক) এই চ্যালেঞ্জের সব কেস পাস করে, (খ) ভবিষ্যতে শুধু few-shot prompt + API key পাল্টে অন্য প্রজেক্টে re-use করা যায়, (গ) প্রতিটা স্তরে backup path আছে যাতে কোনো একক failure পুরো response নষ্ট না করে।

**নেওয়া সিদ্ধান্ত:** Docker container deploy (Vercel নয়) · scratch rebuild · provider chain = Gemini → Groq → local Ollama.

---

## 2. আগের prototype থেকে কী রাখা হচ্ছে

Scratch rebuild মানে ফাইল কপি নয়, কিন্তু চাকা নতুন করে আবিষ্কারও নয়:

| উৎস | কী নেওয়া হবে |
|---|---|
| `app/optimizer.py` (তোমার) | LP formulation টা **কাঠামোগতভাবে সঠিক ছিল** — 120-var scipy HiGHS, 48 eq rows, net-based action reconstruction। এটা reference হিসেবে ব্যবহার করে পরিষ্কার করে আবার লেখা হবে (নিচে §4-এ ৩টা সংশোধনসহ)। |
| `Naim/.../app/engine/validate.py` | Independent replay validator-এর প্যাটার্ন — সমস্যার স্টেটমেন্ট থেকে আলাদা করে লেখা দ্বিতীয় rule-checker। |
| `Naim/.../app/llm/interpreter.py` | `reconcile()` ধারণা — প্রতি note-এ ঠিক একটা entry নিশ্চিত করা, missing হলে ভরাট করা। |
| `Tonmoy/.../app/llm/failover.py` | Typed exception দিয়ে provider routing (`AuthFailure` → skip, `RateLimited`/timeout → next, malformed → এক বার repair retry)। |
| `Tonmoy/.../app/directives/validator.py` | Per-note `_safe_no_op()` downgrade — একটা note খারাপ হলে শুধু সেটা নামানো, বাকিগুলো বাঁচানো। |

---

## 3. Target architecture

কেন্দ্রীয় নীতি: **`gridwise/llm/` energy সম্পর্কে কিছুই জানে না।** ভবিষ্যতে নতুন প্রজেক্টে `problem/` আর প্রম্পট অ্যাসেট পাল্টাবে; `llm/` আর `core/` অপরিবর্তিত থাকবে।

```
gridwise/
  core/                        # PROBLEM-AGNOSTIC
    config.py                  # env: keys, model ids, timeouts, provider order
    errors.py
  llm/                         # PROBLEM-AGNOSTIC — এটাই reusable building block
    types.py                   # LLMRequest, LLMResponse, ProviderError taxonomy
    provider.py                # Protocol: Provider
    providers/gemini.py  groq.py  ollama.py
    chain.py                   # failover + circuit breaker + repair loop + budget
    json_guard.py              # fence stripping, first-JSON extraction, type coercion
  problem/                     # PROBLEM-SPECIFIC
    schemas.py                 # API request/response pydantic models
    directives/
      schema.py                # DirectiveInterpretation, enum, per-type field shapes
      prompt.py                # system prompt + few-shot   ← THE SWAP POINT
      validate.py              # deterministic guardrails + per-note safe downgrade
      overlay.py               # validated directives → per-hour arrays
    solver/
      model.py                 # build_lp(scenario, overlay, slack=False)
      solve.py                 # solve + net + reconcile
      replay.py                # independent validator (written from spec, not solver)
    summary.py                 # plan_summary text (template, NOT LLM — rubric says
                               #   LLM-for-summary-only doesn't satisfy the requirement)
  api/
    app.py  routes.py          # GET /health, POST /optimize-energy
tests/
  fixtures/public_sample_cases.json   ← repo-তে commit করা, path hardcode নয়
Dockerfile  compose.yaml  README.md  .env.example
```

**দুটো কঠিন নিয়ম যা reusability রক্ষা করে:**
1. `llm/`-এর কোনো ফাইল কখনো `problem/` import করবে না (একটা টেস্ট দিয়ে enforce করা হবে)।
2. `solver/` আর `replay.py` pure function — শূন্য I/O, শূন্য network। ফলে API key ছাড়াই ১০টা public case-এ পুরো optimizer টেস্ট করা যাবে।

**স্তর পেরোনো মাত্র দুটো ফাংশন:**

```python
# gridwise/llm/chain.py  — problem-agnostic; T = যেকোনো BaseModel
async def complete_structured(
    *, system: str, user: str, schema: type[T],
    repair_attempts: int = 1,
    providers: Sequence[Provider] | None = None,
) -> tuple[T | None, ProviderTrace]: ...

# gridwise/llm/provider.py
class Provider(Protocol):
    name: str
    async def generate(self, req: LLMRequest) -> LLMResponse: ...   # raw text only
```

Problem layer এভাবে ডাকবে:

```python
raw, trace = await complete_structured(
    system=DIRECTIVE_SYSTEM, user=render_notes(notes), schema=DirectiveBatch)
validated = validate_batch(raw, n_notes=len(notes))  # deterministic; per-note downgrade
overlay   = build_overlay(scenario, validated)
result    = solve(scenario, overlay)                 # pure, offline, unit-testable
```

---

## 4. LP spec — এটাই "সব কেস পাস" হওয়ার কেন্দ্র

Heuristic দিয়ে হবে না: SAMPLE-01 hour 13-এ reference plan `grid 152.5 / solar_used 42.5` (= 170 × 0.25) — ভগ্নাংশ optimum, LP বাধ্যতামূলক।

**Variables** — block-major (hour-major নয়), কারণ directive overlay তখন numpy slice হয়:

```python
N = 24
IDX = {name: np.arange(N) + N*i
       for i, name in enumerate(("grid", "solar", "charge", "discharge", "E"))}  # 120 vars
```

**Overlay arrays** — directive গণিত স্পর্শ করার একমাত্র জায়গা:

| directive | overlay write | overlap rule |
|---|---|---|
| `solar_reduction` | `eff_solar[hours] *= factor` | multiplicative |
| `minimum_battery_reserve` | `min_floor[hours] = max(min_floor, m)` | max |
| `no_charge_window` | `charge_ok[hours] = False` | OR |
| `no_discharge_window` | `discharge_ok[hours] = False` | OR |
| `max_grid_window` | `grid_cap[hours] = min(grid_cap, g)` | min |

Init: `eff_solar = solar[:]`, `min_floor = full(N, battery.minimum_energy_kwh)`, `grid_cap = full(N, inf)`, `charge_ok = discharge_ok = ones(bool)`.

**Objective:** `c[IDX.grid] = tariff`; এবং `c[IDX.charge] += 1e-6`, `c[IDX.discharge] += 1e-6` (কারণ নিচে)।

**A_eq — 49 rows:**

```
balance[h]:   +grid[h] +solar[h] +discharge[h] -charge[h]   = demand[h]
state[h>0]:   +E[h] -E[h-1] -charge[h] +discharge[h]        = 0
state[h==0]:  +E[0]  -charge[0] +discharge[0]               = initial_energy_kwh
eod:          +E[23]                                        = initial_energy_kwh
```

**Bounds:** `grid ∈ [0, grid_cap]`, `solar ∈ [0, eff_solar]`, `charge ∈ [0, max_charge or 0]`, `discharge ∈ [0, max_discharge or 0]`, `E ∈ [min_floor[h], capacity]`।

### আগের কোডের তুলনায় তিনটা সংশোধন

1. **EOD neutrality আলাদা equality row হিসেবে, bounds-pin নয়।** আগের কোড (`app/optimizer.py:93-97`) `E[23]`-এর bound কে `(init_e, init_e)` করে pin করত। সমস্যা: যদি কোনো `minimum_battery_reserve` directive hour 23 কভার করে এবং reserve > initial_energy হয়, তাহলে `lo > hi` হয়ে scipy অদ্ভুত error দেবে, relaxation path-এ যাবে না। Equality row ব্যবহার করলে HiGHS পরিষ্কারভাবে infeasible রিপোর্ট করবে এবং §4.3-এর slack model সেটা সামলাতে পারবে।

2. **`EPS_LOOP = 1e-6` tie-break যোগ করা।** Battery 1:1 lossless, তাই `(charge=c+δ, discharge=d+δ)` একই খরচে একই `E`-path দেয় — HiGHS যেকোনোটা ফেরত দিতে পারে। কিন্তু একই ঘণ্টায় charge>0 **এবং** discharge>0 response schema ভাঙে ("exactly one battery_action")। এই ক্ষুদ্র penalty চক্রাকার loop-কে strictly suboptimal করে দেয়। বিকৃতি সর্বোচ্চ ~1e-3 BDT — 0.01 tolerance-এর চেয়ে তিন ধাপ নিচে।

3. **Netting post-process (guaranteed fix, optimality-preserving):**
   ```python
   net = charge - discharge
   charge, discharge = max(net, 0), max(-net, 0)
   ```
   কেন সবসময় নিরাপদ: balance row আর state row দুটোই শুধু `discharge - charge` এর উপর নির্ভর করে, তাই অপরিবর্তিত; `|net| ≤ max(c,d) ≤` rate limit; আর `charge_ok[h]` false হলে `c=0` তাই `net ≤ 0` — netting কখনো নিষিদ্ধ action **তৈরি** করতে পারে না। `grid` অস্পৃষ্ট, তাই খরচও অপরিবর্তিত।

   Idle threshold: `if c > 1e-9: charge  elif d > 1e-9: discharge  else: idle`, এবং idle হলে `battery_kwh = 0.0` ঠিক শূন্য।

### 4.2 Post-solve reconciliation — reported totals যেন কখনো না মেলে না

Judge নিজে recalculate করে মিলিয়ে দেখে। তাই **LP variable সরাসরি রিপোর্ট করা হবে না** — সব plan থেকেই derive হবে, এই ক্রমে:

```python
solar, charge, discharge = round(x[...], 6) তারপর net (উপরে)
grid[h] = max(0, round(demand[h] + charge[h] - discharge[h] - solar[h], 6))  # balance থেকে
E[h]    = initial_energy থেকে forward replay, netted action দিয়ে                # state থেকে
totals  = sum(grid), sum(grid*tariff), max(grid)   # plan list থেকে, x থেকে নয়
```

এতে judge-এর দুটো সবচেয়ে ঝুঁকিপূর্ণ recalculation স্বতঃসিদ্ধভাবে মিলে যায়। 6 dp রাউন্ডিং-এ ক্রমযোজিত drift ≤ ~2.4e-5 kWh — tolerance-এর ৪০০ গুণ ভেতরে।

**EOD residue repair** (belt-and-braces): replay-এর পর `gap = E[23] - initial`। `|gap| > 1e-9` হলে ঘণ্টাগুলো **ascending tariff** ক্রমে ঘুরে `-gap` প্রয়োগ করো, re-net, re-derive grid, re-replay; প্রথম যে ঘণ্টায় `|E[23]-initial| ≤ 1e-9` **এবং** legality re-check (grid ≥ 0 ও ≤ cap, floor ≤ E ≤ capacity, rate limits, window flags) পাস করে সেটা নাও। প্রতি চেষ্টার মাঝে snapshot/restore। Ascending tariff = সংশোধন শোষণের সবচেয়ে সস্তা জায়গা।

### 4.3 Infeasibility — relax-and-flag, 422 নয়

Rubric বলছে organizer scoring scenario **নিশ্চিতভাবে feasible**। তাই বাস্তব infeasibility মানে গঠনগতভাবেই **LLM কোনো directive ভুল পড়েছে** — scenario অসম্ভব নয়। 422 দিলে ওই কেসে শূন্য; relaxed plan দিলে schedule-validity + cost পয়েন্ট, আর যে directive-গুলো টিকেছে তাদের interpretation পয়েন্ট থাকে। Expected value স্পষ্টতই বেশি।

শুধু **soft** directive শিথিল করো — `max_grid_window` আর `minimum_battery_reserve`-এর উত্থাপিত অংশ। **কখনো physics নয়** (balance, state, capacity, rate limits, EOD neutrality) — physics ভাঙলে judge-এর replay-ই fail করবে, কোনো লাভ নেই।

```
grid[h] - s_g[h] <= grid_cap[h]            # A_ub row, s_g >= 0
-E[h] + s_f[h]   <= -min_floor[h]          # শুধু যেখানে min_floor > base minimum
   ↑ slack model-এ E-এর lower bound আবার battery.minimum_energy_kwh-এ ফিরিয়ে আনতে হবে
```

`BIG_M = 1e4`। আগে nominal solve; শুধু `not res.success` হলে slack সহ re-solve, `feasible=False` সেট করো, আর কোন relaxation হলো তা log করো (response body-তে নয়)।

> **সতর্কতা:** Naim-এর version-এ এই জায়গায় একটা ফাঁক আছে — slack model-এ `E`-এর bound না widened করলে `s_f` অকেজো থাকে আর LP তখনও infeasible। আমাদেরটায় bound widening বাধ্যতামূলক।

---

## 5. LLM chain spec

**Provider order:** Gemini (`gemini-2.0-flash`, native JSON schema) → Groq (`llama-3.3-70b-versatile`, JSON mode) → Ollama (`qwen2.5:3b-instruct`, image-এ baked)।

**একটাই call, সব note একসাথে ব্যাচ করে** (দুই accepted সলিউশনই তাই করেছে) — quota আর latency দুটোই বাঁচে। কিন্তু guardrail **per-note** কাজ করবে, যাতে একটা খারাপ note বাকিগুলো নষ্ট না করে।

**Failure taxonomy → আচরণ** (`llm/types.py` + `chain.py`):

| Exception | আচরণ |
|---|---|
| `AuthFailure` (401/403) | ওই key কখনো retry নয় → পরের candidate |
| `RateLimited` (429) / `Timeout` / `ServerError` (5xx) | পরের candidate, একই key-তে retry নয় |
| `MalformedJSON` | একই provider-এ **এক বার** repair retry (pydantic error text ফেরত দিয়ে), তারপর failover |
| সব candidate শেষ | all-`no_op` interpretation + unconstrained solve (500 নয়) |

**Circuit breaker:** প্রতি provider-এর `cooldown_until` in-memory dict। 429/timeout খেলে ৩০ সেকেন্ড cooldown — পরের request সরাসরি next provider-এ যাবে, বারবার একই দেয়ালে মাথা ঠুকবে না।

**Time budget:** `LLM_TOTAL_BUDGET_SECONDS = 20` hard wall-clock (judge ceiling 30s, বাফার রেখে)। Per-provider timeout: Gemini 6s, Groq 6s, Ollama 8s।

**সবচেয়ে গুরুত্বপূর্ণ — genuine no_op vs failure আলাদা করা:**

```python
class InterpretationOutcome(BaseModel):
    directives: list[DirectiveInterpretation]
    status: Literal["interpreted", "degraded", "failed"]   # internal only
    provider: str | None
```

`status` **response body-তে যাবে না** (schema fixed), কিন্তু structured log + `/health`-এর diagnostics counter-এ যাবে। এটাই আগের bug #১-এর সরাসরি ফিক্স: আগে "model বলেছে note অপ্রাসঙ্গিক" আর "LLM চলেইনি" — দুটোই একই all-`no_op` হয়ে যেত, কোনো সংকেত ছাড়া।

**Prompt (`problem/directives/prompt.py`) — যে তিনটা জিনিস অবশ্যই থাকবে** (দুই accepted সলিউশন থেকে শেখা):

1. **Half-open window, কাজ করা উদাহরণসহ:** `"1 PM to 3 PM" → hours [13,14]`; wraparound: `"11 PM to 2 AM" → [23,0,1]`।
2. **`factor` = অবশিষ্ট ভগ্নাংশ, preposition table দিয়ে:** `"TO x" / "OF normal" / "AT x" → factor = x`; `"BY x" / "DOWN BY x" / "LOSS OF x" → factor = 1-x`। ("cut by three-quarters" → 0.25)
3. **"MAINTENANCE IS NOT A DISTRACTOR"** — charger/battery/PV/grid offline notes **হলো** directive; শুধু প্রশাসনিক note (`"cafeteria menu changes"`) `no_op`। Naim এটা আলাদা নিয়ম হিসেবে লিখেছিল।

**Few-shot overfitting এড়ানো:** আগের `app/fewshot.py`-তে ১৫টা উদাহরণ ছিল যা ১০টা public case-এর প্রায় হুবহু নকল — hidden paraphrase-এ দুর্বল। নতুন few-shot **public wording থেকে ইচ্ছাকৃতভাবে ভিন্ন** ভাষায় লেখা হবে, ৬টা উদাহরণে ছয়টা directive type কভার করে।

---

## 6. Docker

```dockerfile
# multi-stage; ollama pull নিজের layer-এ যাতে cache হয়
FROM python:3.11-slim AS builder
  → pip install scipy numpy fastapi uvicorn pydantic httpx
FROM ... AS runtime
  → ollama binary + `ollama pull qwen2.5:3b-instruct`   ← BUILD TIME, startup নয়
  → non-root user, EXPOSE 8000, uvicorn
```

**কেন build-time pull:** `/health` অবশ্যই ৬০ সেকেন্ডের মধ্যে ready হতে হবে। Startup-এ pull করলে (এটা আমরা আগে একবার ভুলভাবে ভেবেছিলাম) প্রথম boot-এ কয়েক শ MB ডাউনলোড হবে আর readiness মিস হবে। Image বড় হবে (~2-3GB) — সেটা গ্রহণযোগ্য ট্রেডঅফ।

`/health` **কখনো LLM probe করবে না** — bare `{"status":"ok"}` ফেরত দেবে (Naim ইচ্ছাকৃতভাবে এটা করেছে, readiness flakiness এড়াতে)। Ollama warm-up ব্যাকগ্রাউন্ডে হবে।

---

## 7. Implementation phases

প্রতিটা phase শেষে টেস্ট সবুজ না হলে পরেরটায় যাওয়া হবে না।

| # | Phase | Deliverable | Gate |
|---|---|---|---|
| 1 | **Fixture + skeleton** | `tests/fixtures/public_sample_cases.json` commit, package tree, pydantic schemas | `pytest --collect-only` কোনো error ছাড়া চলে |
| 2 | **Solver (offline, no LLM)** | `solver/model.py`, `solve.py`, `replay.py` | **১০টা public case-ই ground-truth directive দিয়ে reference cost মেলে (≤0.01)** এবং independent replay পাস |
| 3 | **Guardrails** | `directives/schema.py`, `validate.py`, `overlay.py` | প্রতিটা invalid field inject করলে শুধু সেই note downgrade হয় |
| 4 | **LLM core (reusable)** | `llm/` সম্পূর্ণ, scripted fake provider সহ | সব failure mode simulate করে পাস; `llm/` → `problem/` import নেই |
| 5 | **API + wiring** | `api/app.py`, routes, summary | e2e fake-provider দিয়ে পুরো pipeline |
| 6 | **Docker + deploy** | Dockerfile, compose, deploy | **deploy smoke test** (নিচে) |
| 7 | **Real-LLM eval** | `scripts/eval_interpretation.py` | ১০ public case + ১২টা হাতে-লেখা paraphrase-এ accuracy রিপোর্ট |

---

## 8. Testing strategy

| ফাইল | কী নিশ্চিত করে |
|---|---|
| `test_solver_public_cases.py` | ১০টা official case, ground-truth directive → reference cost exact match + replay পাস। **Phase 2-এর gate** |
| `test_solver_edges.py` | overlapping directives, EOD repair পথ, infeasible → slack relaxation, ১০০+ random scenario দ্বিতীয় স্বাধীন LP-র বিপরীতে cross-check |
| `test_degeneracy.py` | কোনো ঘণ্টায় charge>0 ও discharge>0 একসাথে আসে না; idle হলে `battery_kwh == 0.0` ঠিক |
| `test_reported_totals.py` | `total_grid_kwh`/`total_cost_bdt`/`peak_grid_kwh` সবসময় `hourly_plan` থেকে recalculated মানের সমান |
| `test_guardrails.py` | hours out-of-range / non-ascending / duplicate index / factor>1 / reserve>capacity — প্রতিটা inject করলে শুধু সেই note `no_op`-এ নামে, বাকি অক্ষত |
| `test_llm_failover.py` | scripted provider দিয়ে: timeout, 429, 401, malformed JSON, empty, সব-down — প্রতিটাতে সঠিক routing আর কখনো crash নয় |
| `test_hermeticity.py` | API key সরিয়ে **পুরো suite** চলে; `llm/` কোনো `problem/` import করে না (reusability enforce) |
| `test_api_contract.py` | exact response schema, `scenario_id` echo, 400 malformed JSON, 24 unique hours |
| **`test_deploy_smoke.py`** | **আগেরবার যেটা ছিল না।** env var থেকে base URL নিয়ে সত্যিকার HTTP call — `/health` ok, আর একটা directive-ওয়ালা scenario পাঠিয়ে assert করে যে **ফেরত আসা interpretation সত্যিই non-`no_op`**, অর্থাৎ production-এ LLM বাস্তবে চলছে |
| `scripts/load_check.py` | ২০টা দ্রুত request — no_op হার, p95 latency, কোন provider উত্তর দিল |

---

## 9. Backup plan — degradation ladder

কখনো crash নয়, কখনো নীরব মিথ্যা নয়। প্রতি ধাপে যতটুকু সম্ভব বাঁচানো হয়:

```
1. Gemini ✓                                    → status="interpreted"
2. Gemini ✗ → Groq ✓                           → status="interpreted"
3. Groq ✗ → Ollama (local, rate-limit নেই) ✓    → status="interpreted"
4. কিছু note ঠিক, কিছু guardrail fail           → খারাপগুলো per-note no_op,
                                                  ভালোগুলো প্রয়োগ, status="degraded"
5. সব provider ✗                               → all no_op + unconstrained solve,
                                                  status="failed" (HTTP 200, log-এ alarm)
6. LP infeasible                               → soft directive slack-relax, feasible=False
7. Slack LP-ও ✗ (অসম্ভব হওয়ার কথা)              → grid-only trivial valid plan
```

**Deployment backup:** প্রাথমিক endpoint Docker (Render/Fly/Railway)। সাথে একটা **stateless mirror** Vercel-এ রাখা যায় — একই কোড, শুধু Ollama provider বাদ (serverless-এ persistent model process চলে না)। `.env`-এ `LLM_PROVIDER_ORDER` দিয়ে নিয়ন্ত্রিত, কোডে কোনো platform-specific `if` নয় — **এটাই সেই kill-switch anti-pattern যা আগেরবার ডুবিয়েছিল**।

---

## 10. এই প্ল্যান আগের সমস্যাগুলো কীভাবে mitigate করে

| # | আগের সমস্যা | এই প্ল্যানে সমাধান |
|---|---|---|
| 1 | **`if VERCEL: raise` kill-switch — deploy-এ LLM কখনো চলেনি** | Docker primary deploy; কোডে কোনো platform-conditional শাখা নিষিদ্ধ; provider order শুধুই env var; আর `test_deploy_smoke.py` **deployed URL-এ** assert করে যে interpretation সত্যিই non-`no_op` — এই bug আর কখনো নীরবে বাঁচতে পারবে না |
| 2 | LLM fail = genuine no_op, কোনো পার্থক্য নেই | `InterpretationOutcome.status` (`interpreted`/`degraded`/`failed`) — internal, log + metrics-এ; genuine `no_op` শুধু তখনই যখন model নিজে বলেছে |
| 3 | কেন fail করছে জানা নেই | Typed exception taxonomy (`AuthFailure`/`RateLimited`/`Timeout`/`MalformedJSON`) + `ProviderTrace` log: কোন provider, কোন status, কত সময়, কোন repair হয়েছে |
| 4 | প্রতি note-এ আলাদা call → quota শেষ | সব note এক call-এ ব্যাচ; guardrail তবু per-note |
| 5 | LLM fail হলে **পুরো** batch all-no_op | Per-note `_safe_no_op()` downgrade + `reconcile()` — একটা খারাপ field বাকি note নষ্ট করে না ("granular degradation", ঠিক যেটা accepted সলিউশন দুটো করেছিল) |
| 6 | একটাই কার্যকর provider | Gemini → Groq → **baked-in Ollama**; শেষটা rate-limit-মুক্ত, তাই chain সবসময় উত্তর নিয়েই শেষ হয় |
| 7 | Sequential fallback-এ সময় যোগ হয় | Hard 20s wall-clock budget + per-provider timeout + **circuit breaker** (৩০s cooldown) — ব্যর্থ provider-এ বারবার সময় নষ্ট হয় না |
| 8 | Local model থাকলে startup ধীর, `/health` মিস | Model **build-time**-এ baked (startup pull নয়); `/health` LLM probe করে না, bare `{"status":"ok"}` |
| 9 | `test_public_cases.py` কখনো চলেনি (missing fixture) | Fixture repo-তে commit, path hardcode নয়; **Phase 2-এর gate** = ১০টা case-ই reference cost মেলা, নাহলে এগোনো যাবে না |
| 10 | Optimizer-এর correctness কখনো verify হয়নি | Public-case exact-match + random scenario cross-check + **স্বাধীন replay validator** (spec থেকে লেখা, solver থেকে নয়) response পাঠানোর আগে |
| 11 | একই ঘণ্টায় charge ও discharge — schema ভাঙার ঝুঁকি | `EPS_LOOP` tie-break + প্রমাণিত-নিরাপদ netting post-process + strict idle threshold; `test_degeneracy.py` পাহারা দেয় |
| 12 | Reported totals আর plan আলাদা হয়ে যেতে পারে | সব কিছু plan থেকে derive: `grid` balance equation থেকে, `E` forward replay থেকে, totals plan list থেকে — judge-এর recalculation স্বতঃসিদ্ধভাবে মেলে |
| 13 | LP infeasible → 500 | Soft-directive slack relaxation + `feasible=False`; physics কখনো শিথিল নয় |
| 14 | Few-shot ১০টা public case-এর নকল → overfit | Public wording থেকে ইচ্ছাকৃত ভিন্ন few-shot; `scripts/eval_interpretation.py` paraphrase-এ accuracy মাপে |
| 15 | ভবিষ্যতে re-use করা কঠিন | `llm/` + `core/` সম্পূর্ণ problem-agnostic, টেস্ট দিয়ে enforce করা; নতুন প্রজেক্টে শুধু `prompt.py` + `schema.py` + solver পাল্টাবে |
| 16 | Async handler-এ blocking sync SDK call | সব provider adapter `httpx.AsyncClient` দিয়ে সত্যিকার async |
| 17 | Secret leak ঝুঁকি | Error message-এ `_redact()`; Docker image-এ কোনো baked key নেই; `test_hermeticity.py` নিশ্চিত করে টেস্টে key লাগে না |

---

## 11. Verification — end-to-end

```bash
# Phase 2 gate — কোনো API key ছাড়াই
pytest tests/test_solver_public_cases.py -v
# প্রত্যাশা: ১০/১০ পাস, প্রতিটা reference cost-এর 0.01 এর মধ্যে

# পুরো suite, network ছাড়া
env -u GEMINI_API_KEY -u GROQ_API_KEY pytest -v

# Container build + local run
docker build -t gridwise:v2 .
docker run -p 8000:8000 --env-file .env gridwise:v2
curl -s localhost:8000/health                       # {"status":"ok"} ≤60s এর মধ্যে
curl -s -X POST localhost:8000/optimize-energy \
  -H 'content-type: application/json' \
  -d @tests/fixtures/sample01_input.json | jq .directive_interpretation
# প্রত্যাশা: note 0 → solar_reduction hours [12,13] factor 0.25; note 1 → no_op

# Deployed endpoint — সেই bug-এর বিরুদ্ধে আসল পাহারা
GRIDWISE_BASE_URL=https://<deployed> pytest tests/test_deploy_smoke.py -v

# Load/latency
python scripts/load_check.py --n 20 --url https://<deployed>
# প্রত্যাশা: relevant note-এ শূন্য no_op, p95 < 15s

# Real-LLM interpretation accuracy
python scripts/eval_interpretation.py
```

**Definition of done:** ১০টা public case exact-cost match · deploy smoke test সবুজ (production-এ LLM বাস্তবে চলছে প্রমাণিত) · ২০-request load-এ relevant note-এ শূন্য `no_op` · API key ছাড়া পুরো suite পাস · `docker run` থেকে `/health` ৬০s-এর মধ্যে ready।
