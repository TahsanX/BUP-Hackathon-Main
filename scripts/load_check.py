"""Hammer a running deployment and report what the judge would experience.

    python scripts/load_check.py --url https://your-deployment --n 20

Reports p95 latency and, more importantly, how often a note that should have
produced a directive came back as no_op — the silent failure mode that costs
both interpretation and application marks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "public_sample_cases.json"


def load_cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())["cases"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    cases = load_cases()

    latencies: list[float] = []
    failures = 0
    wrong_no_op = 0
    expected_directives = 0

    with httpx.Client(timeout=args.timeout) as client:
        for index in range(args.n):
            case = cases[index % len(cases)]
            expected = case["expected_output"]["directive_interpretation"]

            started = time.monotonic()
            try:
                response = client.post(f"{base}/optimize-energy", json=case["input"])
                response.raise_for_status()
                body = response.json()
            except Exception as exc:  # noqa: BLE001 - this is a reporting tool
                failures += 1
                print(f"[{index:>3}] {case['id']:<10} FAILED  {type(exc).__name__}: {exc}")
                continue
            elapsed = time.monotonic() - started
            latencies.append(elapsed)

            got = {e["note_index"]: e["directive_type"] for e in body["directive_interpretation"]}
            missed = []
            for entry in expected:
                if entry["directive_type"] == "no_op":
                    continue
                expected_directives += 1
                if got.get(entry["note_index"]) == "no_op":
                    wrong_no_op += 1
                    missed.append(entry["note_index"])

            flag = f"  MISSED notes {missed}" if missed else ""
            print(f"[{index:>3}] {case['id']:<10} {elapsed:6.2f}s  {body['total_cost_bdt']:>10.2f} BDT{flag}")

    print("\n--- summary ---")
    print(f"requests           : {args.n}")
    print(f"transport failures : {failures}")
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        print(f"latency mean/p95   : {statistics.mean(latencies):.2f}s / {p95:.2f}s")
    print(
        f"relevant notes     : {expected_directives}, "
        f"wrongly no_op: {wrong_no_op} "
        f"({0 if not expected_directives else 100 * wrong_no_op / expected_directives:.1f}%)"
    )

    try:
        diagnostics = httpx.get(f"{base}/diagnostics", timeout=10).json()
        print(f"providers          : {diagnostics.get('providers')}")
        print(f"interpretation     : {diagnostics.get('interpretation_status')}")
    except Exception:  # noqa: BLE001
        pass

    # Target: zero relevant notes lost.
    return 1 if (failures or wrong_no_op) else 0


if __name__ == "__main__":
    raise SystemExit(main())
