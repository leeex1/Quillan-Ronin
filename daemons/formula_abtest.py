#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formula_abtest.py -- Telemetry-only falsification harness (nullion spec v2).

Tests whether a formula set adds SYNERGY or is merely "more knobs moving."

REJECTED from v1: reading qps_recommendation.json (was 47.6h stale).
Every measurement now comes from system_telemetry.json alone.

METHOD (c17-nullion swarm, 2026-11-22):
  Fixed state x_k = [cpu.total_pct/100, memory.used_pct/100,
                     commit_used/commit_total]       -- matches published 3-dim x
  P fixed a priori from qps_governor self-test P_diag
  d_hat_k = delta(context_switches)/delta(t)          -- recursion proxy
  V_hat_k = x_k^T P x_k + zeta * d_hat_k^2
  Delta V_hat_k per interval

  SYNERGY requires full set to LOWER P(Delta V_hat > 0) AND LOWER
  Var(Delta V_hat) at NON-WORSE mean(Delta V_hat). If only the mean
  moves, or violation rate/variance unchanged or worse -> "more knobs",
  FALSIFIED.

  >=200 paired samples per arm. 95% CI of delta[P(Delta V>0)]
  containing 0 -> FALSIFIED. Stale ts -> INCOMPLETE, never PASS.

No process killing. No affinity changes. No model code touched.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(r"F:\02_QUILLAN")
_TELEMETRY = _REPO / "07 - Memory & LanceDB" / "system_telemetry.json"
_SAMPLES = _REPO / "07 - Memory & LanceDB" / "abtest_samples.json"

# P fixed a priori from qps_governor self-test (verified PSD 2026-11-22):
#   P_diag = [4.6244, 6.9452, 7.0350] at E=0
_P = np.diag([4.6244, 6.9452, 7.0350])
_ZETA = 0.10


def _read(path: Path) -> dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def sample_once(label: str) -> dict[str, Any] | None:
    """One labelled point from telemetry only. None if stale/invalid."""
    t = _read(_TELEMETRY)
    if t.get("source") != "host_telemetry":
        return None
    ts = float(t.get("ts", 0.0))
    age = time.time() - ts
    if age < 0 or age > 75:
        return None

    cpu = t.get("cpu", {})
    mem = t.get("memory", {})
    commit = t.get("commit", {})
    cs = t.get("cpu_stats", {})

    try:
        x = np.array(
            [
                float(cpu.get("total_pct", 0.0)) / 100.0,
                float(mem.get("used_pct", 0.0)) / 100.0,
                float(commit.get("commit_used_gb", 0.0))
                / max(float(commit.get("commit_total_gb", 1.0)), 1e-6),
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError):
        return None

    v = float(x.T @ _P @ x)
    return {
        "ts": ts,
        "label": label,
        "x": [round(float(a), 6) for a in x],
        "V": round(v, 6),
        "ctx_switches": int(cs.get("context_switches", 0)),
        "core0": float((cpu.get("per_core_pct") or [0.0])[0]),
    }


def collect(label: str, duration_s: int, interval_s: int) -> int:
    store = _read(_SAMPLES)
    rows = list(store.get("samples", []))
    end = time.time() + duration_s
    n = 0
    while time.time() < end:
        s = sample_once(label)
        if s is not None:
            rows.append(s)
            n += 1
        time.sleep(interval_s)
    _SAMPLES.write_text(json.dumps({"samples": rows}, indent=2), encoding="utf-8")
    print(f"collected {n} valid samples for '{label}' (stale dropped)")
    return 0


def _deltas(rows: list[dict[str, Any]]) -> list[float]:
    """Delta V with d_hat (ctx-switch rate) folded into the potential."""
    out: list[float] = []
    for a, b in zip(rows, rows[1:]):
        dt = float(b["ts"]) - float(a["ts"])
        if dt <= 0:
            continue
        d_hat = (float(b.get("ctx_switches", 0)) - float(a.get("ctx_switches", 0))) / dt
        va = (
            float(a["V"])
            + _ZETA
            * (
                (float(b.get("ctx_switches", 0)) - float(a.get("ctx_switches", 0)))
                / max(dt, 1e-6)
                * 0.01
            )
            ** 2
        )
        vb = float(b["V"]) + _ZETA * (d_hat * 0.01) ** 2
        out.append(vb - va)
    return out


def _wilson_ci(p: float, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dv = _deltas(rows)
    if not dv:
        return {"n": len(rows), "pairs": 0}
    viol = sum(1 for d in dv if d > 0)
    p = viol / len(dv)
    lo, hi = _wilson_ci(p, len(dv))
    return {
        "n": len(rows),
        "pairs": len(dv),
        "P_dV_positive": round(p, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        "var_dV": round(statistics.pstdev(dv) ** 2, 6),
        "mean_dV": round(statistics.fmean(dv), 6),
        "core0_sigma": round(
            statistics.pstdev([float(r.get("core0", 0.0)) for r in rows]), 2
        ),
    }


def report() -> int:
    store = _read(_SAMPLES)
    rows = list(store.get("samples", []))
    baseline = [r for r in rows if r.get("label") == "baseline"]
    formula = [r for r in rows if r.get("label") == "formula"]

    b = _metrics(baseline)
    f = _metrics(formula)
    out: dict[str, Any] = {"baseline": b, "formula": f}

    if b.get("pairs", 0) < 200 or f.get("pairs", 0) < 200:
        out["VERDICT"] = "INCOMPLETE"
        out["reason"] = "need >=200 paired samples per arm"
        print(json.dumps(out, indent=2))
        return 0

    dp = f["P_dV_positive"] - b["P_dV_positive"]
    ci_lo, ci_hi = _wilson_ci(dp, min(b["pairs"], f["pairs"]))
    var_improved = f["var_dV"] <= b["var_dV"]
    mean_not_worse = f["mean_dV"] <= b["mean_dV"]
    ci_excludes_zero = ci_lo > 0 or ci_hi < 0

    out["delta_P_dV_positive"] = round(dp, 4)
    out["delta_ci95"] = [round(ci_lo, 4), round(ci_hi, 4)]
    out["var_improved"] = var_improved
    out["mean_not_worse"] = mean_not_worse

    synergy = ci_excludes_zero and dp < 0 and var_improved and mean_not_worse
    out["VERDICT"] = "SYNERGY" if synergy else "FALSIFIED"
    out["criterion"] = (
        "SYNERGY requires: 95% CI of delta[P(dV>0)] excludes 0 and is "
        "negative, Var(dV) not worse, mean(dV) not worse. Otherwise "
        "'more knobs moving' -> FALSIFIED."
    )
    print(json.dumps(out, indent=2))
    return 0


def clear() -> int:
    if _SAMPLES.exists():
        _SAMPLES.unlink()
    print("samples cleared")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Telemetry-only synergy harness")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("collect")
    c.add_argument("--label", choices=["baseline", "formula"], required=True)
    c.add_argument("--duration", type=int, default=900)
    c.add_argument("--interval", type=int, default=5)

    sub.add_parser("report")
    sub.add_parser("clear")

    args = p.parse_args()
    if args.cmd == "collect":
        return collect(args.label, args.duration, args.interval)
    if args.cmd == "report":
        return report()
    if args.cmd == "clear":
        return clear()
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
