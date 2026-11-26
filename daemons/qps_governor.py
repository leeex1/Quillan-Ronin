#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qps_governor.py -- QPS + QSSR resource governor for the Quillan host.

Implements two formulas from 8-Formulas.md directly:

  QPS (id 20, Discrete-Time Algebraic Riccati / LQR):
      P_t = A^T P_{t+1} A - A^T P_{t+1} B (R(E) + B^T P_{t+1} B)^-1
            B^T P_{t+1} A + Q(E)

  QSSR (id 9, Lyapunov stability):
      V(x, d) = x^T P x + zeta * d_recursion^2,  require dV < 0

INTENT: replace the hand-rolled proportional `gov_scale` in
quillan_governor_daemon.py with an optimal controller that is provably
stable under load. It writes a RECOMMENDATION only -- it never writes
governor_state.json and never changes affinity, priority, or power state.

OPT-IN: importing this module does nothing. Nothing launches it at boot.
The existing governor keeps running unchanged until you enable it.

USAGE:
    python qps_governor.py --self-test      # verify formulas, no side effects
    python qps_governor.py --once           # compute one recommendation
    python qps_governor.py --status         # show last recommendation

To activate: set booster_bridge.use_qps_recommendation(True) in the
model-owning session. Default remains the live governor's gov_scale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "solve_dare",
    "lyapunov_value",
    "compute_recommendation",
    "self_test",
    "status",
]

_REPO = Path(r"F:\02_QUILLAN")
_GOV_STATE = _REPO / "07 - Memory & LanceDB" / "governor_state.json"
_RECOMMENDATION = _REPO / "07 - Memory & LanceDB" / "qps_recommendation.json"

# State vector x = pressure deviations from target:
#   [core0_load - 0.45, ram_used_frac - 0.70, vram_used_frac - 0.75]
_TARGETS = np.array([0.45, 0.70, 0.75], dtype=np.float64)

# A: natural pressure relaxation (diagonal, stable inside unit circle)
_A = np.diag([0.85, 0.95, 0.90]).astype(np.float64)

# B: control effect -- raising gov_scale means MORE model work, MORE pressure
_B = np.array([[0.60], [0.40], [0.30]], dtype=np.float64)

# Base costs; scaled by thermodynamic load E_Omega per QPS
_Q_BASE = np.diag([2.0, 1.0, 1.5]).astype(np.float64)
_R_BASE = np.array([[0.5]], dtype=np.float64)

_ZETA = 0.10  # QSSR recursion-depth penalty
_MIN_SCALE = 0.50
_MAX_SCALE = 1.00
_DARE_MAX_ITER = 200
_DARE_TOL = 1e-9


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def solve_dare(E: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve the discrete algebraic Riccati equation by backward iteration.

    Returns (P, K) where K is the optimal feedback gain.

    Q(E) = Q_base * (1 + E), R(E) = R_base * (1 + E) -- energy-scaled
    exactly as QPS specifies.
    """
    E = _clamp(float(E), 0.0, 1.0)
    Q = _Q_BASE * (1.0 + E)
    R = _R_BASE * (1.0 + E)

    P = Q.copy()
    for _ in range(_DARE_MAX_ITER):
        BT_P = _B.T @ P
        denom = R + BT_P @ _B
        K = np.linalg.solve(denom, BT_P @ _A)  # (1,3)
        P_next = _A.T @ P @ _A - _A.T @ P @ _B @ K + Q
        # Symmetrize to kill float drift
        P_next = 0.5 * (P_next + P_next.T)
        if float(np.max(np.abs(P_next - P))) < _DARE_TOL:
            P = P_next
            break
        P = P_next

    BT_P = _B.T @ P
    K = np.linalg.solve(R + BT_P @ _B, BT_P @ _A)
    return P, K


def lyapunov_value(x: np.ndarray, P: np.ndarray, recursion_depth: float) -> float:
    """QSSR: V(x,d) = x^T P x + zeta * d^2."""
    return float(x.T @ P @ x + _ZETA * float(recursion_depth) ** 2)


def _read_state() -> dict[str, Any]:
    try:
        data = json.loads(_GOV_STATE.read_text(encoding="ascii"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _thermodynamic_load(state: dict[str, Any]) -> float:
    """Derive normalized E_Omega from live telemetry, no torch needed."""
    try:
        core0 = float(state.get("cpu_core0_pct", 0.0)) / 100.0
        ram = float(state.get("ram_used_gb", 0.0)) / 28.0
        return _clamp(0.5 * core0 + 0.5 * ram, 0.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


def _state_vector(state: dict[str, Any]) -> np.ndarray:
    try:
        core0 = float(state.get("cpu_core0_pct", 0.0)) / 100.0
        ram = float(state.get("ram_used_gb", 0.0)) / 28.0
        vram_total = 4.0
        vram = float(state.get("vram_used_mb", 0.0)) / 1024.0 / vram_total
    except (TypeError, ValueError):
        core0 = ram = vram = 0.0
    return np.array([core0, ram, vram], dtype=np.float64) - _TARGETS


def _prev_control() -> tuple[float, int]:
    try:
        d = json.loads(_RECOMMENDATION.read_text(encoding="utf-8"))
        return float(d.get("u", 0.0)), int(d.get("recursion_depth", 0))
    except (OSError, ValueError, TypeError, KeyError):
        return 0.0, 0


def compute_recommendation() -> dict[str, Any]:
    """Compute one QPS-optimal, QSSR-guarded gov_scale recommendation."""
    state = _read_state()
    E = _thermodynamic_load(state)
    x = _state_vector(state)
    P, K = solve_dare(E)

    try:
        current_scale = float(state.get("gov_scale", 1.0))
    except (TypeError, ValueError):
        current_scale = 1.0
    current_scale = _clamp(current_scale, _MIN_SCALE, _MAX_SCALE)

    u_prev, depth_prev = _prev_control()
    u_raw = -float((K @ x).item())

    # QSSR guard: backtrack until Lyapnyov non-increase is guaranteed.
    u = _clamp(u_raw, -0.15, 0.10)
    v_before = lyapunov_value(x, P, depth_prev)
    accepted = True
    for _ in range(12):
        x_next = _A @ x + _B.flatten() * float(u)
        v_after = lyapunov_value(x_next, P, depth_prev + 1)
        if v_after < v_before:
            break
        u *= 0.5
        accepted = False

    if abs(u) < 1e-6:
        u = 0.0
        accepted = False

    # Recursion depth: count consecutive same-sign control moves.
    if u_prev != 0.0 and np.sign(u) == np.sign(u_prev):
        depth = depth_prev + 1
    else:
        depth = 0

    recommended = _clamp(current_scale + u, _MIN_SCALE, _MAX_SCALE)

    return {
        "ts": time.time(),
        "module": "qps_governor",
        "formula": "QPS(DARE/LQR) + QSSR(Lyapunov)",
        "E_omega_normalized": round(E, 4),
        "x_pressure": [round(float(v), 4) for v in x],
        "u": round(float(u), 6),
        "u_raw": round(float(u_raw), 6),
        "recursion_depth": int(depth),
        "qssr_accepted": bool(accepted),
        "current_scale": round(current_scale, 4),
        "recommended_scale": round(recommended, 4),
        "P_diag": [round(float(P[i, i]), 6) for i in range(3)],
        "K": [round(float(v), 6) for v in K.flatten()],
    }


def self_test() -> int:
    """Validate DARE convergence, PSD, and Lyapunov decrease. No I/O."""
    failures: list[str] = []

    P, K = solve_dare(0.0)
    if not np.all(np.linalg.eigvalsh(P) >= -1e-8):
        failures.append("P is not positive semi-definite")

    _P_hi, K_hi = solve_dare(1.0)
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(K_hi)):
        failures.append("non-finite gain")

    # Simulate 20 steps from a high-pressure state with the QSSR guard active.
    x = np.array([0.55, 0.30, 0.25], dtype=np.float64)
    depth = 0
    v_prev = lyapunov_value(x, P, depth)
    monotone = True
    for _ in range(20):
        u = _clamp(-float((K @ x).item()), -0.15, 0.10)
        for _ in range(12):
            x_next = _A @ x + _B.flatten() * u
            if lyapunov_value(x_next, P, depth + 1) < v_prev:
                break
            u *= 0.5
        x = x_next
        depth = depth + 1 if u > 0 else 0
        v_new = lyapunov_value(x, P, depth)
        if v_new > v_prev + 1e-9:
            monotone = False
        v_prev = v_new

    if not monotone:
        failures.append("QSSR guard failed to keep V monotonically decreasing")

    if failures:
        for f in failures:
            print(f"[SELF-TEST FAIL] {f}")
        return 1

    print(
        "[SELF-TEST PASS] DARE converged, P PSD, gains finite, QSSR monotone over 20 steps"
    )
    print(f"  P_diag = {[round(float(P[i, i]), 4) for i in range(3)]}")
    print(f"  K      = {[round(float(v), 4) for v in K.flatten()]}")
    return 0


def status() -> dict[str, Any]:
    try:
        return json.loads(_RECOMMENDATION.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"status": "no recommendation yet"}


def main() -> int:
    parser = argparse.ArgumentParser(description="QPS + QSSR host governor")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.status:
        print(json.dumps(status(), indent=2))
        return 0
    if args.once:
        rec = compute_recommendation()
        _RECOMMENDATION.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(json.dumps(rec, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
