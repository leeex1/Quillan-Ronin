#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""READ-ONLY verification of the four-donor expert bank (board #1247/#1248).

Opens exports/donor_experts_aligned.safetensors via ``safetensors.safe_open``
and, for each of the 34 experts, computes over moe.w1 / moe.wgate / moe.w2:

  * fraction-nonzero  (count_nonzero / numel)  -- the PASS gate: > 1%
  * abs-mean          (mean of |w|)
  * gamma magnitude   (mean of per-output-channel absmean |w|, matching the
                       build's closed-form gamma recomputation)

Key schema confirmed from exports/donor_experts_aligned.report.json:
  moe.w1     [34, 896, 4864]  F16
  moe.wgate  [34, 896, 4864]  F16
  moe.w2     [34, 4864, 896]  F16

Donor slot map: 0-7 qwen, 8-15 llama, 16-23 falcon, 24-31 bitnet,
32-33 consensus. Experts 16-23 were ZERO in the defective build (#1234/#1244).

STRICTLY READ-ONLY: safetensors is opened in read mode only; no file is ever
written. This script never touches canonical weights, gates, or checkpoints.

Exit codes: 0 = all 34 experts pass, 1 = one or more experts effectively
all-zero, 2 = structural error (missing file / key / bad shape).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from safetensors import safe_open

ROOT: Path = Path(r"C:\02_QUILLAN")
DEFAULT_BANK: Path = ROOT / "exports" / "donor_experts_aligned.safetensors"
DEFAULT_REPORT: Path = ROOT / "exports" / "donor_experts_aligned.report.json"

EXPECTED_KEYS: Tuple[str, str, str] = ("moe.w1", "moe.wgate", "moe.w2")
N_EXPERTS: int = 34
DEFAULT_MIN_NONZERO: float = 0.01  # strict ">" per task spec


def donor_for(index: int) -> str:
    """Return the donor label for a 0-based expert index."""
    if 0 <= index <= 7:
        return "qwen"
    if 8 <= index <= 15:
        return "llama"
    if 16 <= index <= 23:
        return "falcon"
    if 24 <= index <= 31:
        return "bitnet"
    if 32 <= index <= 33:
        return "consensus"
    return "UNKNOWN"


def check_report_schema(report: Path) -> None:
    """Warn (never fail) if the report JSON advertises a different key schema."""
    if not report.is_file():
        print(f"[WARN] report not found: {report} -- using confirmed defaults")
        return
    try:
        with report.open("r", encoding="utf-8") as handle:
            data: Dict[str, object] = json.load(handle)
        shapes = data.get("verification", {}).get("shapes", {})  # type: ignore[union-attr]
        if shapes and set(shapes) != set(EXPECTED_KEYS):
            print(
                f"[WARN] report keys {sorted(shapes)} != expected {list(EXPECTED_KEYS)}"
            )
    except (OSError, ValueError, AttributeError) as exc:
        print(f"[WARN] could not parse report {report}: {exc}")


def tensor_stats(slice_obj: object, expert: int) -> Tuple[float, float, float]:
    """Return (fraction_nonzero, abs_mean, gamma_mean) for one expert slice."""
    arr = np.asarray(slice_obj[expert, :, :], dtype=np.float32)
    abs_vals = np.abs(arr)
    frac_nonzero = float(np.count_nonzero(arr) / arr.size) if arr.size else 0.0
    abs_mean = float(abs_vals.mean()) if arr.size else 0.0
    # Per-output-channel absmean (mean over the input axis), then averaged:
    # same closed-form gamma definition used by build_donor_experts.py.
    gamma_mean = float(abs_vals.mean(axis=0).mean()) if arr.size else 0.0
    return frac_nonzero, abs_mean, gamma_mean


def verify(bank: Path, report: Path, min_nonzero: float) -> int:
    """Run the verification. Return a process exit code."""
    if not bank.is_file():
        print(f"[ERROR] bank file not found: {bank}")
        return 2
    check_report_schema(report)

    rows: List[Tuple[int, str, float, float, float, float, float, str]] = []
    try:
        with safe_open(str(bank), framework="np") as handle:
            keys = set(handle.keys())
            missing = [key for key in EXPECTED_KEYS if key not in keys]
            if missing:
                print(
                    f"[ERROR] missing tensor key(s): {missing}; found: {sorted(keys)}"
                )
                return 2
            slices = {key: handle.get_slice(key) for key in EXPECTED_KEYS}
            for key in EXPECTED_KEYS:
                shape = slices[key].get_shape()
                if len(shape) != 3 or shape[0] != N_EXPERTS:
                    print(
                        f"[ERROR] key {key} has shape {shape}; expected leading dim {N_EXPERTS}"
                    )
                    return 2

            for expert in range(N_EXPERTS):
                per_key = {
                    key: tensor_stats(slices[key], expert) for key in EXPECTED_KEYS
                }
                fracs: Sequence[float] = [per_key[k][0] for k in EXPECTED_KEYS]
                abs_means: Sequence[float] = [per_key[k][1] for k in EXPECTED_KEYS]
                gammas: Sequence[float] = [per_key[k][2] for k in EXPECTED_KEYS]
                min_frac = min(fracs)
                verdict = "PASS" if min_frac > min_nonzero else "FAIL"
                rows.append(
                    (
                        expert,
                        donor_for(expert),
                        fracs[0],
                        fracs[1],
                        fracs[2],
                        min_frac,
                        sum(abs_means) / len(abs_means),
                        sum(gammas) / len(gammas),
                        verdict,
                    )
                )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] could not read {bank}: {exc}")
        return 2

    header = (
        f"{'exp':>3}  {'donor':<9} {'nz(w1)':>8} {'nz(wgate)':>9} {'nz(w2)':>8} "
        f"{'min_nz':>8} {'absmean':>10} {'gamma':>10}  result"
    )
    print(f"[READ-ONLY] bank: {bank}")
    print(f"[RULE] PASS iff min nonzero fraction > {min_nonzero:.2%} (>1% default)\n")
    print(header)
    print("-" * len(header))
    for exp, donor, f1, fg, f2, min_frac, abs_mean, gamma, verdict in rows:
        print(
            f"{exp:>3}  {donor:<9} {f1:>8.4f} {fg:>9.4f} {f2:>8.4f} "
            f"{min_frac:>8.4f} {abs_mean:>10.6f} {gamma:>10.6f}  {verdict}"
        )

    failures = [row[0] for row in rows if row[-1] == "FAIL"]
    print("-" * len(header))
    print(
        f"[SUMMARY] PASS {len(rows) - len(failures)}/{len(rows)}  FAIL {len(failures)}/{len(rows)}"
    )
    if failures:
        print(f"[FAIL] effectively all-zero experts: {failures}")
        return 1
    print("[OK] all 34 experts carry non-trivial signal (falcon 16-23 populated).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bank",
        type=Path,
        default=DEFAULT_BANK,
        help="expert-bank .safetensors (read-only)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="build report JSON (schema cross-check)",
    )
    parser.add_argument(
        "--min-nonzero",
        type=float,
        default=DEFAULT_MIN_NONZERO,
        help="PASS threshold on min fraction-nonzero (default 0.01, strict >)",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_nonzero <= 1.0:
        print(f"[ERROR] --min-nonzero must be in [0, 1], got {args.min_nonzero}")
        return 2
    return verify(args.bank, args.report, args.min_nonzero)


if __name__ == "__main__":
    sys.exit(main())
