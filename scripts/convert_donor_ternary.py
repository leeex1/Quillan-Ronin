#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0/1: convert donor_cache_v23 float donors to (q, gamma) ternary cache.

Uses the EXACT shipped per-channel absmean primitive from the canonical adapter
(quillan_weight_adapter._per_channel_absmean_ternarize), which is byte-identical
to build_coherent_ternary_base.py:49-56 ``quantize_to_ternary``.

SAFETY INVARIANTS
  * sources are opened torch.load(..., weights_only=True, map_location="cpu");
  * output path MUST NOT exist (never overwrite a donor or a prior conversion);
  * source sha256 is checked before AND after the write (no destruction);
  * one donor at a time, fp32 working buffers released per tensor (<=16GB RAM).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CORE_MODEL_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model")
sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, str(CORE_MODEL_DIR))

from quillan_weight_adapter import _per_channel_absmean_ternarize  # noqa: E402

DONOR_DIR = Path(r"C:\02_QUILLAN\exports\donor_cache_v23")
DEFAULT_DONORS = ("llama_1b_donors.pt", "qwen_0.5b_donors.pt", "dragon_1b_donors.pt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_one(src: Path, dst: Path, eps: float) -> Dict[str, Any]:
    """Convert one float donor -> (q:int8, gamma:float32) cache; validate loudly."""
    if not src.is_file():
        raise FileNotFoundError(f"source donor missing: {src}")
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {dst}")

    t0 = time.time()
    sha_before = sha256_file(src)
    obj = torch.load(str(src), map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise TypeError(f"donor payload is {type(obj)!r}, expected dict")

    out: Dict[str, torch.Tensor] = {}
    per_tensor: Dict[str, Dict[str, float]] = {}
    failures: list[str] = []  # structural invariants -> hard fail
    warnings: list[str] = []  # statistical bands -> recorded, never gate here
    total_params = 0

    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            failures.append(f"{key}: non-tensor payload {type(value)!r}")
            continue
        w = value.detach().float()
        if w.ndim < 2:
            failures.append(f"{key}: expected >=2D, got shape {tuple(w.shape)}")
            del w
            continue
        # EXACT shipped absmean primitive (per-channel, dim=-1).
        q_float, gamma = _per_channel_absmean_ternarize(w, eps=eps)
        q = q_float.to(torch.int8).contiguous()
        g = gamma.detach().float().contiguous()

        # ---- validation gates ------------------------------------------------
        uniq = set(torch.unique(q).tolist())
        if not uniq.issubset({-1, 0, 1}):
            failures.append(f"{key}: q domain {sorted(uniq)} not subset of {{-1,0,1}}")
        if not torch.isfinite(g).all():
            failures.append(f"{key}: non-finite gamma")
        if not (g > 0).all():
            failures.append(f"{key}: non-positive gamma (min={g.min().item():.3e})")
        n_rows = q.shape[0]
        rows_with_signal = int((q != 0).any(dim=-1).sum().item())
        dead_rows = n_rows - rows_with_signal
        src_dead_rows = int((w == 0).all(dim=-1).sum().item())
        new_dead_rows = dead_rows - src_dead_rows
        # Inherited all-zero rows are a source property (e.g. unused dragon token
        # ids), not a conversion defect. Only NEWLY introduced dead groups fail.
        if new_dead_rows > 0:
            failures.append(
                f"{key}: {new_dead_rows} NEWLY introduced all-zero rows "
                f"(total {dead_rows}, source had {src_dead_rows})"
            )
        elif dead_rows > 0:
            warnings.append(
                f"{key}: {dead_rows}/{n_rows} all-zero rows inherited verbatim "
                f"from source (source all-zero rows={src_dead_rows})"
            )

        rec = q.float() * g  # broadcast (rows,1) across columns
        denom = torch.linalg.vector_norm(w)
        rel = float(torch.linalg.vector_norm(w - rec) / denom) if denom > 0 else 0.0
        mse = float(((w - rec) ** 2).mean())
        zero_frac = float((q == 0).float().mean())
        g_med = float(g.median())
        g_min = float(g.min())
        g_max = float(g.max())
        per_tensor[key] = {
            "numel": int(w.numel()),
            "shape": list(w.shape),
            "rel_err": rel,
            "mse": mse,
            "gamma_median": g_med,
            "gamma_min": g_min,
            "gamma_max": g_max,
            "zero_fraction": zero_frac,
            "dead_rows": dead_rows,
            "source_dead_rows": src_dead_rows,
        }
        # Statistical bands from c21 #929 were measured on QWEN only; llama/dragon
        # legitimately differ. Record as WARNINGS (the authoritative conversion
        # decision is conversion_gate.py #926, untouched here) -- never hard-fail
        # a conversion on a diagnostic band.
        if not (0.005 <= g_med <= 0.05):
            warnings.append(f"{key}: gamma median {g_med:.6f} outside [0.005,0.05]")
        if g_max > 10.0 * g_med:
            warnings.append(f"{key}: gamma max {g_max:.6f} > 10x median {g_med:.6f}")
        if not (0.20 <= zero_frac <= 0.45):
            warnings.append(f"{key}: zero_fraction {zero_frac:.4f} outside [0.20,0.45]")
        if rel > 0.55:
            warnings.append(f"{key}: rel_err {rel:.4f} > qwen-derived 0.55 band")

        out[f"{key}.q"] = q
        out[f"{key}.gamma"] = g
        total_params += int(w.numel())
        del w, q_float, gamma, q, g, rec
        gc.collect()

    if failures:
        raise RuntimeError(f"{src.name} validation FAILED:\n  " + "\n  ".join(failures))

    # key parity: 2 outputs per source tensor
    expected_keys = len(obj) * 2
    if len(out) != expected_keys:
        raise RuntimeError(f"key parity: got {len(out)}, expected {expected_keys}")

    meta: Dict[str, Any] = {
        "method": "per_channel_absmean_reuse_build_coherent_ternary_base_49_56",
        "group": "per_channel(dim=-1)",
        "eps": eps,
        "source": src.name,
        "source_sha256": sha_before,
        "converted_utc": datetime.now(timezone.utc).isoformat(),
        "total_params": total_params,
        "num_source_tensors": len(obj),
        "num_output_tensors": len(out),
    }
    out["__meta__"] = meta  # type: ignore[assignment]  (metadata, not a tensor)

    torch.save(out, str(dst))
    sha_after = sha256_file(src)
    if sha_after != sha_before:
        raise RuntimeError(f"SOURCE MUTATED: {src.name} {sha_before} -> {sha_after}")
    if not dst.is_file() or dst.stat().st_size == 0:
        raise RuntimeError(f"output cache not written: {dst}")

    rels = [v["rel_err"] for v in per_tensor.values()]
    report = {
        "source": str(src),
        "output": str(dst),
        "output_bytes": dst.stat().st_size,
        "source_sha256_before": sha_before,
        "source_sha256_after": sha_after,
        "source_unmodified": sha_before == sha_after,
        "total_params": total_params,
        "source_tensors": len(obj),
        "output_tensors": len(out) - 1,
        "mean_rel_err": sum(rels) / len(rels),
        "max_rel_err": max(rels),
        "warnings": warnings,
        "elapsed_sec": time.time() - t0,
        "per_tensor": per_tensor,
        "meta": meta,
    }
    del obj, out
    gc.collect()
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=DONOR_DIR)
    ap.add_argument("--out-dir", type=Path, default=DONOR_DIR)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--names", nargs="*", default=list(DEFAULT_DONORS))
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_reports: Dict[str, Any] = {}
    for name in args.names:
        src = args.src_dir / name
        dst = args.out_dir / (name.replace(".pt", "_ternary.pt"))
        print(f"[convert] {name} -> {dst.name}", flush=True)
        rep = convert_one(src, dst, args.eps)
        all_reports[name] = rep
        print(
            f"  params={rep['total_params']:,} tensors={rep['output_tensors']} "
            f"mean_rel={rep['mean_rel_err']:.4f} max_rel={rep['max_rel_err']:.4f} "
            f"bytes={rep['output_bytes']:,} src_sha_unchanged={rep['source_unmodified']} "
            f"warnings={len(rep['warnings'])} elapsed={rep['elapsed_sec']:.2f}s",
            flush=True,
        )
        for wmsg in rep["warnings"]:
            print(f"    WARN {wmsg}", flush=True)

    summary = {
        "method": "per_channel_absmean_reuse_build_coherent_ternary_base_49_56",
        "eps": args.eps,
        "donors": all_reports,
        "all_sources_unmodified": all(
            r["source_unmodified"] for r in all_reports.values()
        ),
    }
    print(
        "CONVERT_DONOR_TERNARY_RESULT: "
        + ("PASS" if summary["all_sources_unmodified"] else "FAIL"),
        flush=True,
    )
    if args.report is not None:
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"WROTE {args.report}", flush=True)


if __name__ == "__main__":
    main()
