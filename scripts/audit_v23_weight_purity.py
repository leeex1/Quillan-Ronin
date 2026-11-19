#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
CPU-Only GGUF Weight-Purity & Stage Report Inspector for Quillan-Ronin v23.

1. Inspects GGUF KV metadata (context_length, rope.scaling.*, yarn_beta_fast/slow),
   tensor count, and dequantizes sample o_proj (attn_output) / down_proj (ffn_down)
   layers to verify:
     - finite=True (no NaN/Inf)
     - smooth 32-weight Q8_0 block max/RMS ratio (zero outlier blowup)
     - non-collapsed singular-value spectrum (zero singular-tail collapse)
2. Cross-checks optional v23_stageb/c/d_*_report.json files to assert:
     - param_diff == 0 (consistent total_params & finite=True)
     - max_cond <= 1e4
     - monotonic val_nll / -c 512 PPL progression

Usage:
  python -u audit_v23_weight_purity.py --gguf C:\02_QUILLAN\exports\clean-llama-1b-q8_0.gguf
  python -u audit_v23_weight_purity.py --gguf <gguf> --report <json1> [<json2> ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import gguf
import gguf.quants
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Audit_V23_Purity")


def _safe_kv(reader: gguf.GGUFReader, key: str) -> Any:
    if key not in reader.fields:
        return None
    val = reader.fields[key].contents()
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    return val


def inspect_gguf_purity(gguf_path: Path, max_sample_tensors: int = 4) -> Dict[str, Any]:
    """Inspects GGUF KV metadata and dequantizes sample attention/FFN weights on CPU."""
    if not gguf_path.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}")

    reader = gguf.GGUFReader(str(gguf_path))
    arch = _safe_kv(reader, "general.architecture") or "llama"
    kv_meta = {
        "architecture": arch,
        "tensor_count": len(reader.tensors),
        "context_length": _safe_kv(reader, f"{arch}.context_length"),
        "rope_scaling_type": _safe_kv(reader, f"{arch}.rope.scaling.type"),
        "rope_scaling_factor": _safe_kv(reader, f"{arch}.rope.scaling.factor"),
        "rope_orig_ctx": _safe_kv(reader, f"{arch}.rope.scaling.original_context_length"),
        "yarn_beta_fast": _safe_kv(reader, f"{arch}.rope.scaling.yarn_beta_fast"),
        "yarn_beta_slow": _safe_kv(reader, f"{arch}.rope.scaling.yarn_beta_slow"),
    }

    # Select representative sample o_proj (attn_output) and down_proj (ffn_down) tensors
    candidates = [
        t for t in reader.tensors
        if t.name.endswith("attn_output.weight") or t.name.endswith("ffn_down.weight")
    ]
    if len(candidates) > max_sample_tensors:
        step = max(1, len(candidates) // max_sample_tensors)
        sampled = candidates[::step][:max_sample_tensors]
    else:
        sampled = candidates

    tensor_checks: List[Dict[str, Any]] = []
    all_finite = True
    max_p999_ratio = 0.0
    min_sv_ratio = 1.0
    min_erank_ratio = 1.0
    min_row_l2 = 1e9

    for t in sampled:
        arr = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32, copy=False)
        is_finite = bool(np.isfinite(arr).all())
        all_finite = all_finite and is_finite

        flat = arr.reshape(-1)
        usable = (flat.size // 32) * 32
        blocks = flat[:usable].reshape(-1, 32)
        b_max = np.abs(blocks).max(axis=1)
        b_rms = np.sqrt((blocks ** 2).mean(axis=1)) + 1e-9
        ratios = b_max / b_rms
        mean_par = float(np.mean(ratios))
        p999_par = float(np.percentile(ratios, 99.9))
        max_p999_ratio = max(max_p999_ratio, p999_par)

        # Low-rank singular spectrum & effective rank check on 2D weight matrix
        mat = torch.from_numpy(arr)
        if mat.ndim == 2 and min(mat.shape) >= 16:
            row_l2 = float(torch.linalg.vector_norm(mat, dim=1).min().item())
            min_row_l2 = min(min_row_l2, row_l2)
            q_sv = 16
            _, s_vals, _ = torch.svd_lowrank(mat, q=q_sv, niter=2)
            sv_ratio = float((s_vals[-1] / (s_vals[0] + 1e-12)).item())
            p_s = (s_vals / (s_vals.sum() + 1e-12)).clamp(min=1e-12)
            erank = float(torch.exp(-(p_s * torch.log(p_s)).sum()).item())
            erank_ratio = erank / float(q_sv)
        else:
            row_l2 = 1.0
            sv_ratio = 1.0
            erank_ratio = 1.0
        min_sv_ratio = min(min_sv_ratio, sv_ratio)
        min_erank_ratio = min(min_erank_ratio, erank_ratio)

        tensor_checks.append({
            "name": t.name,
            "shape": list(arr.shape),
            "dtype": str(t.tensor_type.name),
            "finite": is_finite,
            "min_row_l2": round(row_l2, 5),
            "block32_mean_par": round(mean_par, 3),
            "block32_q999_par": round(p999_par, 3),
            "sv16_tail_ratio": round(sv_ratio, 4),
            "sv16_erank_ratio": round(erank_ratio, 4),
        })

    del reader
    purity_ok = (
        all_finite
        and (max_p999_ratio < 5.0)
        and (min_sv_ratio > 1e-4)
        and (min_erank_ratio >= 0.50)
        and (min_row_l2 > 1e-6)
    )
    return {
        "gguf_path": str(gguf_path),
        "file_size_bytes": gguf_path.stat().st_size,
        "kv_metadata": kv_meta,
        "sampled_tensors": tensor_checks,
        "all_finite": all_finite,
        "max_block32_q999_par": round(max_p999_ratio, 3),
        "min_sv16_tail_ratio": round(min_sv_ratio, 4),
        "min_sv16_erank_ratio": round(min_erank_ratio, 4),
        "min_row_l2": round(min_row_l2, 5),
        "gguf_purity_ok": purity_ok,
    }


def inspect_stage_reports(report_paths: List[Path]) -> Dict[str, Any]:
    """Cross-checks Stage B/C/D JSON reports for param_diff == 0, max_cond <= 1e4, and NLL monotonicity."""
    if not report_paths:
        return {"reports_checked": 0, "reports_ok": True}

    param_counts: List[int] = []
    max_cond_seen = 0.0
    nll_ok = True
    details: List[Dict[str, Any]] = []

    for rp in report_paths:
        if not rp.exists():
            raise FileNotFoundError(f"Report not found: {rp}")
        data = json.loads(rp.read_text(encoding="utf-8"))

        pa = data.get("param_audit", {})
        if "total_params" in pa:
            param_counts.append(int(pa["total_params"]))
        assert pa.get("finite", True) is True, f"Non-finite param audit in {rp}"

        # Condition number checks across Stage C/D
        cond_stats = data.get("cond_stats", {})
        for v in cond_stats.values():
            max_cond_seen = max(max_cond_seen, float(v))
        for acc in data.get("accepted_layers", []):
            for k, v in acc.get("stats", {}).items():
                if k.endswith("_cond"):
                    max_cond_seen = max(max_cond_seen, float(v))

        # Monotonic NLL check
        base_nll = data.get("base_val_nll", data.get("base_dev_nll"))
        end_nll = data.get("stageb_val_nll", data.get("final_val_nll", data.get("best_metrics", {}).get("val_nll")))
        if base_nll is not None and end_nll is not None:
            if float(end_nll) > float(base_nll) + 1e-5:
                nll_ok = False

        details.append({
            "report": rp.name,
            "total_params": pa.get("total_params"),
            "base_nll": base_nll,
            "end_nll": end_nll,
        })

    param_diff_zero = len(set(param_counts)) <= 1 if param_counts else True
    cond_ok = max_cond_seen <= 1e4
    return {
        "reports_checked": len(report_paths),
        "param_diff_zero": param_diff_zero,
        "max_cond_seen": round(max_cond_seen, 2),
        "cond_ok": cond_ok,
        "nll_monotonic_ok": nll_ok,
        "details": details,
        "reports_ok": param_diff_zero and cond_ok and nll_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only GGUF Weight-Purity Inspector for v23")
    parser.add_argument("--gguf", type=Path, required=True, help="Path to GGUF file")
    parser.add_argument("--report", type=Path, nargs="*", default=[], help="Optional Stage B/C/D JSON report(s)")
    args = parser.parse_args()

    gguf_res = inspect_gguf_purity(args.gguf)
    rep_res = inspect_stage_reports(args.report)
    overall_ok = gguf_res["gguf_purity_ok"] and rep_res["reports_ok"]

    out = {
        "overall_ok": overall_ok,
        "gguf": gguf_res,
        "reports": rep_res,
    }
    print(json.dumps(out, indent=2))
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
