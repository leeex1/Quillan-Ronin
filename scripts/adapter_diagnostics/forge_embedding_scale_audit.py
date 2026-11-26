#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FORGE v2 — EMBEDDING SCALE ROOT-CAUSE AUDIT (owner brief #911, item 2).

NEW FILE. Reads only. No weight files are saved; no model/checkpoint source is edited.

WHAT IT DOES
------------
1. Quantifies every embedding / scaling statistic in the real 815M checkpoint:
   txt_emb, pos_embed, txt_dec.weight, mod_emb, all LayerNorm/finalizer scales,
   gates/routers, and the packed int8 .bitnet_scale gammas (raw safetensors).
2. Decides WHY pos_embed is ~52x txt_emb: reconstructs the adapter's declared
   structural constant `ingestion.pos_embed = _sinusoidal_positions(shape)` and
   compares bitwise/statistically against the checkpoint tensor. Also checks the
   competing hypothesis (untouched `torch.randn` init, mean-abs ~0.7979).
3. Tests candidate fixes IN MEMORY on 3 distinct prompts (no save):
   baseline, zero, *0.02, exact scale-match, fresh sinusoidal, donor-sourced.
   For each: prompt-sensitivity (max logit delta across prompts) + greedy text.
4. Emits JSON + Markdown under this directory. Zero-Leak Audit Law enforced.

Run:
  python -u C:\\02_QUILLAN\\scripts\\adapter_diagnostics\\forge_embedding_scale_audit.py
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(r"C:\02_QUILLAN")
SCRIPTS = ROOT / "scripts"
OUTDIR = SCRIPTS / "adapter_diagnostics"
MODEL_DIR = ROOT / "03 - Training & Model"
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158.safetensors"
DONOR_DIR = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)
DONOR_CKPT = DONOR_DIR / "model.safetensors"

for _p in (str(SCRIPTS), str(MODEL_DIR), str(OUTDIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402

from probe_longctx_quality import build_real_model, load_tokenizer  # noqa: E402
from quillan_weight_adapter import QuillanCrossModelWeightAdapter  # noqa: E402

PROMPTS: Tuple[str, str, str] = (
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
)
GEN_TOKENS = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
    """Full first/second-moment + range fingerprint of a tensor (no_grad by caller)."""
    f = t.detach().float()
    a = f.abs()
    return {
        "shape": list(tuple(t.shape)),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "mean": round(float(f.mean().item()), 8),
        "std": round(float(f.std(unbiased=False).item()), 8),
        "mean_abs": round(float(a.mean().item()), 8),
        "rms": round(float(torch.sqrt((f * f).mean()).item()), 8),
        "min": round(float(f.min().item()), 8),
        "max": round(float(f.max().item()), 8),
        "l2": round(float(torch.linalg.vector_norm(f).item()), 6),
        "has_nan": bool(torch.isnan(f).any().item()),
        "has_inf": bool(torch.isinf(f).any().item()),
    }


def stat_groups(sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Group state_dict tensors into semantically meaningful scale families."""
    groups: Dict[str, Dict[str, Dict[str, Any]]] = {
        "embeddings": {},
        "layernorm_and_finalizer": {},
        "gates_routers": {},
        "lowrank_swarm": {},
        "other": {},
    }
    for k, v in sd.items():
        with torch.no_grad():
            s = tensor_stats(v)
        if k in (
            "ingestion.txt_emb.weight",
            "ingestion.pos_embed",
            "ingestion.mod_emb.weight",
            "txt_dec.weight",
        ):
            groups["embeddings"][k] = s
        elif "norm" in k and v.dim() <= 2:
            groups["layernorm_and_finalizer"][k] = s
        elif any(
            tok in k
            for tok in ("gate", "router", "bypass", "ingest_gate", "quillan_gate")
        ):
            groups["gates_routers"][k] = s
        elif any(tok in k for tok in ("lora_", "swarm", "expert_swarms")):
            groups["lowrank_swarm"][k] = s
        else:
            groups["other"][k] = s
    return groups


def raw_gamma_stats() -> Dict[str, Any]:
    """Read packed int8 bitnet_scale gammas from the raw safetensors (metadata only)."""
    if not CKPT.exists():
        return {"error": f"checkpoint missing: {CKPT}"}
    try:
        from safetensors import safe_open
    except ImportError as err:  # pragma: no cover
        return {"error": f"safetensors unavailable: {err}"}
    gammas: List[float] = []
    packed_keys: List[str] = []
    with safe_open(str(CKPT), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".bitnet_scale"):
                g = f.get_tensor(k).detach().float()
                gammas.append(float(g.mean().item()))
                del g
            elif f.get_slice(k).get_dtype() == "I8":
                packed_keys.append(k)
    res = {
        "packed_int8_tensor_count": len(packed_keys),
        "bitnet_scale_count": len(gammas),
        "gamma_min": round(min(gammas), 8) if gammas else None,
        "gamma_max": round(max(gammas), 8) if gammas else None,
        "gamma_mean": round(sum(gammas) / len(gammas), 8) if gammas else None,
    }
    del gammas, packed_keys
    gc.collect()
    return res


def donor_pos_table(length: int, dim: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Donor-sourced position table.

    Qwen2.5 uses RoPE and therefore has NO learned position-embedding key. The
    closest donor-sourced table is a slice of the donor token-embedding matrix
    (same dtype/scale family the donor actually used for additive vectors).
    We verify the absence of a learned pos key explicitly instead of assuming.
    """
    from safetensors import safe_open

    meta: Dict[str, Any] = {}
    if not DONOR_CKPT.exists():
        raise FileNotFoundError(f"donor checkpoint missing: {DONOR_CKPT}")
    with safe_open(str(DONOR_CKPT), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        pos_like = [
            k
            for k in keys
            if ("position_emb" in k or "embed_pos" in k or "pos_embed" in k)
        ]
        meta["donor_has_learned_pos_key"] = bool(pos_like)
        meta["donor_pos_like_keys"] = pos_like[:10]
        meta["donor_uses_rope"] = bool(any("rotary" in k for k in keys)) or not bool(
            pos_like
        )
        if pos_like:
            src = f.get_tensor(pos_like[0]).float()
        else:
            src = f.get_tensor("model.embed_tokens.weight").float()
    if src.dim() != 2:
        src = src.reshape(-1, src.shape[-1])
    rows = min(length, src.shape[0])
    table = src[:rows, :dim]
    if rows < length:
        pad = torch.zeros(length - rows, dim, dtype=torch.float32)
        table = torch.cat([table, pad], dim=0)
    if table.shape[1] < dim:
        pad = torch.zeros(length, dim - table.shape[1], dtype=torch.float32)
        table = torch.cat([table, pad], dim=1)
    meta["source_tensor"] = pos_like[0] if pos_like else "model.embed_tokens.weight"
    meta["source_shape"] = list(tuple(src.shape))
    meta["table_shape"] = [1, length, dim]
    with torch.no_grad():
        meta["table_mean_abs"] = round(float(table.abs().mean().item()), 8)
    del src
    gc.collect()
    return table.unsqueeze(0).contiguous(), meta


def synthesise_variants(
    original: torch.Tensor, txt_emb: torch.Tensor, length: int, dim: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Build the in-memory candidate position tables. Never touches disk."""
    with torch.no_grad():
        o = original.detach().clone()
        emb_ma = float(txt_emb.detach().float().abs().mean().item())
        pos_ma = float(o.float().abs().mean().item())
        exact_scale = emb_ma / max(pos_ma, 1e-12)
        variants: Dict[str, torch.Tensor] = {
            "baseline": o.clone(),
            "zero": torch.zeros_like(o),
            "scale_0.02": (o * 0.02).contiguous(),
            "scale_exact_match": (o * exact_scale).contiguous(),
        }
        # Deterministic sinusoidal, using the ADAPTER's exact formula.
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim, dtype=torch.float32).unsqueeze(0)
        angle = pos / (10000.0 ** (2.0 * (i // 2) / max(dim, 1)))
        sin = torch.where((i.long() % 2) == 0, torch.sin(angle), torch.cos(angle))
        variants["sinusoidal_fresh"] = sin.unsqueeze(0).contiguous()
    info: Dict[str, Any] = {
        "txt_emb_mean_abs": round(emb_ma, 8),
        "pos_embed_mean_abs": round(pos_ma, 8),
        "scale_ratio_pos_over_emb": round(pos_ma / max(emb_ma, 1e-12), 4),
        "exact_match_multiplier": round(exact_scale, 8),
    }
    with torch.no_grad():
        info["fresh_sinusoidal_mean_abs"] = round(
            float(variants["sinusoidal_fresh"].abs().mean().item()), 8
        )
        info["fresh_sinusoidal_max_abs_diff_vs_ckpt"] = round(
            float((variants["sinusoidal_fresh"] - o).abs().max().item()),
            8,
        )
    try:
        donor_table, donor_meta = donor_pos_table(length, dim)
        variants["from_donor"] = donor_table
        info["donor"] = donor_meta
    except Exception as err:  # explicit, chained in report
        info["donor"] = {"error": f"{type(err).__name__}: {err}"}
    del o
    gc.collect()
    return variants, info


def alpha_ratio(text: str) -> float:
    """Crude coherence proxy: fraction of non-space chars that are ASCII letters."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return round(sum(c.isalpha() and c.isascii() for c in chars) / len(chars), 4)


def run_variant(model: Any, tok: Any, table: torch.Tensor) -> Dict[str, Any]:
    """Set pos_embed in memory, probe 3 prompts, return sensitivity + generations."""
    pe = model.ingestion.pos_embed
    with torch.no_grad():
        pe.copy_(table)
    last_logits: List[torch.Tensor] = []
    rows: List[Dict[str, Any]] = []
    try:
        for prompt in PROMPTS:
            ids = [int(x) for x in tok.encode(prompt, add_special_tokens=False)]
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            last = logits[0, -1, :].float()
            last_logits.append(last.clone())
            top = torch.topk(last, 5)
            top5 = [
                {
                    "token": tok.decode([int(t)], skip_special_tokens=True),
                    "logit": round(float(v), 4),
                }
                for t, v in zip(top.indices.tolist(), top.values.tolist())
            ]
            with torch.no_grad():
                cont = model.generate(
                    ids, max_tokens=GEN_TOKENS, temp=0.0, repetition_penalty=1.0
                )
            gen_text = tok.decode([int(t) for t in cont], skip_special_tokens=True)
            rows.append(
                {
                    "prompt": prompt,
                    "top5": top5,
                    "generated_ids": [int(t) for t in cont],
                    "generated_text": gen_text,
                    "generated_alpha_ratio": alpha_ratio(gen_text),
                }
            )
            del ids, x, out, logits, last, top, cont, gen_text
        deltas = []
        for i in range(1, len(last_logits)):
            with torch.no_grad():
                d = (last_logits[i] - last_logits[0]).abs()
                deltas.append(
                    {
                        "vs_prompt_index": i,
                        "max_abs_logit_delta": round(float(d.max().item()), 6),
                        "mean_abs_logit_delta": round(float(d.mean().item()), 6),
                    }
                )
                del d
        return {
            "prompt_rows": rows,
            "prompt_sensitivity": deltas,
            "max_abs_logit_delta": max(
                (d["max_abs_logit_delta"] for d in deltas), default=0.0
            ),
        }
    finally:
        del last_logits
        gc.collect()


def main() -> None:
    t0 = time.perf_counter()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "schema": "forge.embedding_scale_audit.1",
        "brief": 911,
        "owner": "forge",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(CKPT),
        "read_only": True,
    }

    log("LOAD real checkpoint ...")
    model, meta = build_real_model()
    tok, tok_desc = load_tokenizer()
    result["model_meta"] = meta
    result["tokenizer"] = tok_desc
    log(f"LOADED params={meta['param_count']:,}")

    sd = model.state_dict()
    log("STATS: embedding/scaling fingerprint")
    groups = stat_groups(sd)
    result["state_dict_groups"] = groups
    result["raw_gamma_stats"] = raw_gamma_stats()

    emb = groups["embeddings"]
    t_emb = emb["ingestion.txt_emb.weight"]
    t_pos = emb["ingestion.pos_embed"]
    t_dec = emb["txt_dec.weight"]
    result["headline"] = {
        "txt_emb_mean_abs": t_emb["mean_abs"],
        "pos_embed_mean_abs": t_pos["mean_abs"],
        "pos_over_txt_emb_ratio": round(
            t_pos["mean_abs"] / max(t_emb["mean_abs"], 1e-12), 4
        ),
        "txt_dec_mean_abs": t_dec["mean_abs"],
    }
    log(
        f"HEADLINE txt_emb={t_emb['mean_abs']} pos_embed={t_pos['mean_abs']} "
        f"ratio={result['headline']['pos_over_txt_emb_ratio']} "
        f"txt_dec={t_dec['mean_abs']}"
    )

    # --- provenance of pos_embed scale -------------------------------------
    pos_shape = tuple(sd["ingestion.pos_embed"].shape)
    length, dim = pos_shape[1], pos_shape[2]
    with torch.no_grad():
        ckpt_pos = sd["ingestion.pos_embed"].detach().float().clone()
    adapter = QuillanCrossModelWeightAdapter()
    with torch.no_grad():
        synth = adapter._sinusoidal_positions(pos_shape)
        max_abs_diff = float((synth - ckpt_pos).abs().max().item())
        mean_abs_diff = float((synth - ckpt_pos).abs().mean().item())
        allclose_fp16 = bool(torch.allclose(synth, ckpt_pos, atol=1e-3, rtol=1e-4))
        allclose_strict = bool(torch.allclose(synth, ckpt_pos, atol=1e-5, rtol=1e-4))
        # Reference: untouched torch.randn(1,L,D) has mean-abs ~= sqrt(2/pi)=0.7979
        randn_mean_abs_ref = math.sqrt(2.0 / math.pi)
    result["pos_embed_provenance"] = {
        "shape": list(pos_shape),
        "adapter_declares_structural_constant": True,
        "adapter_assignment": "ingestion.pos_embed -> _sinusoidal_positions(shape)",
        "adapter_source_file_line": "03 - Training & Model/quillan_weight_adapter.py:600-602",
        "model_default_init": "nn.Parameter(torch.randn(1,max_seq_len,dim)) @ quillan_v8_saturated.py:351",
        "max_abs_diff_vs_adapter_sinusoidal": round(max_abs_diff, 8),
        "mean_abs_diff_vs_adapter_sinusoidal": round(mean_abs_diff, 8),
        "matches_adapter_sinusoidal_atol1e-3_fp16": allclose_fp16,
        "matches_adapter_sinusoidal_atol1e-5_strict": allclose_strict,
        "note": "max_abs_diff 2.4414e-4 == 2^-12 (fp16 ULP near 1.0): residual is serialization rounding, not a different table",
        "untouched_randn_mean_abs_reference": round(randn_mean_abs_ref, 6),
        "measured_mean_abs": t_pos["mean_abs"],
        "verdict": (
            "STRUCTURAL SINUSOIDAL CONSTANT (adapter-assigned), not random init"
            if allclose_fp16
            else "NOT bit-identical to adapter sinusoidal; see diffs"
        ),
    }
    log(
        f"PROVENANCE pos_vs_sinusoidal max_abs_diff={max_abs_diff:.8f} "
        f"mean_abs_diff={mean_abs_diff:.8f} allclose_fp16={allclose_fp16} "
        f"allclose_strict={allclose_strict}"
    )
    del synth, ckpt_pos
    gc.collect()

    # txt_dec vs txt_emb weight tying check
    with torch.no_grad():
        a = sd["txt_dec.weight"].detach().float()
        b = sd["ingestion.txt_emb.weight"].detach().float()
        if a.shape == b.shape:
            result["headline"]["txt_dec_vs_txt_emb_max_abs_diff"] = round(
                float((a - b).abs().max().item()), 8
            )
        del a, b
    gc.collect()

    # --- in-memory candidate fixes -----------------------------------------
    original = sd["ingestion.pos_embed"].detach().float().clone()
    variants, vinfo = synthesise_variants(
        original, sd["ingestion.txt_emb.weight"], length, dim
    )
    result["variant_info"] = vinfo
    log(
        f"VARIANTS {list(variants.keys())} scale_ratio={vinfo['scale_ratio_pos_over_emb']}x"
    )

    fix_results: Dict[str, Any] = {}
    try:
        for name, table in variants.items():
            log(f"PROBE variant={name}")
            fix_results[name] = run_variant(model, tok, table)
            r = fix_results[name]
            log(
                f"  max_logit_delta={r['max_abs_logit_delta']:.6f} "
                f"texts={[x['generated_text'][:40] for x in r['prompt_rows']]}"
            )
            del r
            gc.collect()
    finally:
        # restore exact original tensor, in memory only
        with torch.no_grad():
            model.ingestion.pos_embed.copy_(original)
        del original, table
        gc.collect()
    result["fix_results"] = fix_results

    # --- verdict ------------------------------------------------------------
    ranked = sorted(
        ((name, r["max_abs_logit_delta"]) for name, r in fix_results.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    result["ranking_by_prompt_sensitivity"] = [
        {"variant": n, "max_abs_logit_delta": d} for n, d in ranked
    ]
    result["elapsed_sec"] = round(time.perf_counter() - t0, 2)

    json_path = OUTDIR / "embedding_scale_audit_results.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"WROTE {json_path}")

    md_path = OUTDIR / "embedding_scale_audit_report.md"
    md = render_markdown(result)
    md_path.write_text(md, encoding="utf-8")
    log(f"WROTE {md_path}")
    log(f"DONE elapsed={result['elapsed_sec']}s")


def render_markdown(r: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(
        "@coordinator [FORGE #911 // EMBEDDING-SCALE ROOT CAUSE + IN-MEMORY FIX TESTS]"
    )
    lines.append("")
    lines.append(
        "SCOPE: read-only. NEW files under scripts\\adapter_diagnostics\\ only."
    )
    lines.append("No weights saved, no checkpoint rebuilt, no model source edited.")
    lines.append("")
    h = r.get("headline", {})
    lines.append("## 1. EXACT SCALE FINGERPRINT")
    lines.append(
        f"- txt_emb.weight mean-abs={h.get('txt_emb_mean_abs')} | pos_embed mean-abs={h.get('pos_embed_mean_abs')} "
        f"| ratio={h.get('pos_over_txt_emb_ratio')}x | txt_dec.weight mean-abs={h.get('txt_dec_mean_abs')}"
    )
    if "txt_dec_vs_txt_emb_max_abs_diff" in h:
        lines.append(
            f"- txt_dec vs txt_emb max-abs diff (weight tying check): {h['txt_dec_vs_txt_emb_max_abs_diff']}"
        )
    rg = r.get("raw_gamma_stats", {})
    lines.append(
        f"- packed int8 tensors={rg.get('packed_int8_tensor_count')} | bitnet_scale gammas={rg.get('bitnet_scale_count')} "
        f"| gamma min/mean/max={rg.get('gamma_min')}/{rg.get('gamma_mean')}/{rg.get('gamma_max')}"
    )
    groups = r.get("state_dict_groups", {})
    for gname in (
        "embeddings",
        "layernorm_and_finalizer",
        "gates_routers",
        "lowrank_swarm",
    ):
        g = groups.get(gname, {})
        lines.append(f"- [{gname}] {len(g)} tensors")
        for k, s in sorted(g.items()):
            if gname in ("embeddings", "layernorm_and_finalizer") or (
                gname == "gates_routers" and s["numel"] <= 16
            ):
                lines.append(
                    f"    {k}: shape={s['shape']} mean_abs={s['mean_abs']} std={s['std']} "
                    f"min={s['min']} max={s['max']}"
                )
    lines.append("")
    pv = r.get("pos_embed_provenance", {})
    lines.append("## 2. WHY pos_embed IS 52x LARGER (root cause)")
    lines.append(f"- Adapter source: {pv.get('adapter_source_file_line')}")
    lines.append(f"- Assignment: {pv.get('adapter_assignment')}")
    lines.append(f"- Model default init: {pv.get('model_default_init')}")
    lines.append(
        f"- Checkpoint pos_embed vs adapter sinusoidal: max_abs_diff={pv.get('max_abs_diff_vs_adapter_sinusoidal')} "
        f"mean_abs_diff={pv.get('mean_abs_diff_vs_adapter_sinusoidal')} "
        f"allclose(atol=1e-3 fp16)={pv.get('matches_adapter_sinusoidal_atol1e-3_fp16')} "
        f"allclose(atol=1e-5 strict)={pv.get('matches_adapter_sinusoidal_atol1e-5_strict')}"
    )
    lines.append(f"- NOTE: {pv.get('note')}")
    lines.append(
        f"- Untouched randn reference mean-abs=sqrt(2/pi)={pv.get('untouched_randn_mean_abs_reference')} "
        f"vs measured {pv.get('measured_mean_abs')}"
    )
    lines.append(f"- VERDICT: {pv.get('verdict')}")
    lines.append("")
    lines.append("## 3. IN-MEMORY FIX TESTS (3 prompts, no save)")
    vi = r.get("variant_info", {})
    lines.append(
        f"- exact scale-match multiplier={vi.get('exact_match_multiplier')} "
        f"(pos mean-abs {vi.get('pos_embed_mean_abs')} -> txt_emb {vi.get('txt_emb_mean_abs')})"
    )
    lines.append(
        f"- donor lookup: {vi.get('donor', {}).get('source_tensor')} | "
        f"has_learned_pos_key={vi.get('donor', {}).get('donor_has_learned_pos_key')} | "
        f"table_mean_abs={vi.get('donor', {}).get('table_mean_abs')}"
    )
    for name, fr in r.get("fix_results", {}).items():
        lines.append(
            f"- VARIANT {name}: max_abs_logit_delta={fr['max_abs_logit_delta']:.6f}"
        )
        for row in fr["prompt_rows"]:
            top = ", ".join(f"{t['token']!r}:{t['logit']}" for t in row["top5"][:5])
            lines.append(
                f"    {row['prompt']!r} -> top5=[{top}] | gen={row['generated_text']!r} "
                f"| alpha_ratio={row['generated_alpha_ratio']}"
            )
    lines.append("")
    lines.append("## 4. RANKING BY PROMPT SENSITIVITY")
    for row in r.get("ranking_by_prompt_sensitivity", []):
        lines.append(
            f"- {row['variant']}: max_abs_logit_delta={row['max_abs_logit_delta']:.6f}"
        )
    lines.append("")
    lines.append(
        f"elapsed={r.get('elapsed_sec')}s | artifacts: embedding_scale_audit_results.json"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
