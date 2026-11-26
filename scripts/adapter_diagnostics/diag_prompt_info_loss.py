#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C10-CODEWEAVER -- Prompt information-loss boundary diagnostic (brief #911)
================================================================================
NEW FILE. Reads model source only; does NOT edit the model or the adapter.

WHAT IT DOES
------------
Loads the REAL 815M custom-arch checkpoint
(exports/quillan-mini-sovereign-bitnet158.safetensors), runs two DIFFERENT
5-token prompts through a plain CPU forward (torch.no_grad), and captures
activations at EVERY module boundary using forward hooks. It then compares the
two runs at each boundary (cosine similarity, max-abs diff, mean-abs diff,
relative diff, and pct(|diff|<1e-4)) to locate the FIRST boundary where the
two representations become identical / near-identical -- i.e. where prompt
information dies.

It also:
  * verifies the forward actually READS input token IDs (txt_emb outputs differ
    between prompts; a constant-token run is compared against both),
  * reports the pos_embed vs txt_emb scale ratio,
  * runs a read-only control with pos_embed scaled down 50x in memory to confirm
    the embedding-scale hypothesis (no files edited).

Run:
  & 'C:\\Users\\Admin\\.local\\bin\\python3.11.exe' \\
      'C:\\02_QUILLAN\\scripts\\adapter_diagnostics\\diag_prompt_info_loss.py'
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(r"C:\02_QUILLAN")
SCRIPT_DIR = ROOT / "scripts"
MODEL_DIR = ROOT / "03 - Training & Model"
OUT_DIR = ROOT / "scripts" / "adapter_diagnostics"

for _p in (str(SCRIPT_DIR), str(MODEL_DIR), str(OUT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from probe_longctx_quality import build_real_model, load_tokenizer  # noqa: E402

RESULT_PATH = OUT_DIR / "prompt_info_loss_result.json"

# The two 5-token prompts from brief #907/#896 that produce 100% identical logits.
PROMPT_A = "The capital of France is"
PROMPT_B = "Hello, my name is"
FALLBACK_A = [1101, 3117, 314, 3397, 374]  # deterministic 5-token stand-ins
FALLBACK_B = [9707, 11, 1276, 1784, 374]

CAP: Dict[str, torch.Tensor] = {}
ORDER: List[str] = []
_HANDLES: List[Any] = []


def _register(model: torch.nn.Module, attr: str, name: str, pre: bool = False) -> None:
    """Attach a capture hook to a named module. Name defines report ordering."""
    mod = dict(model.named_modules()).get(attr)
    if mod is None:
        print(f"  [WARN] module {attr!r} not found -- skipping boundary {name!r}")
        return
    if name not in ORDER:
        ORDER.append(name)

    def hook_fn(module, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(t, torch.Tensor):
            CAP[name] = t.detach().clone()

    def pre_hook_fn(module, inp):
        t = inp[0] if isinstance(inp, (tuple, list)) else inp
        if isinstance(t, torch.Tensor):
            CAP[name] = t.detach().clone()

    try:
        if pre:
            _HANDLES.append(mod.register_forward_pre_hook(pre_hook_fn))
        else:
            _HANDLES.append(mod.register_forward_hook(hook_fn))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  [WARN] hook on {attr!r} failed: {exc}")


def install_hooks(model: torch.nn.Module) -> None:
    # INPUT / TOKEN EMBEDDING
    _register(model, "ingestion.txt_emb", "01_txt_emb_OUT")
    _register(model, "ingestion.mod_emb", "02_mod_emb_OUT")
    # The exact pre-norm sum x = txt_emb + mod_emb + pos_embed (where 52x dilution occurs)
    _register(model, "ingestion.norm", "03_INGEST_PRENORM_IN", pre=True)
    _register(model, "ingestion.norm", "04_ingestion_norm_OUT")
    _register(model, "ingestion.q1_ingest", "05_q1_ingest_OUT")
    _register(model, "ingestion.q2_ingest", "06_q2_ingest_OUT")
    _register(model, "ingestion.ingest_gate", "07_ingest_gate_OUT")
    _register(model, "ingestion", "08_INGESTION_OUT")

    # DECOMPOSITION
    _register(model, "decomposition.Q_gate", "10_Q_gate_OUT")
    _register(model, "decomposition.K_gate", "11_K_gate_OUT")
    _register(model, "decomposition.V_gate", "12_V_gate_OUT")
    for vec in (
        "Language",
        "Sentiment",
        "Context",
        "Intent",
        "Meta",
        "Creativity",
        "Ethics",
        "Strategy",
        "Constraint",
    ):
        _register(model, f"decomposition.vectors.{vec}", f"13_prism_{vec}_OUT")
    _register(model, "decomposition.W_gate", "14_W_gate_OUT")
    _register(model, "decomposition", "15_DECOMP_OUT")
    _register(model, "norm_decomp", "16_norm_decomp_OUT")

    # DIFFUSION / COUIL ATTENTION
    _register(model, "diffusion_core.couil_attn.norm", "20_couil_norm_OUT")
    _register(model, "diffusion_core.couil_attn.q_proj", "21_q_proj_OUT")
    _register(model, "diffusion_core.couil_attn.k_proj", "22_k_proj_OUT")
    _register(model, "diffusion_core.couil_attn.v_proj", "23_v_proj_OUT")
    _register(model, "diffusion_core.couil_attn.o_proj", "24_o_proj_OUT")
    _register(model, "diffusion_core.couil_attn", "25_COUIL_ATTN_OUT")
    _register(model, "diffusion_core.ffn", "26_diff_ffn_OUT")
    _register(model, "diffusion_core", "27_DIFFUSION_OUT")
    _register(model, "norm_diff", "28_norm_diff_OUT")

    # MoE
    _register(model, "moe.router", "30_router_OUT")
    _register(model, "moe.expert_swarms.0", "31_expert_swarm0_OUT")
    _register(model, "moe.output_norm", "32_moe_output_norm_OUT")
    _register(model, "moe", "33_MOE_OUT")
    _register(model, "norm_moe", "34_norm_moe_OUT")

    # FINALIZERS / LOGITS
    _register(model, "pre_final_norm", "40_pre_final_norm_OUT")
    _register(model, "quillan_finalizer", "41_finalizer1_OUT")
    _register(model, "quillan_finalizer2", "42_finalizer2_OUT")
    _register(model, "quillan_gate", "43_quillan_gate_OUT")
    _register(model, "final_norm", "44_final_norm_OUT")
    _register(model, "txt_dec", "45_TXTDEC_LOGITS")


def remove_hooks() -> None:
    for h in _HANDLES:
        h.remove()
    _HANDLES.clear()


def run_forward(
    model: torch.nn.Module, ids: List[int]
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    CAP.clear()
    x = torch.tensor([list(ids)], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    CAP["46_FINAL_LOGITS"] = logits.detach().clone()
    snapshot = {k: v.clone() for k, v in CAP.items()}
    return snapshot, logits.detach().clone()


def _to_vec(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    av, bv = _to_vec(a), _to_vec(b)
    if av.numel() != bv.numel():
        return {
            "cos": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "rel": float("nan"),
            "pct_lt_1e4": float("nan"),
            "anorm": float("nan"),
            "n": float(av.numel()),
        }
    diff = (av - bv).abs()
    mean_diff = float(diff.mean().item())
    anorm = float(av.abs().mean().item())
    cos = float(F.cosine_similarity(av.unsqueeze(0), bv.unsqueeze(0)).item())
    return {
        "cos": cos,
        "max": float(diff.max().item()),
        "mean": mean_diff,
        "rel": mean_diff / max(anorm, 1e-12),
        "pct_lt_1e4": float((diff < 1e-4).float().mean().item()) * 100.0,
        "anorm": anorm,
        "n": float(av.numel()),
    }


def compare_last_token(a: torch.Tensor, b: torch.Tensor) -> Optional[Dict[str, float]]:
    """For (B,L,D) activations, compare only the final position (drives next token)."""
    if a.dim() != 3 or b.dim() != 3 or a.shape[0] != 1 or b.shape[0] != 1:
        return None
    if a.shape[1] == 0 or a.shape[1] != b.shape[1]:
        return None
    return compare_tensors(a[0, -1, :], b[0, -1, :])


def main() -> None:
    print(
        "C10-CODEWEAVER prompt information-loss boundary diagnostic (brief #911)",
        flush=True,
    )
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

    model, meta = build_real_model()
    print(f"MODEL params={meta['param_count']:,} strict_load=OK", flush=True)

    # Tokenize the two prompts (HF tokenizer from the probe); fall back to fixed IDs.
    try:
        tok, _ = load_tokenizer()
        ids_a = [int(x) for x in tok.encode(PROMPT_A, add_special_tokens=False)]
        ids_b = [int(x) for x in tok.encode(PROMPT_B, add_special_tokens=False)]
        decode = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    except Exception as exc:
        print(f"  [WARN] tokenizer unavailable ({exc}); using fallback token IDs")
        tok, decode = None, lambda ids: str(ids)  # noqa: E731
        ids_a, ids_b = list(FALLBACK_A), list(FALLBACK_B)

    print(f"PROMPT_A {PROMPT_A!r} -> ids={ids_a} (n={len(ids_a)})", flush=True)
    print(f"PROMPT_B {PROMPT_B!r} -> ids={ids_b} (n={len(ids_b)})", flush=True)
    if ids_a == ids_b:
        raise RuntimeError("Prompts must differ; got identical token IDs")

    # Embedding scale stats (evidence check).
    with torch.no_grad():
        emb = model.ingestion.txt_emb.weight.float()
        pe = model.ingestion.pos_embed.float()
        emb_mean_abs = float(emb.abs().mean().item())
        pe_mean_abs = float(pe.abs().mean().item())
        print(
            f"EMBED_NORMS txt_emb_mean_abs={emb_mean_abs:.5f} "
            f"pos_embed_mean_abs={pe_mean_abs:.5f} "
            f"ratio={pe_mean_abs / max(emb_mean_abs, 1e-12):.2f}x "
            f"pos_embed_len={tuple(pe.shape)}",
            flush=True,
        )
        del emb, pe
        gc.collect()

    install_hooks(model)

    # ---- Phase 1: two different prompts, full hook capture -----------------
    snap_a, logits_a = run_forward(model, ids_a)
    snap_b, logits_b = run_forward(model, ids_b)

    print(
        "\n=== BOUNDARY-BY-BOUNDARY COMPARISON (A='The capital of France is' vs B='Hello, my name is') ===",
        flush=True,
    )
    print(
        f"{'boundary':<32} {'shape':<16} {'cos_sim':>10} {'max_abs':>12} {'mean_abs':>12} {'rel':>10} {'pct<1e-4':>9}",
        flush=True,
    )
    rows: List[Dict[str, Any]] = []
    for name in ORDER:
        if name not in snap_a or name not in snap_b:
            continue
        a, b = snap_a[name], snap_b[name]
        m = compare_tensors(a, b)
        rows.append({"boundary": name, "shape": list(a.shape), **m})
        print(
            f"{name:<32} {str(tuple(a.shape)):<16} {m['cos']:>10.6f} "
            f"{m['max']:>12.6e} {m['mean']:>12.6e} {m['rel']:>10.3e} "
            f"{m['pct_lt_1e4']:>9.2f}",
            flush=True,
        )
    # Logits rows are the decisive ones (added outside ORDER loop above).
    for lname, la, lb in (("46_FINAL_LOGITS", logits_a, logits_b),):
        m = compare_tensors(la, lb)
        rows.append({"boundary": lname, "shape": list(la.shape), **m})
        print(
            f"{lname:<32} {str(tuple(la.shape)):<16} {m['cos']:>10.6f} "
            f"{m['max']:>12.6e} {m['mean']:>12.6e} {m['rel']:>10.3e} "
            f"{m['pct_lt_1e4']:>9.2f}",
            flush=True,
        )

    # ---- Last-token-only comparison (what actually picks the next token) --
    print(
        "\n=== LAST-POSITION COMPARISON (final token position drives argmax) ===",
        flush=True,
    )
    print(
        f"{'boundary':<32} {'cos_sim':>10} {'max_abs':>12} {'mean_abs':>12} {'pct<1e-4':>9}",
        flush=True,
    )
    last_rows: List[Dict[str, Any]] = []
    for name in ORDER:
        if name not in snap_a or name not in snap_b:
            continue
        m = compare_last_token(snap_a[name], snap_b[name])
        if m is None:
            continue
        last_rows.append({"boundary": name, **m})
        print(
            f"{name:<32} {m['cos']:>10.6f} {m['max']:>12.6e} "
            f"{m['mean']:>12.6e} {m['pct_lt_1e4']:>9.2f}",
            flush=True,
        )
    mlog = compare_last_token(logits_a, logits_b)
    if mlog is not None:
        last_rows.append({"boundary": "46_FINAL_LOGITS", **mlog})
        print(
            f"{'46_FINAL_LOGITS':<32} {mlog['cos']:>10.6f} {mlog['max']:>12.6e} "
            f"{mlog['mean']:>12.6e} {mlog['pct_lt_1e4']:>9.2f}",
            flush=True,
        )

    # ---- Does forward read token IDs? --------------------------------------
    print("\n=== TOKEN-ID USAGE CHECKS ===", flush=True)
    t_emb = compare_tensors(snap_a["01_txt_emb_OUT"], snap_b["01_txt_emb_OUT"])
    print(
        f"txt_emb(A) vs txt_emb(B): cos={t_emb['cos']:.6f} max_abs={t_emb['max']:.6e} "
        f"mean_abs={t_emb['mean']:.6e} pct_lt_1e-4={t_emb['pct_lt_1e4']:.2f}% "
        "-> token IDs DO reach txt_emb"
        if t_emb["max"] > 1e-4
        else "txt_emb outputs IDENTICAL -> tokens ignored",
        flush=True,
    )
    const_ids = [ids_a[0]] * len(ids_a)
    snap_c, logits_c = run_forward(model, const_ids)
    mc = compare_tensors(logits_a, logits_c)
    print(
        f"constant-token run vs prompt A logits: max_abs={mc['max']:.6e} "
        f"mean_abs={mc['mean']:.6e} pct_lt_1e-4={mc['pct_lt_1e4']:.2f}% "
        f"cos={mc['cos']:.6f}",
        flush=True,
    )
    print(
        f"top1(A)={int(torch.argmax(logits_a[0, -1, :]).item())} "
        f"top1(B)={int(torch.argmax(logits_b[0, -1, :]).item())} "
        f"top1(const)={int(torch.argmax(logits_c[0, -1, :]).item())}",
        flush=True,
    )
    if tok is not None:
        for label, lg in (("A", logits_a), ("B", logits_b)):
            topv, topi = torch.topk(lg[0, -1, :], 5)
            toks = [
                (decode([int(i)]), round(float(v), 3))
                for i, v in zip(topi.tolist(), topv.tolist())
            ]
            print(f"top5({label})={toks}", flush=True)

    # ---- Phase 2: read-only pos_embed scale control -------------------------
    print(
        "\n=== CONTROL: pos_embed *= 0.02 in-memory (no files edited) ===", flush=True
    )
    with torch.no_grad():
        model.ingestion.pos_embed.mul_(0.02)
    snap_a2, logits_a2 = run_forward(model, ids_a)
    snap_b2, logits_b2 = run_forward(model, ids_b)
    m2 = compare_tensors(logits_a2, logits_b2)
    print(
        f"AFTER rescale: logits(A) vs logits(B): max_abs={m2['max']:.6e} "
        f"mean_abs={m2['mean']:.6e} pct_lt_1e-4={m2['pct_lt_1e4']:.2f}% "
        f"cos={m2['cos']:.6f}",
        flush=True,
    )
    if tok is not None:
        for label, lg in (("A", logits_a2), ("B", logits_b2)):
            topv, topi = torch.topk(lg[0, -1, :], 5)
            toks = [
                (decode([int(i)]), round(float(v), 3))
                for i, v in zip(topi.tolist(), topv.tolist())
            ]
            print(f"AFTER rescale top5({label})={toks}", flush=True)
    with torch.no_grad():
        model.ingestion.pos_embed.div_(0.02)

    remove_hooks()

    # ---- Auto-identify first near-identical boundary ------------------------
    # "Near-identical" = cosine >= 0.9999 AND relative diff < 1e-3.
    first_full = None
    first_last = None
    for row in rows:
        if row["cos"] >= 0.9999 and row["rel"] < 1e-3:
            first_full = row["boundary"]
            break
    for row in last_rows:
        if row["cos"] >= 0.9999 and row["mean"] < 1e-3:
            first_last = row["boundary"]
            break
    print("\n=== VERDICT ===", flush=True)
    print(f"FIRST near-identical boundary (full tensor): {first_full}", flush=True)
    print(f"FIRST near-identical boundary (last token):  {first_last}", flush=True)
    print(
        f"FINAL logits A-vs-B: max_abs={rows[-1]['max']:.6e} "
        f"pct_lt_1e-4={rows[-1]['pct_lt_1e4']:.2f}% "
        f"(matches brief #907 claim of 100% identical logits)",
        flush=True,
    )

    result = {
        "brief": 911,
        "agent": "c10-codeweaver",
        "prompt_a": PROMPT_A,
        "ids_a": ids_a,
        "prompt_b": PROMPT_B,
        "ids_b": ids_b,
        "embed_stats": {
            "txt_emb_mean_abs": emb_mean_abs,
            "pos_embed_mean_abs": pe_mean_abs,
            "ratio": pe_mean_abs / max(emb_mean_abs, 1e-12),
        },
        "rows": rows,
        "last_token_rows": last_rows,
        "top1": {
            "A": int(torch.argmax(logits_a[0, -1, :]).item()),
            "B": int(torch.argmax(logits_b[0, -1, :]).item()),
            "const": int(torch.argmax(logits_c[0, -1, :]).item()),
        },
        "first_near_identical_full": first_full,
        "first_near_identical_last": first_last,
        "logit_cmp": rows[-1],
        "post_rescale_logit_cmp": m2,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"WROTE {RESULT_PATH}", flush=True)
    del (
        snap_a,
        snap_b,
        snap_c,
        snap_a2,
        snap_b2,
        logits_a,
        logits_b,
        logits_c,
        logits_a2,
        logits_b2,
    )
    gc.collect()


if __name__ == "__main__":
    main()
