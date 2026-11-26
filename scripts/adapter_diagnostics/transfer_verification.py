#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C18-SHEPHERD / brief #911 OWNER TASK 4 -- FUNCTIONAL TRANSFER VERIFICATION.

QUESTION UNDER TEST
-------------------
Does `exports\\quillan-mini-sovereign-bitnet158.safetensors` (815,396,270 params,
sha256 53e60d33...ac7a5c) actually carry the Qwen2.5-0.5B donor's COMPUTATION,
or merely its numbers parked in shape-compatible slots?

Prior "evidence" (#894) was STRUCTURAL ONLY: 530/530 strict keys, donor fraction
0.9072. #906/#907 showed prompt-blind output. This instrument is the missing
FUNCTIONAL test: feed the SAME fixed prompt set to (a) the clean donor and
(b) the custom checkpoint, then compare representations at every aligned
conceptual point where the dimension allows it.

STRICT READ-ONLY CONTRACT
-------------------------
  * Never writes to any model, checkpoint, tokenizer, or config.
  * Writes NEW files only, under scripts\\adapter_diagnostics\\.
  * Every comparison states VALID (same width / same vocab) or INVALID
    (architecturally non-corresponding), explicitly. No silent comparisons.

OUTPUT
------
  transfer_verification_results.json  (all exact numbers)
  transfer_verification_report.md     (human-readable verdict)
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
DIAG = ROOT / "scripts" / "adapter_diagnostics"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "03 - Training & Model"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SNAP = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
)
CKPT = ROOT / "exports" / "quillan-mini-sovereign-bitnet158.safetensors"
ACTORS = DIAG / "_acts"  # intermediate activation cache (NEW dir)

# The SAME fixed prompts go to BOTH models. First four are the exact strings
# used in #906/#907 so this test is directly comparable to the failure report.
PROMPTS: Tuple[str, ...] = (
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Quillan-Ronin is a sovereign",
    "The quick brown fox jumps over the lazy dog and then",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def cos_rows(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Per-token cosine similarity between two [1,L,D] tensors."""
    a = a.float().reshape(-1, a.shape[-1])
    b = b.float().reshape(-1, b.shape[-1])
    if a.shape != b.shape:
        return {"valid": False, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    c = F.cosine_similarity(a, b, dim=-1)
    return {
        "valid": True,
        "mean": round(float(c.mean()), 6),
        "min": round(float(c.min()), 6),
        "max": round(float(c.max()), 6),
        "std": round(float(c.std()), 6),
    }


def cos_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return round(
        float(
            F.cosine_similarity(
                a.float().reshape(1, -1), b.float().reshape(1, -1), dim=-1
            )[0]
        ),
        6,
    )


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """Linear CKA on [n_tokens, d] matrices. Tokens=samples, features=dims.
    Returns None if n<3 or widths differ (we do NOT project mismatched widths).
    """
    x = x.float().reshape(-1, x.shape[-1])
    y = y.float().reshape(-1, y.shape[-1])
    if x.shape[0] < 3 or y.shape[0] < 3 or x.shape[1] != y.shape[1]:
        return None
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xtx = x @ x.t()
    yty = y @ y.t()
    hsic = float((xtx * yty).sum())
    den = math.sqrt(float((xtx * xtx).sum()) * float((yty * yty).sum()))
    return round(hsic / den, 6) if den > 1e-12 else None


def topk_sets(logits: torch.Tensor, k: int) -> List[set[int]]:
    flat = logits.float().reshape(-1, logits.shape[-1])  # [L, V]
    return [set(row.topk(k).indices.tolist()) for row in flat]


def overlap_stats(
    a: torch.Tensor, b: torch.Tensor, ks: Sequence[int] = (1, 5, 10, 20)
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"valid": a.shape == b.shape}
    if a.shape != b.shape:
        return out
    vocab = a.shape[-1]
    for k in ks:
        sa = topk_sets(a, k)
        sb = topk_sets(b, k)
        hits = [len(x & y) for x, y in zip(sa, sb)]
        out[f"top{k}"] = {
            "mean_overlap": round(sum(hits) / len(hits), 4),
            "mean_jaccard": round(
                sum(len(x & y) / len(x | y) for x, y in zip(sa, sb)) / len(sa), 6
            ),
            "per_position": hits,
            "chance_overlap": round(k * k / vocab, 6),
        }
    return out


def rank_of(token_id: int, logits_row: torch.Tensor) -> int:
    """1-based rank of token_id in a logits row (1 = top-1)."""
    return int((logits_row.float() > logits_row.float()[token_id]).sum().item()) + 1


def build_query_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(SNAP), local_files_only=True)
    return tok


def encode_all(tok: Any) -> Dict[str, List[int]]:
    return {
        p: [int(x) for x in tok.encode(p, add_special_tokens=False)] for p in PROMPTS
    }


# --------------------------------------------------------------------------- #
# Phase A -- CLEAN DONOR
# --------------------------------------------------------------------------- #
def phase_donor(tokenized: Dict[str, List[int]]) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM

    log("[donor] loading clean Qwen2.5-0.5B-Instruct from HF snapshot ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(SNAP), local_files_only=True, dtype=torch.float32
    )
    model.eval()
    log(f"[donor] load {time.perf_counter() - t0:.1f}s")

    capture: Dict[str, Any] = {}
    with torch.no_grad():
        for p, ids in tokenized.items():
            x = torch.tensor([ids], dtype=torch.long)
            out = model(x, output_hidden_states=True)
            hs = out.hidden_states  # tuple (embed + 24 layers), each [1,L,896]
            capture[p] = {
                "ids": ids,
                "embed": model.model.embed_tokens(x).float().clone(),
                "hidden_states": [h.float().clone() for h in hs],
                "final_hidden": hs[-1].float().clone(),
                "logits": out.logits.float().clone(),
            }
            del x, out, hs
    del model
    gc.collect()
    ACTORS.mkdir(parents=True, exist_ok=True)
    torch.save(capture, str(ACTORS / "donor_acts.pt"))
    # Summary metrics only; keep full tensors out of JSON.
    return {
        "source": str(SNAP),
        "n_hidden_states": len(next(iter(capture.values()))["hidden_states"]),
        "width": capture[PROMPTS[0]]["final_hidden"].shape[-1],
        "logits_width": capture[PROMPTS[0]]["logits"].shape[-1],
        "token_lengths": {p: len(v["ids"]) for p, v in capture.items()},
        "top1_by_prompt": {
            p: [
                [
                    tok_str(int(i))
                    for i in v["logits"][0, -1, :].topk(3).indices.tolist()
                ]
                for v in [capture[p]]
            ][0]
            for p in PROMPTS
        },
    }


_TOK: Any = None


def tok_str(i: int) -> str:
    if _TOK is None:
        return str(i)
    return repr(_TOK.decode([i], skip_special_tokens=True))


# --------------------------------------------------------------------------- #
# Phase B -- CUSTOM CHECKPOINT
# --------------------------------------------------------------------------- #
def phase_custom(tokenized: Dict[str, List[int]]) -> Dict[str, Any]:
    from probe_longctx_quality import build_real_model

    log("[custom] loading 815M checkpoint (strict) ...")
    t0 = time.perf_counter()
    model, meta = build_real_model()
    log(f"[custom] load {time.perf_counter() - t0:.1f}s params={meta['param_count']:,}")

    modules = {
        "emb_lookup": model.ingestion.txt_emb,
        "ingestion": model.ingestion,
        "decomposition": model.decomposition,
        "diffusion_core": model.diffusion_core,
        "moe": model.moe,
        "final_norm": model.final_norm,
        "logits": model.txt_dec,
    }
    capture: Dict[str, Any] = {}
    for p, ids in tokenized.items():
        acts: Dict[str, torch.Tensor] = {}
        handles = []

        def mk(name: str):
            def hook(mod: Any, inp: Any, out: Any) -> None:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    acts[name] = t.detach().float().clone()

            return hook

        for name, mod in modules.items():
            handles.append(mod.register_forward_hook(mk(name)))

        # Capture the actual decoder input (= final hidden state) via hook input.
        def dec_in(mod: Any, inp: Any, out: Any) -> None:
            t = inp[0] if isinstance(inp, (tuple, list)) else inp
            if isinstance(t, torch.Tensor):
                acts["decoder_input"] = t.detach().float().clone()

        handles.append(model.txt_dec.register_forward_hook(dec_in))

        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            _ = model(x)
        for h in handles:
            h.remove()
        capture[p] = {"ids": ids, **acts}
        del x, acts, handles
        gc.collect()

    del model
    gc.collect()
    torch.save(capture, str(ACTORS / "custom_acts.pt"))
    return {
        "checkpoint": str(CKPT),
        "param_count": meta["param_count"],
        "captured_modules": sorted(modules.keys()) + ["decoder_input"],
        "width": capture[PROMPTS[0]]["emb_lookup"].shape[-1],
        "logits_width": capture[PROMPTS[0]]["logits"].shape[-1],
    }


# --------------------------------------------------------------------------- #
# Phase C -- COMPARISON
# --------------------------------------------------------------------------- #
def phase_compare(tokenized: Dict[str, List[int]]) -> Dict[str, Any]:
    d = torch.load(
        str(ACTORS / "donor_acts.pt"), map_location="cpu", weights_only=False
    )
    c = torch.load(
        str(ACTORS / "custom_acts.pt"), map_location="cpu", weights_only=False
    )
    R: Dict[str, Any] = {}

    # ---- 1. WEIGHT-LEVEL / INPUT EMBEDDING (VALID: both [L,896], same vocab) ----
    R["A_input_embedding_cosine"] = {
        p: cos_rows(c[p]["emb_lookup"], d[p]["embed"]) for p in PROMPTS
    }
    R["A_embedding_validity"] = (
        "VALID: donor embed_tokens and custom ingestion.txt_emb are both [L,896] "
        "over the same 151936-token Qwen vocabulary; this is the purest test of "
        "whether donor knowledge (the token manifold) physically arrived."
    )

    # ---- 2. FINAL HIDDEN STATE (VALID width 896; semantic caveat noted) ----
    R["B_final_hidden_cosine_vs_donor_last_layer"] = {
        p: cos_rows(c[p]["decoder_input"], d[p]["final_hidden"]) for p in PROMPTS
    }
    # Which donor layer does the custom final hidden resemble most? (CKA, tokens=samples)
    all_c = torch.cat([c[p]["decoder_input"] for p in PROMPTS], dim=1).squeeze(0)
    cka_by_layer: Dict[str, Optional[float]] = {}
    for i, h in enumerate(d[PROMPTS[0]]["hidden_states"]):
        all_d = torch.cat([d[p]["hidden_states"][i] for p in PROMPTS], dim=1).squeeze(0)
        cka_by_layer[f"donor_layer_{i}"] = linear_cka(all_c, all_d)
    R["B_final_hidden_linear_CKA_by_donor_layer"] = cka_by_layer
    R["B_final_hidden_validity"] = (
        "VALID width (896==896) for cosine and linear CKA. INVALID as a claim of "
        "layer correspondence: the custom graph is NOT a transformer stack, so "
        "'custom final hidden ~ donor layer k' measures representational geometry "
        "only, not a layer-aligned functional equivalence."
    )

    # ---- 3. LOGITS (VALID: both [L,151936], identical vocab) ----
    R["C_logits_cosine"] = {
        p: cos_rows(c[p]["logits"], d[p]["logits"]) for p in PROMPTS
    }
    R["C_logits_overlap"] = {
        p: overlap_stats(c[p]["logits"], d[p]["logits"]) for p in PROMPTS
    }
    # Cross-prompt confusion: does custom logit for prompt i match donor for prompt j?
    conf = []
    for i, pi in enumerate(PROMPTS):
        row = []
        for j, pj in enumerate(PROMPTS):
            row.append(
                cos_rows(c[pi]["logits"][:, -1, :], d[pj]["logits"][:, -1, :])["mean"]
            )
        conf.append(row)
    R["C_cross_prompt_logit_cosine_custom_vs_donor"] = {
        "rows": [tok_str_hash(p) for p in PROMPTS],
        "matrix": conf,
    }
    # Donor top-1 token's rank inside custom distribution (last position).
    rank_rows = []
    for p in PROMPTS:
        dt1 = int(d[p]["logits"][0, -1, :].argmax().item())
        rank_rows.append(
            {
                "prompt": p,
                "donor_top1_id": dt1,
                "donor_top1_token": tok_str(dt1),
                "rank_in_custom": rank_of(dt1, c[p]["logits"][0, -1, :]),
            }
        )
    R["C_donor_top1_rank_in_custom"] = rank_rows
    R["C_logits_validity"] = (
        "VALID: both produce [L,151936] logits over the identical Qwen tokenizer. "
        "Top-k overlap chance baseline = k^2/151936 (e.g. 1 -> 0.0000066, 20 -> 0.0026)."
    )

    # ---- 4. PROMPT SENSITIVITY / DISCRIMINATION (the decisive transfer test) ----
    sens: Dict[str, Any] = {}
    for stage in (
        "emb_lookup",
        "ingestion",
        "decomposition",
        "diffusion_core",
        "moe",
        "final_norm",
        "logits",
    ):
        pairs = []
        for i in range(len(PROMPTS)):
            for j in range(i + 1, len(PROMPTS)):
                a = c[PROMPTS[i]][stage][0, -1, :].float().reshape(1, -1)
                b = c[PROMPTS[j]][stage][0, -1, :].float().reshape(1, -1)
                if a.shape == b.shape:
                    pairs.append(round(float(F.cosine_similarity(a, b)[0]), 6))
        sens[f"custom_{stage}"] = {
            "mean_pair_cosine": round(sum(pairs) / len(pairs), 6) if pairs else None,
            "interpretation": "1.0 = prompt-blind (all prompts collapse); ~0 = discriminative",
        }
    # Donor control at final hidden + logits.
    for stage in ("final_hidden", "logits"):
        pairs = []
        for i in range(len(PROMPTS)):
            for j in range(i + 1, len(PROMPTS)):
                a = d[PROMPTS[i]][stage][0, -1, :].float().reshape(1, -1)
                b = d[PROMPTS[j]][stage][0, -1, :].float().reshape(1, -1)
                pairs.append(round(float(F.cosine_similarity(a, b)[0]), 6))
        sens[f"donor_{stage}"] = {"mean_pair_cosine": round(sum(pairs) / len(pairs), 6)}
    R["D_prompt_discrimination"] = sens

    # ---- 5. FIRST DIVERGENCE POINT (chained similarity from the shared input) ----
    # At the input the two models share the donor embedding exactly. Walk the custom
    # graph forward and record cosine of each stage vs the donor embedding (same
    # width 896 for all hidden stages) plus the prompt-collapse statistic above.
    chain = {}
    for stage in (
        "emb_lookup",
        "ingestion",
        "decomposition",
        "diffusion_core",
        "moe",
        "final_norm",
    ):
        vals = [cos_rows(c[p][stage], d[p]["embed"])["mean"] for p in PROMPTS]
        chain[stage] = round(sum(vals) / len(vals), 6)
    R["E_cosine_to_donor_embedding_along_custom_graph"] = chain

    return R


def tok_str_hash(p: str) -> str:
    return p[:24]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="reuse cached activation .pt files in _acts/",
    )
    args = ap.parse_args()

    global _TOK
    DIAG.mkdir(parents=True, exist_ok=True)
    ACTORS.mkdir(parents=True, exist_ok=True)
    _TOK = build_query_tokenizer()
    tokenized = encode_all(_TOK)

    result: Dict[str, Any] = {
        "owner": "c18-shepherd",
        "brief": 911,
        "question": "functional donor-knowledge transfer: computation vs shape-compatible numbers",
        "prompts": list(PROMPTS),
    }
    if args.compare_only:
        result["phase_donor"] = {"source": str(SNAP), "mode": "cached"}
        result["phase_custom"] = {"checkpoint": str(CKPT), "mode": "cached"}
    else:
        result["phase_donor"] = phase_donor(tokenized)
        result["phase_custom"] = phase_custom(tokenized)
    result["comparison"] = phase_compare(tokenized)

    out_json = DIAG / "transfer_verification_results.json"
    out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"WROTE {out_json}")
    log("TRANSFER_VERIFICATION_DONE")


if __name__ == "__main__":
    main()
