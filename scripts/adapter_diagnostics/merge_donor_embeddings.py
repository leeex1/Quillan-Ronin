#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
#956/#957 — THREE-DONOR, PROCRUSTES-ALIGNED, TOKEN-STRING-ALIGNED EMBEDDING MERGE
================================================================================

WHY THIS MODULE EXISTS
----------------------
Diagnosis (verified 2026-11-20): ``quillan_weight_adapter.adapt_state_dict`` took
ONE source state_dict and picked a SINGLE embedding via
``_find(src, "model.embed_tokens.weight", ...)``. ``_pool_donors`` only pools
per-layer PROJECTIONS *within one source*. Consequently only ONE donor's token
semantics ever reached ``ingestion.txt_emb`` / ``txt_dec``; the other two donors'
token meaning was silently discarded. Prime suspect for prompt-blind output.

CRITICAL REFINEMENT (#957): token-string matching ALONE is insufficient. Even
when Qwen's 'but' is matched to Llama's 'but' by string, the vectors live in
DIFFERENT learned coordinate systems; direct averaging is meaningless. Therefore
we first solve orthogonal Procrustes on >=512 shared token strings and rotate
EACH ENTIRE donor table into the target Qwen frame, THEN merge.

PIPELINE
--------
  1. Build token-string -> id maps per donor; canonicalize byte-level markers so
     'Ġbut'/'▁but'/'but' all map to one shared string. Donors use DIFFERENT ids
     for the same word ('but' = Qwen 8088, Llama 8248, Dragon 2538) — alignment
     is by STRING, never by raw index.
  2. Reuse the EXISTING primitive
     ``v23_common.compute_shared_vocab_procrustes`` (NOT reimplemented):
     P_d2h = U @ V^T from SVD of centered X_donor^T @ X_qwen over shared strings.
  3. Rotate the ENTIRE donor table (q,gamma reconstructed to float) by P_d2h —
     including non-overlap tokens — so all donors share one coordinate frame.
  4. Merge in the aligned frame:
       OVERLAP (string in 2+ donors): unit-normalize each aligned vector, average
         into ONE canonical vector (no donor's scale dominates).
       NON-OVERLAP (string in 1 donor): preserve its aligned normalized vector.
       MISSING in all donors: documented ZERO structural default.
  5. Re-quantize merged floats back to (q int8, gamma) with the SAME per-channel
     absmean primitive (build_coherent_ternary_base.py:49-56), reused via
     quillan_weight_adapter.to_ternary_pair(..., group_dim=-1).
  6. The SAME merged float weight feeds txt_emb AND txt_dec (via adapter override).

INVARIANTS: deterministic (no RNG), NO BACKPROP, donor gamma used at decode
(never re-scaled from an already-ternary code), honest coverage reported.

Author: C10-CODEWEAVER (Build Loop), v5.4.0-oni — directives #956/#957
"""

from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\02_QUILLAN")
MODEL_DIR = ROOT / "03 - Training & Model"
SCRIPTS = ROOT / "scripts"
for _p in (str(MODEL_DIR), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Byte-identical reuse of build_coherent_ternary_base.py:49-56 quantize_to_ternary.
from quillan_weight_adapter import to_ternary_pair  # noqa: E402

# REUSE the existing Procrustes primitive -- do not reimplement (#957).
import v23_common as v23  # noqa: E402

HF_HUB = Path(r"C:\Users\Admin\.cache\huggingface\hub")
DONOR_CACHE = ROOT / "exports" / "donor_cache_v23"

TARGET_VOCAB = 151936
TARGET_DIM = 896
EPS = 1e-8


# --------------------------------------------------------------------------- #
# Donor specifications (fixed order => determinism)                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DonorSpec:
    name: str
    ckpt: Path
    tokenizer_glob: Optional[str] = None
    tokens_json: Optional[Path] = None


def _resolve(glob_pattern: str) -> Path:
    matches = sorted(HF_HUB.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(
            f"no HF tokenizer matches {glob_pattern!r} under {HF_HUB}"
        )
    return matches[0]


DONOR_SPECS: Tuple[DonorSpec, ...] = (
    DonorSpec(
        name="qwen",
        ckpt=DONOR_CACHE / "qwen_0.5b_donors_ternary.pt",
        tokenizer_glob="models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/tokenizer.json",
    ),
    DonorSpec(
        name="llama",
        ckpt=DONOR_CACHE / "llama_1b_donors_ternary.pt",
        tokenizer_glob="models--unsloth--Llama-3.2-1B-Instruct/snapshots/*/tokenizer.json",
    ),
    DonorSpec(
        name="dragon",
        ckpt=DONOR_CACHE / "dragon_1b_donors_ternary.pt",
        tokens_json=DONOR_CACHE / "dragon_tokens.json",
    ),
)


# --------------------------------------------------------------------------- #
# Token-string canonicalization                                                 #
# --------------------------------------------------------------------------- #
def canonicalize_token(s: str) -> str:
    """Map byte-level BPE pieces from all families to one comparable string."""
    s = s.replace("Ġ", " ").replace("▁", " ").replace("Â", "")
    s = unicodedata.normalize("NFKC", s)
    return s.strip()


@dataclass(frozen=True)
class TokenStringMaps:
    name: str
    id_to_raw: Mapping[int, str]
    raw_to_ids: Mapping[str, Tuple[int, ...]]
    canon_to_ids: Mapping[str, Tuple[int, ...]]
    size: int


# Shared representative-picker used by both alignment and Procrustes vocab.
def _prefer_id(maps: "TokenStringMaps", candidates: Sequence[int], canon: str) -> int:
    return int(
        min(candidates, key=lambda i: (0 if maps.id_to_raw[i] == canon else 1, i))
    )


def load_token_string_maps(spec: DonorSpec) -> TokenStringMaps:
    """Build id<->string maps for one donor from its tokenizer / vocab list."""
    if spec.tokens_json is not None:
        with spec.tokens_json.open("r", encoding="utf-8") as handle:
            vocab: List[str] = json.load(handle)
        id_to_raw: Dict[int, str] = {i: s for i, s in enumerate(vocab)}
    elif spec.tokenizer_glob is not None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(_resolve(spec.tokenizer_glob)))
        id_to_raw = {}
        for i in range(tok.get_vocab_size()):
            raw = tok.id_to_token(i)
            if raw is not None:
                id_to_raw[i] = raw
    else:
        raise ValueError(f"donor {spec.name!r} has no tokenizer source")

    raw_to_ids: Dict[str, List[int]] = {}
    canon_to_ids: Dict[str, List[int]] = {}
    for i, raw in id_to_raw.items():
        raw_to_ids.setdefault(raw, []).append(i)
        canon_to_ids.setdefault(canonicalize_token(raw), []).append(i)
    return TokenStringMaps(
        name=spec.name,
        id_to_raw=id_to_raw,
        raw_to_ids={k: tuple(v) for k, v in raw_to_ids.items()},
        canon_to_ids={k: tuple(v) for k, v in canon_to_ids.items()},
        size=len(id_to_raw),
    )


def canonical_vocab(maps: TokenStringMaps) -> Dict[str, int]:
    """canonical STRING -> representative id, for the reused Procrustes primitive."""
    return {s: _prefer_id(maps, ids, s) for s, ids in maps.canon_to_ids.items() if s}


def pick_donor_id(
    maps: TokenStringMaps, target_raw: str, target_canon: str
) -> Optional[int]:
    """Choose the donor id carrying the SAME token string as the target token."""
    exact = maps.raw_to_ids.get(target_raw)
    if exact:
        return int(min(exact))
    canon = maps.canon_to_ids.get(target_canon)
    if canon:
        return _prefer_id(maps, canon, target_canon)
    return None


def build_alignment(
    target_id_to_raw: Mapping[int, str],
    maps_by_donor: Mapping[str, TokenStringMaps],
    target_vocab: int = TARGET_VOCAB,
) -> Dict[str, torch.Tensor]:
    """Per donor: LongTensor[target_vocab] of STRING-aligned donor ids (-1 = missing)."""
    aligned: Dict[str, torch.Tensor] = {
        name: torch.full((target_vocab,), -1, dtype=torch.long)
        for name in maps_by_donor
    }
    rows_by_donor: Dict[str, List[int]] = {name: [] for name in maps_by_donor}
    cols_by_donor: Dict[str, List[int]] = {name: [] for name in maps_by_donor}
    for tid in range(target_vocab):
        raw = target_id_to_raw.get(tid)
        if raw is None:
            continue
        canon = canonicalize_token(raw)
        if not canon:
            continue
        for name, maps in maps_by_donor.items():
            did = pick_donor_id(maps, raw, canon)
            if did is not None:
                rows_by_donor[name].append(tid)
                cols_by_donor[name].append(did)
    for name in aligned:
        if rows_by_donor[name]:
            aligned[name][torch.tensor(rows_by_donor[name], dtype=torch.long)] = (
                torch.tensor(cols_by_donor[name], dtype=torch.long)
            )
    return aligned


# --------------------------------------------------------------------------- #
# Decode / rotate / merge primitives                                            #
# --------------------------------------------------------------------------- #
def decode_donor_float(spec: DonorSpec) -> torch.Tensor:
    """Reconstruct float embedding from the donor's (q,gamma) using ITS OWN gamma."""
    sd = torch.load(str(spec.ckpt), map_location="cpu", weights_only=True)
    q = sd["embed_tokens.q"].float()
    g = sd["embed_tokens.gamma"].float().reshape(-1, 1)
    del sd
    if q.shape[0] != g.shape[0]:
        raise ValueError(f"{spec.name}: q/gamma row count mismatch")
    return q * g


def donor_to_host_rotation(
    donor_emb: torch.Tensor,
    donor_vocab: Mapping[str, int],
    host_emb: torch.Tensor,
    host_vocab: Mapping[str, int],
    aligned_donor_but_id: Optional[int],
    aligned_host_but_id: Optional[int],
    target_dim: int = TARGET_DIM,
    chunk_size: int = 8192,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Rotate a FULL donor table into the host frame via the REUSED primitive.

    Returns (rotated_emb[V_donor, target_dim], diagnostics). Diagnostics include
    shared-pair count, orthogonality error, and the cosine of the shared token
    'but' BEFORE (naive center-crop) vs AFTER Procrustes rotation — the #957
    evidence that space alignment materially improved cross-donor agreement.
    """
    # REUSED existing primitive (v23_common.compute_shared_vocab_procrustes).
    p_d2h, n_shared = v23.compute_shared_vocab_procrustes(
        host_vocab=dict(host_vocab),
        donor_vocab=dict(donor_vocab),
        host_emb=host_emb,
        donor_emb=donor_emb,
    )
    if p_d2h.shape[1] != target_dim:
        raise ValueError(
            f"Procrustes output width {p_d2h.shape[1]} != target {target_dim}"
        )

    rotated = torch.empty((donor_emb.shape[0], target_dim), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, donor_emb.shape[0], chunk_size):
            end = min(start + chunk_size, donor_emb.shape[0])
            rotated[start:end] = donor_emb[start:end] @ p_d2h

    diag: Dict[str, object] = {
        "n_shared_tokens": int(n_shared),
        "P_shape": [int(x) for x in p_d2h.shape],
        "orthogonality_error": float(
            torch.norm(p_d2h.T @ p_d2h - torch.eye(target_dim)).item()
        ),
    }
    if (
        aligned_donor_but_id is not None
        and aligned_host_but_id is not None
        and aligned_donor_but_id < donor_emb.shape[0]
    ):
        hv = host_emb[aligned_host_but_id].float()
        dv = donor_emb[aligned_donor_but_id].float()
        src_dim = dv.shape[0]
        crop = 0 if src_dim == target_dim else (src_dim - target_dim) // 2
        naive = dv[crop : crop + target_dim] if crop else dv
        with torch.no_grad():
            before = float(
                F.cosine_similarity(naive.unsqueeze(0), hv.unsqueeze(0)).item()
            )
            after = float(
                F.cosine_similarity(
                    rotated[aligned_donor_but_id].unsqueeze(0), hv.unsqueeze(0)
                ).item()
            )
        diag["cos_but_before_rotate"] = before
        diag["cos_but_after_rotate"] = after
        diag["cos_but_delta"] = after - before
    return rotated, diag


def normalize_unit(v: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    w = v.detach().float()
    return w / w.norm(dim=-1, keepdim=True).clamp_min(eps)


def requantize_absmean(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize a float vector to (q int8, gamma) with the SAME absmean primitive."""
    q, gamma = to_ternary_pair(v, group_dim=-1)
    return q.to(torch.int8), gamma.to(torch.float32)


def merge_normalized_vectors(
    vectors: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure merge core: unit-normalize, average, re-quantize (no donors/IO)."""
    if not vectors:
        raise ValueError("merge_normalized_vectors needs at least one vector")
    merged = torch.stack([normalize_unit(v) for v in vectors], dim=0).mean(dim=0)
    return requantize_absmean(merged)


# --------------------------------------------------------------------------- #
# Full three-donor Procrustes-aligned merge                                    #
# --------------------------------------------------------------------------- #
def merge_donor_embeddings(
    specs: Sequence[DonorSpec] = DONOR_SPECS,
    target_vocab: int = TARGET_VOCAB,
    target_dim: int = TARGET_DIM,
    chunk_size: int = 8192,
    verbose: bool = True,
) -> Dict[str, object]:
    """Merge all donors' embed_tokens into the target after Procrustes rotation."""
    if not specs:
        raise ValueError("no donor specs supplied")
    maps_by_donor: Dict[str, TokenStringMaps] = {
        s.name: load_token_string_maps(s) for s in specs
    }
    if "qwen" not in maps_by_donor:
        raise ValueError("a 'qwen' donor is required as the target-frame host")
    aligned = build_alignment(
        maps_by_donor["qwen"].id_to_raw, maps_by_donor, target_vocab
    )

    host_spec = next(s for s in specs if s.name == "qwen")
    host_emb = decode_donor_float(host_spec)  # (151936, 896) — defines target frame
    host_canon_vocab = canonical_vocab(maps_by_donor["qwen"])

    but_ids = maps_by_donor["qwen"].raw_to_ids.get("but")
    but_tid = int(min(but_ids)) if but_ids else None
    if but_tid is not None and but_tid >= target_vocab:
        but_tid = None  # reduced-vocab test runs cannot index this token
    but_host_id = int(aligned["qwen"][but_tid]) if but_tid is not None else None

    acc = torch.zeros((target_vocab, target_dim), dtype=torch.float32)
    count = torch.zeros((target_vocab,), dtype=torch.int32)
    procrustes: Dict[str, object] = {}

    for spec in specs:
        if spec.name == "qwen":
            donor_emb = host_emb  # already in target frame
            procrustes[spec.name] = {"identity_host_frame": True}
        else:
            raw_emb = decode_donor_float(spec)
            don_but_id = (
                int(aligned[spec.name][but_tid]) if but_tid is not None else None
            )
            donor_emb, diag = donor_to_host_rotation(
                donor_emb=raw_emb,
                donor_vocab=canonical_vocab(maps_by_donor[spec.name]),
                host_emb=host_emb,
                host_vocab=host_canon_vocab,
                aligned_donor_but_id=don_but_id,
                aligned_host_but_id=but_host_id,
                target_dim=target_dim,
                chunk_size=chunk_size,
            )
            procrustes[spec.name] = diag
            del raw_emb
            if verbose:
                print(
                    f"[procrustes] {spec.name}: n_shared={diag['n_shared_tokens']} "
                    f"P={diag['P_shape']} orth_err={diag['orthogonality_error']:.2e} "
                    f"cos('but') {diag.get('cos_but_before_rotate')} -> "
                    f"{diag.get('cos_but_after_rotate')}",
                    flush=True,
                )

        ids = aligned[spec.name]
        valid = ids >= 0
        for start in range(0, target_vocab, chunk_size):
            end = min(start + chunk_size, target_vocab)
            mask = valid[start:end]
            if not bool(mask.any()):
                continue
            sel = ids[start:end][mask]
            w = normalize_unit(donor_emb[sel])  # aligned frame, unit normalized
            block = torch.zeros((end - start, target_dim), dtype=torch.float32)
            block[mask] = w
            acc[start:end] += block
            count[start:end] += mask.to(torch.int32)
        if verbose:
            print(
                f"[merge] {spec.name}: aligned {int(valid.sum())}/{target_vocab} rows "
                f"({100.0 * valid.sum().item() / target_vocab:.2f}%)",
                flush=True,
            )
        if spec.name != "qwen":
            del donor_emb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    merged = acc / count.clamp_min(1).unsqueeze(1)
    merged[count == 0] = 0.0  # ZERO structural default

    q_out = torch.zeros((target_vocab, target_dim), dtype=torch.int8)
    gamma_out = torch.full((target_vocab, 1), EPS, dtype=torch.float32)
    for start in range(0, target_vocab, chunk_size):
        end = min(start + chunk_size, target_vocab)
        qc, gc = requantize_absmean(merged[start:end])
        q_out[start:end] = qc
        gamma_out[start:end] = gc

    float_weight = q_out.to(torch.float32) * gamma_out

    per_donor: Dict[str, Dict[str, object]] = {}
    n_strings = int((aligned["qwen"] >= 0).sum())
    for name, ids in aligned.items():
        matched = int((ids >= 0).sum())
        per_donor[name] = {
            "matched": matched,
            "pct_of_target_vocab": round(100.0 * matched / target_vocab, 3),
            "pct_of_token_strings": round(100.0 * matched / n_strings, 3)
            if n_strings
            else 0.0,
        }
    overlap_counts = {str(k): int((count == k).sum()) for k in range(1, len(specs) + 1)}
    coverage: Dict[str, object] = {
        "target_vocab": target_vocab,
        "target_dim": target_dim,
        "target_token_strings_available": n_strings,
        "target_rows_without_any_token_string": target_vocab - n_strings,
        "per_donor": per_donor,
        "overlap_counts_by_donor_count": overlap_counts,
        "missing_in_all_donors": int((count == 0).sum()),
        "structural_default": "zero row for strings missing in all donors (or no target string)",
        "procrustes": procrustes,
    }
    return {
        "q": q_out,
        "gamma": gamma_out,
        "float_weight": float_weight,
        "aligned_ids": aligned,
        "coverage": coverage,
        "maps": maps_by_donor,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = merge_donor_embeddings(verbose=True)
    print(json.dumps(result["coverage"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
