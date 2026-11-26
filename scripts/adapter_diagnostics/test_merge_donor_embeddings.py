#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
#956/#957 — REAL TEST: Procrustes-aligned, token-string embedding merge. NO FAKE PASS.

Asserts:
  (a) token-string alignment: 'but'/'and'/'the'/'is' resolve by STRING in all
      three donors despite DIFFERENT ids; raw-index lookup grabs a wrong word.
  (b) PROCRUSTES APPLIED: REUSED v23_common.compute_shared_vocab_procrustes gives
      >=512 shared pairs, near-orthogonal P, and cosine(Qwen 'but', P@donor 'but')
      MATERIALLY HIGHER than the un-rotated baseline (measured live).
  (c) OVERLAP merged/shared + NON-OVERLAP preserved (pure core + aligned recompute).
  (d) output q in {-1,0,+1} int8, gamma finite/positive, deterministic two-run.
  (e) honest cross-donor coverage percentages.
  (f) adapter wiring: override reaches BOTH txt_emb and txt_dec.

Run: python -u scripts\adapter_diagnostics\test_merge_donor_embeddings.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

DIAG = Path(__file__).resolve().parent
ROOT = DIAG.parent.parent
MODEL_DIR = ROOT / "03 - Training & Model"
for p in (str(DIAG), str(ROOT / "scripts"), str(MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from merge_donor_embeddings import (  # noqa: E402
    DONOR_SPECS,
    build_alignment,
    canonical_vocab,
    canonicalize_token,
    decode_donor_float,
    donor_to_host_rotation,
    load_token_string_maps,
    merge_donor_embeddings,
    merge_normalized_vectors,
    normalize_unit,
    pick_donor_id,
    requantize_absmean,
)

PASSED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED
    assert ok, f"FAIL {name} :: {detail}"
    PASSED += 1
    print(f"PASS {name}" + (f" :: {detail}" if detail else ""), flush=True)


def _token_id(maps, word: str) -> int:
    ids = maps.raw_to_ids.get(word)
    assert ids, f"word {word!r} missing from donor {maps.name}"
    return int(min(ids))


# --------------------------------------------------------------------------- #
# (c) Pure merge core                                                           #
# --------------------------------------------------------------------------- #
def test_merge_core() -> None:
    a = torch.tensor([3.0, 0.0, 0.0, 0.0])
    b = torch.tensor([6.0, 0.0, 0.0, 0.0])
    q, g = merge_normalized_vectors([a, b])
    check(
        "core_overlap_normalized_no_dominance",
        int(q[0].item()) == 1 and bool((g > 0).all()),
        f"q={q.tolist()} gamma={g.reshape(-1).tolist()}",
    )

    q_u, _ = merge_normalized_vectors([b])
    check(
        "core_nonoverlap_preserved",
        int(q_u[0].item()) == 1 and abs(float(normalize_unit(b)[0]) - 1.0) < 1e-6,
        f"single donor preserved: q={q_u.tolist()}",
    )

    q_c, _ = merge_normalized_vectors([a, -a])
    check(
        "core_exact_conflict_cancels_to_zero",
        bool((q_c == 0).all()),
        f"q={q_c.tolist()}",
    )

    all_q = torch.cat([q, q_u, q_c]).reshape(-1)
    check(
        "core_q_ternary_int8",
        q.dtype == torch.int8
        and bool(torch.isin(all_q, torch.tensor([-1, 0, 1])).all()),
        f"dtype={q.dtype} unique={torch.unique(all_q).tolist()}",
    )
    check(
        "core_deterministic_pure",
        torch.equal(merge_normalized_vectors([a, b])[0], q),
        "repeated pure merge bitwise identical",
    )


# --------------------------------------------------------------------------- #
# (a) Real token-string alignment                                               #
# --------------------------------------------------------------------------- #
def test_real_alignment(maps_by_donor) -> dict:
    ids_by_word: dict[str, dict[str, int]] = {}
    for word in ("but", "and", "the", "is"):
        ids: dict[str, int] = {}
        for name, maps in maps_by_donor.items():
            did = pick_donor_id(maps, word, canonicalize_token(word))
            assert did is not None, f"{word!r} missing in {name}"
            raw = maps.id_to_raw[did]
            check(
                f"align_{word}_{name}_string_match",
                canonicalize_token(raw) == word,
                f"id={did} raw={raw!r}",
            )
            ids[name] = did
        ids_by_word[word] = ids
    for word in ("but", "and", "the"):
        d = ids_by_word[word]
        check(
            f"align_{word}_ids_differ",
            d["qwen"] != d["llama"] and d["qwen"] != d["dragon"],
            str(d),
        )
    qwen_but = ids_by_word["but"]["qwen"]
    llama_same_idx = maps_by_donor["llama"].id_to_raw.get(qwen_but, "")
    dragon_same_idx = maps_by_donor["dragon"].id_to_raw.get(qwen_but, "")
    check(
        "align_index_lookup_would_be_wrong",
        canonicalize_token(llama_same_idx) != "but"
        and canonicalize_token(dragon_same_idx) != "but",
        f"qwen 'but' id={qwen_but}; llama@{qwen_but}={llama_same_idx!r}; "
        f"dragon@{qwen_but}={dragon_same_idx!r}",
    )
    return ids_by_word


# --------------------------------------------------------------------------- #
# (d/e) Full merge: contract, coverage, determinism                             #
# --------------------------------------------------------------------------- #
def test_full_merge() -> dict:
    result = merge_donor_embeddings(verbose=True)
    q, gamma, fw = result["q"], result["gamma"], result["float_weight"]
    cov = result["coverage"]
    check("full_q_shape", tuple(q.shape) == (151936, 896), str(tuple(q.shape)))
    check(
        "full_q_int8_ternary",
        q.dtype == torch.int8
        and bool(torch.isin(q.reshape(-1)[::997], torch.tensor([-1, 0, 1])).all()),
        f"dtype={q.dtype}",
    )
    check(
        "full_gamma_shape_finite_positive",
        tuple(gamma.shape) == (151936, 1)
        and bool(torch.isfinite(gamma).all())
        and bool((gamma > 0).all()),
        f"min={gamma.min():.6g} max={gamma.max():.6g}",
    )
    check(
        "full_float_weight_shape",
        tuple(fw.shape) == (151936, 896),
        str(tuple(fw.shape)),
    )

    pd = cov["per_donor"]
    check(
        "coverage_llama_positive",
        pd["llama"]["matched"] > 0,
        f"llama={pd['llama']['matched']} ({pd['llama']['pct_of_token_strings']}% of strings)",
    )
    check(
        "coverage_dragon_positive",
        pd["dragon"]["matched"] > 0,
        f"dragon={pd['dragon']['matched']} ({pd['dragon']['pct_of_token_strings']}% of strings)",
    )
    oc = cov["overlap_counts_by_donor_count"]
    check(
        "overlap_and_unique_exist",
        oc.get("1", 0) > 0 and sum(int(v) for k, v in oc.items() if k != "1") > 0,
        f"overlap_counts={oc}",
    )
    print("COVERAGE " + __import__("json").dumps(cov, ensure_ascii=False), flush=True)
    return result


# --------------------------------------------------------------------------- #
# (b/c) PROCRUSTES APPLIED + aligned-frame numeric provenance                   #
# --------------------------------------------------------------------------- #
def test_procrustes_and_provenance(result: dict, maps_by_donor, ids_by_word) -> None:
    q, gamma = result["q"], result["gamma"]
    aligned = result["aligned_ids"]
    procrustes_meta = result["coverage"]["procrustes"]
    but_tid = ids_by_word["but"]["qwen"]

    # Check the diagnostics recorded in the real merge.
    for name in ("llama", "dragon"):
        d = procrustes_meta[name]
        check(
            f"procrustes_{name}_shared_pairs",
            d["n_shared_tokens"] >= 512,
            f"n_shared={d['n_shared_tokens']}",
        )
        check(
            f"procrustes_{name}_orthogonal",
            d["orthogonality_error"] < 1e-3,
            f"orth_err={d['orthogonality_error']:.2e}",
        )
        check(
            f"procrustes_{name}_cos_materially_higher",
            d["cos_but_after_rotate"] > d["cos_but_before_rotate"] + 0.10,
            f"cos('but') {d['cos_but_before_rotate']:.4f} -> "
            f"{d['cos_but_after_rotate']:.4f} (delta {d['cos_but_delta']:.4f})",
        )

    # Independent recompute: load host + donors, recompute the SAME rotations with
    # the REUSED primitive, then rebuild the merged 'but' row from the ALIGNED
    # frame and require bitwise equality -> proves string+space aligned merge.
    host_emb = decode_donor_float(DONOR_SPECS[0])
    host_vocab = canonical_vocab(maps_by_donor["qwen"])
    host_but = int(aligned["qwen"][but_tid])
    aligned_vectors = [normalize_unit(host_emb[host_but])]
    for spec in DONOR_SPECS[1:]:
        raw = decode_donor_float(spec)
        don_but = int(aligned[spec.name][but_tid])
        rotated, diag = donor_to_host_rotation(
            donor_emb=raw,
            donor_vocab=canonical_vocab(maps_by_donor[spec.name]),
            host_emb=host_emb,
            host_vocab=host_vocab,
            aligned_donor_but_id=don_but,
            aligned_host_but_id=host_but,
        )
        # Independently confirmed number (not just trusting recorded metadata).
        check(
            f"procrustes_{spec.name}_independent_cos",
            diag["cos_but_after_rotate"] > diag["cos_but_before_rotate"] + 0.10,
            f"{diag['cos_but_before_rotate']:.4f} -> {diag['cos_but_after_rotate']:.4f}",
        )
        aligned_vectors.append(normalize_unit(rotated[don_but]))
        del raw, rotated
        gc.collect()

    expected_q, expected_g = requantize_absmean(torch.stack(aligned_vectors).mean(0))
    check(
        "provenance_but_from_three_ALIGNED_ids",
        int(aligned["qwen"][but_tid]) == ids_by_word["but"]["qwen"]
        and int(aligned["llama"][but_tid]) == ids_by_word["but"]["llama"]
        and int(aligned["dragon"][but_tid]) == ids_by_word["but"]["dragon"],
        f"aligned ids={[int(aligned[n][but_tid]) for n in aligned]} "
        f"expected={ids_by_word['but']}",
    )
    check(
        "provenance_but_row_numeric_exact",
        torch.equal(expected_q, q[but_tid])
        and torch.allclose(expected_g, gamma[but_tid], atol=1e-6),
        "merged 'but' row == mean(normalized Qwen/P@Llama/P@Dragon 'but')",
    )
    del host_emb
    gc.collect()


# --------------------------------------------------------------------------- #
# (d) Deterministic two-run of the full merge                                   #
# --------------------------------------------------------------------------- #
def test_deterministic_two_run() -> None:
    r1 = merge_donor_embeddings(target_vocab=1024, verbose=False)
    q1, g1 = r1["q"].clone(), r1["gamma"].clone()
    del r1
    gc.collect()
    r2 = merge_donor_embeddings(target_vocab=1024, verbose=False)
    check(
        "deterministic_two_run_bitwise",
        torch.equal(q1, r2["q"]) and torch.equal(g1, r2["gamma"]),
        f"q_equal={torch.equal(q1, r2['q'])} gamma_equal={torch.equal(g1, r2['gamma'])}",
    )
    del r2, q1, g1
    gc.collect()


# --------------------------------------------------------------------------- #
# (f) Adapter wiring                                                            #
# --------------------------------------------------------------------------- #
def test_adapter_wiring() -> None:
    from test_custom_arch_transfer import (
        EXPERTS,
        FFN,
        HIDDEN,
        LAYERS,
        VOCAB,
        _build_donor,
        _build_target,
    )
    from quillan_weight_adapter import QuillanCrossModelWeightAdapter, SourceDims

    model = _build_target()
    override = torch.full((VOCAB, HIDDEN), 0.123, dtype=torch.float32)
    donor = _build_donor()  # complete single-donor fixture (norms/layers/mlp)
    dims = SourceDims(
        hidden=HIDDEN, ffn=FFN, vocab=VOCAB, num_layers=LAYERS, arch="llama"
    )
    adapter = QuillanCrossModelWeightAdapter(num_experts=EXPERTS, ternarize=False)
    sd = adapter.adapt_state_dict(
        donor,
        source_arch="llama",
        target_model=model,
        source_dims=dims,
        embedding_override=override,
    )
    check(
        "wiring_txt_emb_uses_override",
        torch.allclose(sd["ingestion.txt_emb.weight"], override),
    )
    check(
        "wiring_txt_dec_uses_override", torch.allclose(sd["txt_dec.weight"], override)
    )


def main() -> None:
    print("=" * 72, flush=True)
    print(
        "#956/#957 PROCRUSTES-ALIGNED THREE-DONOR EMBEDDING MERGE — REAL TEST",
        flush=True,
    )
    print("=" * 72, flush=True)
    test_merge_core()
    maps_by_donor = {s.name: load_token_string_maps(s) for s in DONOR_SPECS}
    for name, m in maps_by_donor.items():
        print(f"TOKENIZER {name}: {m.size} entries", flush=True)
    ids_by_word = test_real_alignment(maps_by_donor)
    result = test_full_merge()
    test_procrustes_and_provenance(result, maps_by_donor, ids_by_word)
    del result
    gc.collect()
    test_deterministic_two_run()
    test_adapter_wiring()
    print("=" * 72, flush=True)
    print(f"MERGE_DONOR_EMBEDDINGS_TEST PASS | checks={PASSED}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
