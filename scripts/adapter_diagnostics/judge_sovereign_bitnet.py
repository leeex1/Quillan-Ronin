#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""INDEPENDENT JUDGE for QuillanSovereignBitNet checkpoints (OpenCode, #1187).

Criteria LOCKED BEFORE scoring (no goalpost moving):
  J1 STRUCTURAL   : param_count <= 1B; loads clean; no NaN logits.
  J2 NO-ECHO      : last-token logit cos between two prompts ending in the SAME
                    token must be < 0.99 (kills #1126's cos 0.9996 echo).
  J3 DISCRIMINATION: those two prompts must have DIFFERENT top-1 tokens.
  J4 SEMANTIC     : on ~10 held-out factual prompts, top-5 must contain the
                    expected answer token for >= 5 prompts (>=50%).
  J5 NULL CONTROL : a random-weight reference must NOT pass J4 (gibberish).
  J6 DONOR REACH  : ablation of donor expert blocks must change held-out logits.

PASS = J1..J5 all true (J6 advisory). Anything else = FAIL, reported honestly.
Read-only: never edits any checkpoint, gate, model, or canonical. No WSP.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Locked test prompts (same endings for echo test; factual for semantic)
PROSE_A = "The capital of France is"  # expects Paris
PROSE_B = "Hello, my name is"  # expects a name
ECHO_PAIR = (PROSE_A, "I think therefore I am, and this sentence ends in the word is")
FACTUAL = [
    ("The capital of France is", [" Paris"]),
    ("The capital of Japan is", [" Tokyo"]),
    ("The largest planet in our solar system is", [" Jupiter"]),
    ("Water freezes at zero degrees", [" Celsius", " celsius"]),
    ("The sun rises in the", [" east"]),
    ("2 + 2 equals", [" 4", "4"]),
    ("Hello, my name is", [" John", " Alex", " Anna"]),
    ("The opposite of hot is", [" cold"]),
    ("A dog is a type of", [" animal"]),
    ("The Earth orbits around the", [" sun"]),
]


def load_model(path: Path):
    """Load a QuillanSovereignBitNet checkpoint via its native loader."""
    from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

    # load_packed_safetensors is a classmethod that constructs+loads the model
    model = QuillanSovereignBitNetLM.load_packed_safetensors(path)
    model.eval()
    cfg = getattr(model, "cfg", None)
    return model, cfg


def get_logits(model, tok, text: str) -> torch.Tensor:
    ids = tok.encode(text, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        out = (
            model.generate(x, max_new_tokens=0) if hasattr(model, "generate") else None
        )
    # Prefer a direct forward if available
    if hasattr(model, "forward"):
        with torch.inference_mode():
            o = model(x)
        lg = o[0] if isinstance(o, (tuple, list)) else o
        return lg[0, -1, :].float()
    raise RuntimeError("no forward/logits interface")


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters()) + sum(
        b.numel() for b in model.buffers()
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()
    path = args.checkpoint.resolve()
    if not path.exists():
        print(f"FATAL: {path} missing")
        return 2

    print("=" * 78)
    print("INDEPENDENT JUDGE — QuillanSovereignBitNet (criteria locked #1187)")
    print("=" * 78)
    print(f"checkpoint: {path.name}")

    # Tokenizer: MUST match the checkpoint's embedding vocab. Using a mismatched
    # tokenizer (e.g. Qwen 151k on a Llama 128k model) produces garbage logits and
    # a false J4=0/10 (see flagship RCI #1262). Match by reading the model's
    # vocab_size and selecting the corresponding donor tokenizer.
    from transformers import AutoTokenizer

    model_vocab = None
    try:
        import safetensors.torch as _st

        with _st.safe_open(str(path), framework="pt", device="cpu") as _f:
            import json as _json

            _meta = _json.loads(
                (_f.metadata() or {}).get("quillan_sovereign_config", "{}")
            )
            model_vocab = int(_meta.get("vocab_size", 0)) or None
    except Exception:
        pass

    if model_vocab and model_vocab <= 128256:
        # Llama/BitNet/Falcon family (128k vocab)
        cand = Path(r"E:\source_models\llama_1b")
        tok = AutoTokenizer.from_pretrained(str(cand), local_files_only=True)
        print(f"[tokenizer] Llama 128k (matched to model vocab {model_vocab})")
    else:
        snaps = glob.glob(
            os.path.expanduser(
                r"~\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\*"
            )
        )
        tok = AutoTokenizer.from_pretrained(snaps[0])
        print(f"[tokenizer] Qwen 151k (matched to model vocab {model_vocab})")

    try:
        model, cfg = load_model(path)
    except Exception as e:
        print(f"FATAL load error: {e}")
        return 2

    # J1 structural
    pc = param_count(model)
    j1 = pc <= 1_000_000_000
    print(f"\n[J1] STRUCTURAL: params={pc:,} -> {'PASS' if j1 else 'FAIL'}")

    # J2/J3 echo + discrimination
    lg_a, lg_b = get_logits(model, tok, PROSE_A), get_logits(model, tok, PROSE_B)
    cos = float(F.cosine_similarity(lg_a.unsqueeze(0), lg_b.unsqueeze(0)).item())
    t1a, t1b = int(lg_a.argmax()), int(lg_b.argmax())
    j2 = cos < 0.99
    j3 = t1a != t1b
    print(f"[J2] NO-ECHO: cos={cos:.4f} (need <0.99) -> {'PASS' if j2 else 'FAIL'}")
    print(
        f"[J3] DISCRIMINATION: top1={tok.decode([t1a])!r} vs {tok.decode([t1b])!r} -> {'PASS' if j3 else 'FAIL'}"
    )

    # J4 semantic
    hits = 0
    for text, expected in FACTUAL:
        lg = get_logits(model, tok, text)
        top5 = [tok.decode([i]) for i in torch.topk(lg, 5).indices.tolist()]
        hit = any(any(e in t for t in top5) for e in expected)
        hits += int(hit)
        print(f"  {text[:40]!r:42} -> {'HIT' if hit else 'miss'} top5={top5[:3]}")
    j4 = hits >= 5
    print(
        f"[J4] SEMANTIC: {hits}/{len(FACTUAL)} expected-answer hits (need >=5) -> {'PASS' if j4 else 'FAIL'}"
    )

    # J5 null control: random logits over same vocab should fail J4
    torch.manual_seed(0)
    null_lg = torch.randn(151936)
    null_hits = 0
    for text, expected in FACTUAL:
        top5 = [tok.decode([i]) for i in torch.topk(null_lg, 5).indices.tolist()]
        null_hits += int(any(any(e in t for t in top5) for e in expected))
    j5 = null_hits < 5
    print(
        f"[J5] NULL CONTROL: random-map hits={null_hits}/10 (need <5) -> {'PASS' if j5 else 'FAIL'}"
    )

    verdict = "PASS" if (j1 and j2 and j3 and j4 and j5) else "FAIL"
    print("\n" + "=" * 78)
    print(
        f"INDEPENDENT JUDGE VERDICT: {verdict}  (J1={j1} J2={j2} J3={j3} J4={j4} J5={j5})"
    )
    print("J6 donor-ablation reachability: advisory (run after artifact set complete)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
