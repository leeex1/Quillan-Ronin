#!/usr/bin/env python3
"""Token-space WSP adapter: map ONI hidden -> compiled donor-token knowledge.

Owner correction 2026-11-27: 'cat' is knowledge in token/embedding/output space,
not donor Q/K/V geometry. This script never loads a donor transformer and never
edits ONI tensors. It fits a SEPARATE readout adapter under torch.no_grad():
  logits' = lm_head(fused) + adapter(fused)
using closed-form ridge on paired (ONI hidden, donor-token distribution) rows.

Knowledge source: exports/quillan_embedding_compiled.safetensors (237566 rows,
union-slot space), mapped to locked 151648 tokenizer IDs via the already-built
slot_for_token_id table in oni_donor_alignment_rebuilt.pt.
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

EXPORTS = Path(r"F:\02_QUILLAN\exports")
OUT = EXPORTS / "oni_transfer" / "knowledge_readout"
BASE = EXPORTS / "oni_transfer" / "quillan_mini_oni_transfer.pt"
EMB = EXPORTS / "quillan_embedding_compiled.safetensors"
ALIGN = EXPORTS / "oni_transfer" / "oni_donor_alignment_rebuilt.pt"
TOK_DIR = EXPORTS / "quillan_tokenizer_v54_fourdonor"
V = 151648
SPECIALS = list(range(151643, 151648))
CHUNK = 64
N_TOKENS = 4096


def log(s: str) -> None:
    print(f"[TOKEN-KNOWLEDGE] {s}", flush=True)


def load_oni():
    blob = torch.load(BASE, map_location="cpu", weights_only=True, mmap=True)
    oc = blob["metadata"]["owner_config"]
    cfg = replace(
        owner_config("mini"),
        vocab_size=int(oc["vocab_size"]),
        max_seq_len=int(oc["max_seq_len"]),
        hidden_dim=int(oc["hidden_dim"]),
        ffn_dim=int(oc["ffn_dim"]),
        n_layer=int(oc["n_layer"]),
        n_head=int(oc["n_head"]),
        num_experts=int(oc["num_experts"]),
    )
    model = QuillanRoninOni(cfg).eval()
    model.load_state_dict(blob["state_dict"], strict=False)
    return model


def select_token_ids(tok: Tokenizer) -> list[int]:
    prompts = [
        "cat",
        "dog",
        "The capital of France is",
        "2 + 2 =",
        "Paris",
        "Hello",
        "water",
        "planet",
        "mammal",
        "prime number",
        "sky",
        "programming",
        "hot",
        "cold",
        "Earth",
        "sun",
        "mathematics",
        "France",
        "Germany",
        "river",
        "ocean",
        "tree",
        "animal",
        "science",
    ]
    ids = []
    for p in prompts:
        ids.extend(tok.encode(p).ids)
    ids = [i for i in ids if 0 <= i < 151643]
    spread = torch.linspace(0, 151642, N_TOKENS).long().tolist()
    ids.extend(spread)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:N_TOKENS]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("loading tokenizer and alignment")
    tok = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
    align = torch.load(ALIGN, map_location="cpu", weights_only=True, mmap=True)
    slots = align["slot_for_token_id"]
    token_ids = select_token_ids(tok)
    log(f"selected {len(token_ids)} tokens")

    log("loading compiled donor embedding rows")
    with safe_open(str(EMB), framework="pt", device="cpu") as f:
        sl = f.get_slice("model.embed_tokens.weight")
        all_slots = slots[:V].tolist()
        E_all = sl[all_slots].float()  # (V, 2048)

    log("loading frozen ONI model")
    model = load_oni()
    H = model.cfg.hidden_dim

    # Capture real final-LN hidden for each selected token.
    captured = {}
    hook = model.ln_f.register_forward_hook(
        lambda m, inp, out: captured.__setitem__("h", out.detach())
    )
    hidden = []
    with torch.inference_mode():
        for start in range(0, len(token_ids), CHUNK):
            ids = token_ids[start : start + CHUNK]
            captured.clear()
            model(
                torch.tensor([[i] for i in ids]),
                return_aux=False,
                compute_aux=False,
                deliberation=False,
            )
            hidden.append(captured["h"][:, 0, :].float())
    hook.remove()
    X = torch.cat(hidden, dim=0)
    log(f"captured hidden {tuple(X.shape)}")

    # Fit adapter: X -> centered log-softmax of donor embedding similarity.
    G = X.T @ X
    Bmat = torch.zeros((H, V), dtype=torch.float32)
    denom = 2048.0**0.5
    for start in range(0, len(token_ids), CHUNK):
        end = min(start + CHUNK, len(token_ids))
        rows = E_all[token_ids[start:end]]
        sim = rows @ E_all.T / denom
        y = F.log_softmax(sim.float(), dim=-1)
        y = y - y.mean(dim=1, keepdim=True)
        Bmat += X[start:end].T @ y
    lam = max(1e-5, 1e-2 * float(G.diag().mean()))
    G.add_(torch.eye(H) * lam)
    try:
        W = torch.linalg.solve(G, Bmat)  # (H, V)
    except torch.linalg.LinAlgError:
        W = torch.linalg.lstsq(G, Bmat).solution
    delta = W.T.contiguous()  # (V, H)
    delta[SPECIALS] = 0.0
    log(f"adapter {tuple(delta.shape)} norm={float(delta.norm()):.4f}")

    out_path = OUT / "quillan_mini_token_knowledge_adapter.pt"
    torch.save(
        {
            "adapter_weight": delta,
            "base": str(BASE),
            "knowledge_source": str(EMB),
            "method": "token_space_WSP_closed_form_ridge_separate_readout",
            "no_backprop": True,
            "vocab": V,
            "special_rows_zeroed": SPECIALS,
            "token_count": len(token_ids),
            "limitations": [
                "Diagnostic adapter; not a promoted model checkpoint.",
                "Token association transfer, not sentence-level reasoning.",
                "No ARC-AGI/HLE claim without held-out scoring.",
            ],
        },
        out_path,
    )
    log(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
