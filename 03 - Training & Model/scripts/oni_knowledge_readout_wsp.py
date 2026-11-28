#!/usr/bin/env python3
"""Token-space WSP: separate readout adapter, frozen ONI body.

Owner correction 2026-11-27: WSP is a technique, donors supply KNOWLEDGE only,
not geometry or routing. Donor Q/K/V never enter the ONI body. The adapter is
created at the readout only, leaving tied wte/lm_head and all sovereign tensors
byte-identical. No backprop. FWSP O4-style closed-form ridge.

Runtime assets:
  pair1 = exports/quillan_pair_qwen_bitnet_hf  (qwen+bitnet merge)
  pair2 = exports/quillan_pair_llama_falcon_hf (llama+falcon merge)
  4->2->1 = mean of pair teacher probabilities after token mapping.
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from oni_v2_config import owner_config
from quillan_v5_4_oni import QuillanRoninOni

EXPORTS = Path(r"F:\02_QUILLAN\exports")
OUT = EXPORTS / "oni_transfer" / "knowledge_readout"
PAIRS = {
    "p1": EXPORTS / "quillan_pair_qwen_bitnet_hf",
    "p2": EXPORTS / "quillan_pair_llama_falcon_hf",
}
TOK_Q = EXPORTS / "quillan_tokenizer_v54_fourdonor"
PROMPTS = [
    "cat",
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
    "Water boils at",
    "The largest planet is",
    "A dog is a",
    "Paris is the",
    "One plus one equals",
    "The sun is a",
    "Python is a programming",
    "The opposite of hot is",
    "An example of a mammal is",
    "The Earth has one",
    "In mathematics, a prime number is",
    "The color of the sky is",
]
V = 151648
SPECIALS = list(range(151643, 151648))
BASE = EXPORTS / "oni_transfer" / "quillan_mini_oni_transfer.pt"


def log(s: str) -> None:
    print(f"[WSP-KR] {s}", flush=True)


def load_oni():
    blob = torch.load(BASE, map_location="cpu", weights_only=True, mmap=True)
    cfgd = blob["metadata"]["owner_config"]
    from dataclasses import replace

    cfg = replace(
        owner_config("mini"),
        vocab_size=int(cfgd["vocab_size"]),
        max_seq_len=int(cfgd["max_seq_len"]),
        hidden_dim=int(cfgd["hidden_dim"]),
        ffn_dim=int(cfgd["ffn_dim"]),
        n_layer=int(cfgd["n_layer"]),
        n_head=int(cfgd["n_head"]),
        num_experts=int(cfgd["num_experts"]),
    )
    model = QuillanRoninOni(cfg).eval()
    model.load_state_dict(blob["state_dict"], strict=False)
    return model, blob["state_dict"]


def hash_sd(sd) -> str:
    keys = sorted(k for k in sd if k not in ("lm_head.weight",))
    return hashlib.sha256(
        "".join(keys + [str(tuple(sd[k].shape)) for k in keys]).encode()
    ).hexdigest()[:16]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("loading Quillan tokenizer")
    tok_q = Tokenizer.from_file(str(TOK_Q / "tokenizer.json"))
    log("loading frozen ONI model")
    model, sd = load_oni()
    before = hash_sd(sd)
    log(f"sovereign key manifest {before}")

    teachers = {}
    for name, path in PAIRS.items():
        log(f"loading teacher pair {name} from {path}")
        tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        mdl = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype="auto", low_cpu_mem_usage=True
        ).eval()
        probs = []
        for p in PROMPTS:
            ids = tok.encode(p).ids
            if not ids:
                continue
            with torch.inference_mode():
                out = mdl(torch.tensor([ids]))
            lg = out.logits[0, -1]
            q = F.softmax(lg / max(1.0, float(lg.abs().mean())), dim=-1)
            # map donor token IDs -> Quillan token IDs by exact string
            row = torch.zeros(V)
            for did in torch.topk(q, 4096).indices.tolist():
                s = tok.decode([did], skip_special_tokens=True)
                if not s:
                    continue
                qids = tok_q.encode(s).ids
                if not qids:
                    continue
                mass = float(q[did]) / len(qids)
                for qid in qids:
                    if 0 <= qid < 151643:
                        row[qid] += mass
            row[151643:] = 0.0
            z = float(row.sum())
            if z > 0:
                row /= z
            probs.append(row)
        teachers[name] = torch.stack(probs)
        del mdl
        log(f"teacher {name}: {tuple(teachers[name].shape)}")

    n = min(t.shape[0] for t in teachers.values())
    q1 = teachers["p1"][:n]
    q2 = teachers["p2"][:n]
    target = (0.5 * q1 + 0.5 * q2).clamp_min(1e-12)
    target = target / target.sum(dim=1, keepdim=True)
    y = torch.log(target) - torch.log(target).mean(dim=1, keepdim=True)
    log(f"4->2->1 teacher target {tuple(y.shape)}")

    # Frozen upstream features: real final-LN hidden captured by hook.
    captured = {}
    hook = model.ln_f.register_forward_hook(
        lambda m, inp, out: captured.__setitem__("h", out.detach())
    )
    X = []
    prompts_used = []
    for p in PROMPTS:
        ids = tok_q.encode(p).ids
        if not ids:
            continue
        captured.clear()
        with torch.inference_mode():
            model(
                torch.tensor([ids]),
                return_aux=False,
                compute_aux=False,
                deliberation=False,
            )
        X.append(captured["h"][0, -1].float())
        prompts_used.append(p)
    hook.remove()
    X = torch.stack(X)[:n]
    log(f"captured real final-LN hidden from {len(prompts_used)} prompts")
    log(f"frozen features {tuple(X.shape)}")

    # FWSP O4 closed-form ridge, scale-relative lambda.
    G = X.T @ X
    lam = max(1e-6, 1e-2 * float(G.diag().mean()))
    G += torch.eye(G.shape[0]) * lam
    Pmat = X.T @ y
    try:
        A = torch.linalg.solve(G, Pmat)
    except torch.linalg.LinAlgError:
        A = torch.linalg.lstsq(G, Pmat).solution
    A = A[:64]  # rank cap
    delta_read = A.T  # (V, 64)
    delta_read[SPECIALS] = 0.0
    log(f"readout delta {tuple(delta_read.shape)} norm={float(delta_read.norm()):.4f}")

    after = hash_sd(sd)
    if before != after:
        raise RuntimeError(f"sovereign manifest changed {before} -> {after}")

    out_path = OUT / "quillan_mini_knowledge_readout.pt"
    torch.save(
        {
            "delta_read": delta_read,
            "base": str(BASE),
            "method": "token_space_WSP_O4_separate_readout_adapter",
            "no_backprop": True,
            "sovereign_manifest": before,
            "specials_zeroed": SPECIALS,
            "limitations": [
                "Diagnostic adapter; NOT a promoted model checkpoint.",
                "Next-token distribution alignment is not proof of reasoning.",
                "No ARC-AGI/HLE claim without held-out scoring.",
            ],
        },
        out_path,
    )
    log(f"saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
