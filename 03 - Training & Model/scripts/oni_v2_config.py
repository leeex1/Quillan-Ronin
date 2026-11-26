#!/usr/bin/env python3
"""Quillan ONI v2 config — owner-locked spec 2026-11-26.

Locks: tokenizer/vocab 151,648; context 2,097,152; tri-stack loop
(recirculation + GRT + GRU); 2/34/34 sovereign graph; ternary substrate.
No donor backbone, no hifi/lofi body.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from quillan_v5_4_oni import QuillanOniConfig

OWNER_VOCAB = 151648  # 151643 Qwen base + 5 sovereign specials
OWNER_CTX = 2097152  # 2M locked deployment context


def owner_config(model: str = "main") -> QuillanOniConfig:
    """Build ONI config with owner-locked vocab/context and tri-stack flags."""
    if model == "mini":
        # Owner-locked 2026-11-26: 896/1792/L6, n_head=14, head_dim=64
        # -> 650,666,808 params (0.6507B, within 0.5-0.7B target).
        base = replace(
            QuillanOniConfig.mini_6l(),
            hidden_dim=896,
            ffn_dim=1792,
            n_layer=6,
            n_head=14,
            head_dim=64,
        )
    elif model == "main":
        # Owner-locked 2026-11-26: 1152/2304/L12, n_head=18, head_dim=64
        # -> 1,076,774,730 params (1.0768B, within 1.0-1.2B target).
        base = replace(
            QuillanOniConfig.main_12l(),
            hidden_dim=1152,
            ffn_dim=2304,
            n_layer=12,
            n_head=18,
            head_dim=64,
        )
    else:
        raise ValueError(f"unknown model {model}")
    return replace(
        base,
        vocab_size=OWNER_VOCAB,
        max_seq_len=OWNER_CTX,
        use_recurrent_diffusion=True,
        use_grt=True,
        use_bitnet_optimizer=True,
        use_swarm_prover_bitnet=True,
        num_experts=34,
        # preserve ONI sovereign graph: dense_pull, swarm, prism, governor
        router_mode="dense_pull",
    )


if __name__ == "__main__":
    for m in ("mini", "main"):
        c = owner_config(m)
        print(
            m,
            "vocab",
            c.vocab_size,
            "ctx",
            c.max_seq_len,
            "hidden",
            c.hidden_dim,
            "ffn",
            c.ffn_dim,
            "layers",
            c.n_layer,
            "experts",
            c.num_experts,
            "grt",
            c.use_grt,
            "recur",
            c.use_recurrent_diffusion,
        )
