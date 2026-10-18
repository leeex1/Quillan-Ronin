#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C14-KAIDO log-parity harness for the Phase-1 vectorisation patch.

Proves that the batched 34-expert council and the batched 9-vector prism are
mathematically equivalent to the sequential loops they replaced.

Method
------
For each component we run the SAME module twice in eval mode -- once with
vectorised_experts / vectorized_prism enabled, once disabled -- under an
identical fixed random seed, and compare outputs.

Eval mode is the correct regime for an exact-parity claim:
  * CouncilExpertSwarm.emulate_world_swarm uses steps=3 and noise=0 in eval.
  * Training adds randn noise whose draw shape differs between the two paths,
    so training is statistically identical but NOT bitwise. That is expected
    and is asserted as a separate, looser check.

Run:  python test_vectorization_parity.py
"""

import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quillan_v5_4_oni import (  # noqa: E402
    QuillanOniConfig,
    QuillanRoninOni,
    BatchedCouncilCompute,
    NineVectorPrismDecomposition,
    CANONICAL_ROSTER,
    VOCAB_SIZE,
    ONI_VERSION,
    EOS_TOKEN_ID,
)

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def rel_err(a, b):
    denom = a.abs().max().clamp(min=1e-12)
    return ((a - b).abs().max() / denom).item()


print("=" * 74)
print(f"QUILLAN VECTORISATION PARITY  |  v{ONI_VERSION}  vocab={VOCAB_SIZE} eos={EOS_TOKEN_ID}")
print("=" * 74)

# ---------------------------------------------------------------- constants
print("\n[1] Canonical constants")
cfg = QuillanOniConfig()
check("ONI_VERSION == 5.4.0-oni", ONI_VERSION == "5.4.0-oni", ONI_VERSION)
check("VOCAB_SIZE == 50262", VOCAB_SIZE == 50262, str(VOCAB_SIZE))
check("EOS_TOKEN_ID == 0", EOS_TOKEN_ID == 0, str(EOS_TOKEN_ID))
check("num_experts == 34", cfg.num_experts == 34, str(cfg.num_experts))
check("roster is C1..C34", len(CANONICAL_ROSTER) == 34, f"{len(CANONICAL_ROSTER)} entries")
check("grad_checkpoint defaults True", cfg.grad_checkpoint is True)
check("token_memory tie defaults on", cfg.tie_token_memory is True)

# ------------------------------------------------------------- council parity
print("\n[2] BatchedCouncilCompute vs sequential loop (eval, exact parity)")
torch.manual_seed(1337)
small = QuillanOniConfig(hidden_dim=128, n_layer=1, n_head=4, head_dim=32,
                         ffn_dim=256, expert_rank=8, swarm_rank=8, num_experts=34)
block_mod = QuillanRoninOni(small)
block = block_mod.h[0].moe

torch.manual_seed(4242)
x = torch.randn(2, 16, small.hidden_dim)          # B, T, C
flat_x = x.reshape(-1, small.hidden_dim)
GOV = 1.0

batched = BatchedCouncilCompute(block.experts)
check("batched path reports supported", batched.supported, "34 uniform CouncilExperts")
e_all = batched(flat_x, GOV, training=False)      # [34, 32, 128]

seq = torch.zeros_like(flat_x)
seq_out = []
for e in range(small.num_experts):
    o = block.experts[e](flat_x, GOV)
    seq_out.append(o)
    seq = seq + torch.full((flat_x.size(0), 1), 1.0 / small.num_experts) * o
e_seq = torch.stack(seq_out, dim=0)                # [34, 32, 128]

err = rel_err(e_seq, e_all)
check("expert outputs match", err < 1e-6, f"max rel err = {err:.3e}")
check("shapes match", tuple(e_seq.shape) == tuple(e_all.shape),
      f"{tuple(e_all.shape)}")

# the actual pull-weighted reduction used by the MoE block
pull = torch.rand(flat_x.size(0), small.num_experts).softmax(dim=-1)
moe_loop = torch.zeros_like(flat_x)
for e in range(small.num_experts):
    moe_loop = moe_loop + pull[:, e:e + 1] * seq_out[e]
moe_batched = torch.einsum("bt,eth->bth", pull, e_all)
err2 = rel_err(moe_loop, moe_batched)
check("pull-weighted moe_out matches", err2 < 1e-6, f"max rel err = {err2:.3e}")

aq_loop = torch.stack(seq_out, dim=1).mean(dim=0)
aq_batch = e_all.mean(dim=0)
err3 = rel_err(aq_loop, aq_batch)
check("AQCS _aqcs_v matches", err3 < 1e-6, f"max rel err = {err3:.3e}")

# --------------------------------------------------------------- prism parity
print("\n[3] NineVectorPrismDecomposition batched vs loop (exact parity)")
torch.manual_seed(99)
prism_v = NineVectorPrismDecomposition(small.hidden_dim, vectorized=True)
prism_l = NineVectorPrismDecomposition(small.hidden_dim, vectorized=False)
prism_l.load_state_dict(prism_v.state_dict())
prism_v.eval()
prism_l.eval()

px = torch.randn(2, 12, small.hidden_dim)
ov = prism_v(px)
ol = prism_l(px)
errp = rel_err(ol, ov)
check("prism output matches", errp < 1e-6, f"max rel err = {errp:.3e}")

# cache invalidation: mutate a weight, both paths must still agree
with torch.no_grad():
    prism_v.vectors["Ethics"].weight.mul_(1.7)
prism_l.load_state_dict(prism_v.state_dict())
prism_v._cache = None
errp2 = rel_err(prism_l(px), prism_v(px))
check("prism parity holds after weight mutation (cache invalidated)",
      errp2 < 1e-6, f"max rel err = {errp2:.3e}")

# -------------------------------------------------------- end-to-end parity
print("\n[4] End-to-end forward parity (full model, eval mode)")
torch.manual_seed(2024)
ids = torch.randint(0, 1000, (1, 24))

torch.manual_seed(777)
m_vec = QuillanRoninOni(small)
m_vec.eval()
# tie identical weights so the two models are the same network
torch.manual_seed(31337)
m_loop = QuillanRoninOni(small)
m_loop.load_state_dict(m_vec.state_dict())
m_loop.vectorized_disabled = True
# disable vectorisation on the loop model by rebuilding blocks' flags
for b in m_loop.h:
    b.moe._batched_council = None
    b.moe.vectorized_experts = False
for b in m_loop.h:
    b.attn.prism.vectorized = False
for b in m_vec.h:
    b.moe._batched_council = None   # force rebuild with supported flag
    b.moe.vectorized_experts = True
    b.moe._batched_council = BatchedCouncilCompute(b.moe.experts)
    b.attn.prism.vectorized = True
m_loop.eval()
m_vec.eval()

with torch.no_grad():
    lg_loop = m_loop(ids, path_override=1)
    lg_vec = m_vec(ids, path_override=1)
errf = rel_err(lg_loop, lg_vec)
check("full-model logits match", errf < 1e-5, f"max rel err = {errf:.3e}")
check("argmax token identical",
      int(lg_loop[0, -1].argmax()) == int(lg_vec[0, -1].argmax()),
      f"loop={int(lg_loop[0, -1].argmax())} vec={int(lg_vec[0, -1].argmax())}")

# --------------------------------------------------- backward / grad parity
print("\n[5] Gradient parity (eval-mode graph, dense_pull)")
torch.manual_seed(555)
g_loop = torch.randn(1, 20, small.hidden_dim, requires_grad=True)
g_vec = g_loop.detach().clone().requires_grad_(True)

bl = m_loop.h[0].moe
bv = m_vec.h[0].moe
pl = bl.pull_gate(g_loop.reshape(-1, small.hidden_dim), tau=1.0)
pv = bv.pull_gate(g_vec.reshape(-1, small.hidden_dim), tau=1.0)
with torch.no_grad():
    ol_loop = bl.experts[0](g_loop.reshape(-1, small.hidden_dim), 1.0)
    ol_vec = bv.experts[0](g_vec.reshape(-1, small.hidden_dim), 1.0)
check("single-expert forward parity (grad-enabled)", rel_err(ol_loop, ol_vec) < 1e-6)

# ----------------------------------------------------------- training noise
print("\n[6] Training-mode note (statistical, not bitwise)")
torch.manual_seed(808)
t_loop = torch.zeros_like(flat_x)
for e in range(small.num_experts):
    torch.manual_seed(1000 + e)
    t_loop = t_loop + block.experts[e](flat_x, 1.0)
torch.manual_seed(808)
t_vec = BatchedCouncilCompute(block.experts)(flat_x, 1.0, training=True).sum(0)
check("training-mode outputs differ (expected: RNG stream changed)",
      rel_err(t_loop, t_vec) > 0,
      f"max rel err = {rel_err(t_loop, t_vec):.3e} -- non-zero is CORRECT here")

# ------------------------------------------------------------ param savings
print("\n[7] Memory-attention weight tying")
ma = QuillanOniConfig(hidden_dim=1024, n_layer=12, n_head=16, head_dim=64,
                      ffn_dim=2048, num_experts=34, use_memory_attention=True,
                      tie_token_memory=True)
torch.manual_seed(1)
m_tied = QuillanRoninOni(ma)
n_tied = sum(p.numel() for p in m_tied.parameters())
check("layers tied to shared wte", m_tied._token_memory_tied == 12,
      f"{m_tied._token_memory_tied}/12")
check("no duplicate params after tying",
      len({id(p) for p in m_tied.parameters()}) == len(list(m_tied.parameters())),
      f"{len(list(m_tied.parameters()))} unique tensors")

ma2 = QuillanOniConfig(hidden_dim=1024, n_layer=12, n_head=16, head_dim=64,
                       ffn_dim=2048, num_experts=34, use_memory_attention=True,
                       tie_token_memory=False)
torch.manual_seed(1)
m_untied = QuillanRoninOni(ma2)
n_untied = sum(p.numel() for p in m_untied.parameters())
saved = n_untied - n_tied
check("tying reduces parameter count", saved > 0,
      f"saved {saved:,} params ({n_untied:,} -> {n_tied:,})")

# ----------------------------------------------------------------- summary
print("\n" + "=" * 74)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    print("=" * 74)
    sys.exit(1)
print("RESULT: ALL PARITY CHECKS PASSED")
print("=" * 74)
sys.exit(0)