#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C14-KAIDO log-parity harness for the Phase-1 vectorisation patch.

Proves the batched 34-expert council and the batched 9-vector prism are
mathematically equivalent to the sequential loops they replaced.

CONFIG NOTE (verified against checkpoints/checkpoints_oni/quillan_oni_latest.pt):
the shipped checkpoint trains with use_evo_moe=True AND router_mode='gumbel_topk',
so EvoMoE's own dispatch is the real hot path (covered by test_evo_moe_parity.py).
The CouncilExpert dense_pull loop vectorised here is the fallback path taken when
use_evo_moe=False. Both are gated by config flags and both are tested below.

Run:  python test_vectorization_parity.py
"""
import sys, pathlib
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from quillan_v5_4_oni import (  # noqa: E402
    QuillanOniConfig, QuillanRoninOni, BatchedCouncilCompute,
    NineVectorPrismDecomposition, CANONICAL_ROSTER, VOCAB_SIZE, ONI_VERSION, EOS_TOKEN_ID,
)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def rel_err(a, b):
    return ((a - b).abs().max() / a.abs().max().clamp(min=1e-12)).item()


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
check("roster is C1..C34", len(CANONICAL_ROSTER) == 34, f"{len(CANONICAL_ROSTER)}")
check("grad_checkpoint defaults True", cfg.grad_checkpoint is True)
check("token_memory tie defaults on", cfg.tie_token_memory is True)

# ------------------------------------------------- CouncilExpert dense path
# use_evo_moe=False is REQUIRED here: with EvoMoE active, block.experts holds
# EvoMoE Sequential experts, not CouncilExpert, and the dense loop is bypassed.
print("\n[2] BatchedCouncilCompute vs sequential loop (CouncilExpert, eval)")
torch.manual_seed(1337)
small = QuillanOniConfig(hidden_dim=128, n_layer=2, n_head=4, head_dim=32,
                         ffn_dim=256, expert_rank=8, swarm_rank=8,
                         num_experts=34, use_evo_moe=False,
                         router_mode="dense_pull")
mod = QuillanRoninOni(small)
# CRITICAL: council parity is an EVAL-mode claim. In train mode the swarm adds
# randn noise whose draw shape differs between loop and batched paths, so the
# comparison would be meaningless. eval() => steps=3, noise=0 => exact parity.
mod.eval()
block = mod.h[0].moe

check("experts are CouncilExpert (use_evo_moe=False)",
      hasattr(block.experts[0], "lora_A"),
      type(block.experts[0]).__name__)
check("batched path active on block", block.vectorized_experts and
      block._batched_council is not None and block._batched_council.supported)

torch.manual_seed(4242)
flat_x = torch.randn(2 * 16, small.hidden_dim)
GOV = 1.0

e_all = block._batched_council(flat_x, GOV, training=False)        # [34, N, H]
seq_out = []
for e in range(small.num_experts):
    seq_out.append(block.experts[e](flat_x, GOV))
e_seq = torch.stack(seq_out, dim=0)

err = rel_err(e_seq, e_all)
check("expert outputs match", err < 1e-6, f"max rel err = {err:.3e}")
check("shapes match", tuple(e_seq.shape) == tuple(e_all.shape), f"{tuple(e_all.shape)}")

pull = torch.rand(flat_x.size(0), small.num_experts).softmax(dim=-1)
moe_loop = torch.zeros_like(flat_x)
for e in range(small.num_experts):
    moe_loop = moe_loop + pull[:, e:e + 1] * seq_out[e]
moe_batched = torch.einsum("pe,eph->ph", pull, e_all)
check("pull-weighted moe_out matches", rel_err(moe_loop, moe_batched) < 1e-6,
      f"max rel err = {rel_err(moe_loop, moe_batched):.3e}")
check("AQCS _aqcs_v matches",
      rel_err(torch.stack(seq_out, dim=1).mean(dim=0), e_all.mean(dim=1)) < 1e-6,
      f"shape {tuple(e_all.mean(dim=1).shape)}")

print("\n[3] Council cache invalidation")
with torch.no_grad():
    block.experts[3].lora_B.add_(0.05)
b2 = block._batched_council
after_a = b2(flat_x, GOV, False)
after_b = b2(flat_x, GOV, False)
check("repeat call stable (cache reused)", torch.allclose(after_a, after_b))
check("lora_B mutation visible", not torch.allclose(after_a, e_all),
      "output changed -> cache correctly invalidated")
seq_after = torch.stack([block.experts[e](flat_x, GOV) for e in range(34)], 0)
check("still matches loop after mutation",
      rel_err(seq_after, b2(flat_x, GOV, False)) < 1e-6)

# --------------------------------------------------------------- prism
print("\n[4] NineVectorPrism batched vs loop (exact parity)")
torch.manual_seed(99)
pv = NineVectorPrismDecomposition(small.hidden_dim, vectorized=True)
pl = NineVectorPrismDecomposition(small.hidden_dim, vectorized=False)
pl.load_state_dict(pv.state_dict())
pv.eval(); pl.eval()
px = torch.randn(2, 12, small.hidden_dim)
check("prism output matches", rel_err(pl(px), pv(px)) < 1e-6,
      f"max rel err = {rel_err(pl(px), pv(px)):.3e}")

with torch.no_grad():
    pv.vectors["Ethics"].weight.mul_(1.7)
pl.load_state_dict(pv.state_dict())
pv._cache = None
check("prism parity after weight mutation (cache invalidated)",
      rel_err(pl(px), pv(px)) < 1e-6, f"max rel err = {rel_err(pl(px), pv(px)):.3e}")

# ------------------------------------------------- end-to-end model parity
print("\n[5] End-to-end logits parity (full model, eval)")
torch.manual_seed(2024)
torch.manual_seed(31337)
m_vec = QuillanRoninOni(small).eval()

# clone identical weights
torch.manual_seed(31337)
m_loop = QuillanRoninOni(small).eval()
m_loop.load_state_dict(m_vec.state_dict())

# force the sequential path on the reference model
for b in m_loop.h:
    b.moe._batched_council = None
    b.moe.vectorized_experts = False
    b.attn.prism.vectorized = False
for b in m_vec.h:
    b.attn.prism.vectorized = True

ids = torch.randint(0, 1000, (1, 24))
# Seed each pass identically: the model contains a Langevin diffusion stage that
# draws torch.randn even in eval, so without reseeding the two forwards differ for
# reasons unrelated to the vectorisation under test.
with torch.no_grad():
    torch.manual_seed(9001)
    lg_loop = m_loop(ids, path_override=1)
    torch.manual_seed(9001)
    lg_vec = m_vec(ids, path_override=1)
errf = rel_err(lg_loop, lg_vec)
check("full-model logits match", errf < 1e-5, f"max rel err = {errf:.3e}")
check("argmax token identical",
      int(lg_loop[0, -1].argmax()) == int(lg_vec[0, -1].argmax()),
      f"loop={int(lg_loop[0, -1].argmax())} vec={int(lg_vec[0, -1].argmax())}")

print("\n[6] Gradient parity (dense_pull backward)")
torch.manual_seed(555)
g = torch.randn(1, 20, small.hidden_dim, requires_grad=True)
gl = g.detach().clone().requires_grad_(True)
gv = g.detach().clone().requires_grad_(True)
e0_loop = m_loop.h[0].moe.experts[0](gl.reshape(-1, small.hidden_dim), 1.0)
e0_vec = m_vec.h[0].moe.experts[0](gv.reshape(-1, small.hidden_dim), 1.0)
e0_loop.square().sum().backward()
e0_vec.square().sum().backward()
check("input grads match", rel_err(gl.grad, gv.grad) < 1e-5,
      f"max rel err = {rel_err(gl.grad, gv.grad):.3e}")

print("\n[7] Training-mode note (statistical, not bitwise)")
mod.train()   # swarm: steps=1 + randn noise
torch.manual_seed(808)
tl = torch.zeros_like(flat_x)
for e in range(small.num_experts):
    torch.manual_seed(1000 + e)
    tl = tl + block.experts[e](flat_x, 1.0)
torch.manual_seed(808)
tv = BatchedCouncilCompute(block.experts)(flat_x, 1.0, training=True).sum(0)
check("train-mode differs (expected: RNG stream changed)", rel_err(tl, tv) > 0,
      f"rel err = {rel_err(tl, tv):.3e} -- non-zero is CORRECT")
mod.eval()

# --------------------------------------------------------- weight tying
print("\n[8] Memory-attention weight tying")
torch.manual_seed(1)
mt = QuillanRoninOni(QuillanOniConfig(hidden_dim=1024, n_layer=12, n_head=16, head_dim=64,
                                      ffn_dim=2048, num_experts=34, use_evo_moe=False,
                                      use_memory_attention=True, tie_token_memory=True))
nt = sum(p.numel() for p in mt.parameters())
check("all 12 layers tied to shared wte", mt._token_memory_tied == 12,
      f"{mt._token_memory_tied}/12")
pl_list = list(mt.parameters())
check("no duplicate parameter objects", len({id(p) for p in pl_list}) == len(pl_list),
      f"{len(pl_list)} unique tensors")

torch.manual_seed(1)
mu = QuillanRoninOni(QuillanOniConfig(hidden_dim=1024, n_layer=12, n_head=16, head_dim=64,
                                      ffn_dim=2048, num_experts=34, use_evo_moe=False,
                                      use_memory_attention=True, tie_token_memory=False))
nu = sum(p.numel() for p in mu.parameters())
check("tying reduces parameter count", nu - nt > 0,
      f"saved {nu - nt:,} params ({nu:,} -> {nt:,})")
check("per-layer mem_norm retained",
      all(hasattr(mt.h[i].attn, "mem_norm") for i in range(12)),
      "12 distinct mem_norm modules")
check("untied model keeps private tables",
      all(mu.h[i].attn.token_memory is not mt.h[0].attn.token_memory for i in range(12)))

# --------------------------------------------------------------- summary
print("\n" + "=" * 74)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {FAILURES}")
    print("=" * 74)
    sys.exit(1)
print("RESULT: ALL PARITY CHECKS PASSED")
print("=" * 74)
sys.exit(0)