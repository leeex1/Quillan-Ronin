import sys, time, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
import evo_moe

torch.set_num_threads(4)
H = 512
cfg = QuillanOniConfig(hidden_dim=H, n_layer=2, n_head=8, head_dim=64, ffn_dim=1024,
                       num_experts=34, expert_rank=8, swarm_rank=8, use_evo_moe=True)
m = QuillanRoninOni(cfg).eval()
x = torch.randint(0, 50000, (4, 128))


def bench(n=12):
    with torch.no_grad():
        for _ in range(3):
            m(x, path_override=1)
        t = time.perf_counter()
        for _ in range(n):
            m(x, path_override=1)
        return (time.perf_counter() - t) / n * 1000


with torch.no_grad():
    ev = m.h[0].moe.evo_moe
    for e in ev.experts:
        for p in e.parameters():
            p.data.add_(torch.randn_like(p) * 0.02)

a = bench()
ev.vectorized = True
b = bench()
print(f"EvoMoE grouped (34 experts, ranks 8/24/48)   : {b:8.2f} ms/fwd")
print(f"EvoMoE sequential loop (reference)         : {a:8.2f} ms/fwd")
print(f"speedup                                    : {a / b:8.2f}x")

cfg2 = QuillanOniConfig(hidden_dim=H, n_layer=2, n_head=8, head_dim=64, ffn_dim=1024,
                        num_experts=34, expert_rank=8, swarm_rank=8,
                        use_evo_moe=False, router_mode="dense_pull")
m2 = QuillanRoninOni(cfg2).eval()


def bench2(n=8):
    with torch.no_grad():
        for _ in range(2):
            m2(x, path_override=1)
        t = time.perf_counter()
        for _ in range(n):
            m2(x, path_override=1)
        return (time.perf_counter() - t) / n * 1000


for blk in m2.h:
    blk.moe._batched_council = None
    blk.moe.vectorized_experts = False
    blk.attn.prism.vectorized = False
c = bench2()
for blk in m2.h:
    blk.attn.prism.vectorized = True
    blk.moe.vectorized_experts = True
    blk.moe._batched_council = __import__("quillan_v5_4_oni").BatchedCouncilCompute(blk.moe.experts)
d = bench2()
print(f"\nCouncil dense_pull batched (34 experts)     : {d:8.2f} ms/fwd")
print(f"Council dense_pull sequential (reference)   : {c:8.2f} ms/fwd")
print(f"speedup                                    : {c / d:8.2f}x")