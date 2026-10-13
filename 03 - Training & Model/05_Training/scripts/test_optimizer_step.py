import sys
import torch

sys.path.insert(0, r"c:\02_QUILLAN\scripts")
from quillan_v10_unrolled_sovereign import QuillanRoninSovereign, QuillanArchConfig
from quillan_muonk2_optimizer import create_quillan_muonk2_optimizer

torch.set_num_threads(2)
cfg = QuillanArchConfig()
model = QuillanRoninSovereign(cfg)
opt = create_quillan_muonk2_optimizer(model)

inp = torch.randint(0, 50257, (1, 384), dtype=torch.long)
lbl = torch.randint(0, 50257, (1, 384), dtype=torch.long)

opt.zero_grad()
logits, loss = model(inp, labels=lbl)
loss.backward()
opt.step()

print(f"[SUCCESS] In-place MuonK2 + AdamW Optimizer step completed! Loss: {loss.item():.4f}", flush=True)
