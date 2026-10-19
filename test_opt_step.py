import sys
import os
import faulthandler
faulthandler.enable()
import traceback
from pathlib import Path
import torch

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass

sys.path.insert(0, r"c:\02_QUILLAN\03 - Training & Model\scripts")
sys.path.insert(0, r"c:\02_QUILLAN\03 - Training & Model")

from quillan_train_pipeline import QuillanTrainingOrchestrator

print("[1] Initializing Orchestrator...")
orch = QuillanTrainingOrchestrator(n_layer=6, device_str="cpu")

print("[2] Loading checkpoint...")
orch.load_checkpoint(Path(r"E:\quillan_checkpoints_oni\quillan_6l_latest.pt"))

print("[3] Creating Fused Optimizer...")
from quillan_fused_optimizer import create_quillan_fused_optimizer
opt = create_quillan_fused_optimizer(orch.model, lr=2.5e-5, lr_muon=0.015, weight_decay=0.01, ccrl_limit=4.0, ns_steps=3, use_sophia=False)

print("[4] Generating dummy batch...")
x = torch.randint(0, 1000, (1, 256), dtype=torch.long)
y = torch.randint(0, 1000, (1, 256), dtype=torch.long)

print("[5] Forward pass...")
out = orch.model(x, labels=y, return_aux=True)
logits, ce_loss, aux = out
total_loss = ce_loss + 0.01 * orch.model.total_aux_loss(aux)
print(f"    Loss: {total_loss.item():.4f}")

print("[6] Backward pass...")
total_loss.backward()
print("    Backward pass completed successfully.")

print("[7] Testing clip_grad_norm_...")
torch.nn.utils.clip_grad_norm_(orch.model.parameters(), 1.0)
print("    clip_grad_norm_ passed.")

print("[8] Running full opt.step()...")
opt.step()
print("    Full opt.step() completed successfully!")

print("[9] Cleaning step 1 graph & zeroing grad with set_to_none=True...")
del out, logits, ce_loss, aux, total_loss
import gc
gc.collect()
opt.zero_grad(set_to_none=True)
print("    Step 1 graph freed, memory reclaimed.")

print("[10] Forward pass 2...")
try:
    out2 = orch.model(x, labels=y, return_aux=True)
    print("    Forward pass 2 done.")
    loss2 = out2[1] + 0.01 * orch.model.total_aux_loss(out2[2])
    print(f"    Loss 2: {loss2.item():.4f}")
except Exception as e:
    print(f"    FORWARD PASS 2 FAILED: {e}")
    traceback.print_exc()

print("[11] Backward pass 2...")
try:
    loss2.backward()
    print("    Backward pass 2 done.")
except Exception as e:
    print(f"    BACKWARD PASS 2 FAILED: {e}")
    traceback.print_exc()

print("[12] Clip grad norm 2...")
try:
    torch.nn.utils.clip_grad_norm_(orch.model.parameters(), 1.0)
    print("    Clip grad norm 2 done.")
except Exception as e:
    print(f"    CLIP GRAD NORM 2 FAILED: {e}")
    traceback.print_exc()

print("[13] Running full opt.step() 2...")
try:
    opt.step()
    print("    Full opt.step() 2 completed successfully! ALL SYSTEMS GREEN.")
except Exception as e:
    print(f"    OPT.STEP 2 FAILED: {e}")
    traceback.print_exc()
