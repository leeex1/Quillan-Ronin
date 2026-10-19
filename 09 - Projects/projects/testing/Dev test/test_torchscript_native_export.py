#!/usr/bin/env python3
"""
Diagnostic test to compile Quillan-Ronin v5.3.1 to TorchScript C++ and benchmark native CPU speed.
"""
import sys, time, torch
from pathlib import Path

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_thinking_reasoning_master.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
model.eval()

# Dummy input
dummy_input = torch.randint(0, 50257, (1, 64), dtype=torch.long)

print("[*] Benchmarking PyTorch Standard Forward Pass...", flush=True)
t0 = time.time()
for _ in range(20):
    with torch.no_grad():
        out = model(dummy_input)
dt_py = (time.time() - t0) / 20
print(f"[+] PyTorch Latency: {dt_py*1000:.1f}ms per token ({1.0/dt_py:.1f} tok/s)\n", flush=True)

print("[*] Tracing model to TorchScript C++ Engine...", flush=True)
try:
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
    
    # Save traced model
    out_dir = REPO_ROOT / "exports"
    out_dir.mkdir(exist_ok=True)
    ts_path = out_dir / "quillan_ronin_v531_native.pt"
    traced_model.save(str(ts_path))
    print(f"[+] Saved Native TorchScript C++ Model to: {ts_path.name} ({ts_path.stat().st_size/(1024**2):.1f} MB)\n", flush=True)

    # Benchmark TorchScript
    t0 = time.time()
    for _ in range(20):
        with torch.no_grad():
            out_ts = traced_model(dummy_input)
    dt_ts = (time.time() - t0) / 20
    print(f"[+] TorchScript C++ Latency: {dt_ts*1000:.1f}ms per token ({1.0/dt_ts:.1f} tok/s)", flush=True)
    print(f"[+] Speedup: {dt_py/dt_ts:.2f}x native acceleration!", flush=True)

except Exception as e:
    print(f"[-] TorchScript Tracing Notice: {e}")
