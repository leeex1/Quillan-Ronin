#!/usr/bin/env python3
import torch, os, sys
from pathlib import Path

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", torch.cuda.get_device_properties(0).total_memory / (1024**3))
else:
    print("CPU Threads:", torch.get_num_threads())

pt_file = Path(r"C:\02_QUILLAN\training_data\quillan_corpus_CLEAN_V7.pt")
if pt_file.exists():
    print(f"\nChecking {pt_file.name} (Size: {pt_file.stat().st_size / (1024**2):.1f} MB)...")
    data = torch.load(pt_file, map_location="cpu", weights_only=False)
    print("Type:", type(data))
    if isinstance(data, dict):
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k}: len={len(v) if hasattr(v, '__len__') else type(v)}")
    elif hasattr(data, 'shape'):
        print(f"  Tensor shape: {list(data.shape)}, dtype={data.dtype}")
    elif isinstance(data, list):
        print(f"  List length: {len(data)}")
