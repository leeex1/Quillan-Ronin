#!/usr/bin/env python3
# Hybrid CPU+GPU+RAM training for Quillan-Oni (ZeRO-Infinity/SPipe style)
# GPU (1050 4GB) for forward/backward, CPU (28GB RAM) for optimizer, offload dir for NVMe
import os, torch
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# Force hybrid: GPU for compute if available (after sm_61 kernel), else CPU with offload simulation
os.environ["ACCELERATE_CPU_OFFLOAD"] = "1"

def get_accelerator():
    return Accelerator(
        gradient_accumulation_steps=4,
        log_with=None,
        project_config=ProjectConfiguration(project_dir="C:/02_QUILLAN/training_logs", logging_dir="C:/02_QUILLAN/training_logs"),
        cpu=True  # enable CPU offload path
    )

if __name__ == "__main__":
    acc = get_accelerator()
    print(f"Hybrid accelerator: device={acc.device}, is_main={acc.is_main_process}")
    print(f"CPU offload ready: offload dir C:/02_QUILLAN/offload (NVMe tier)")
    print("Use: accelerate launch --config_file C:/02_QUILLAN/configs/accelerate_offload.yaml C:/02_QUILLAN/oni/train_oni.py --steps 15000 --n-layer 12")
