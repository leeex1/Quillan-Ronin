#!/usr/bin/env python3
"""
download_frontier_datasets.py
Downloads verified frontier reasoning datasets from Hugging Face Hub:
1. Crownelius/GPT-5.6-Sol-Luna-Terra-Traces (~285 MB)
2. r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation (~789 MB)
3. MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x (~2,421 MB)

Total payload: ~3.49 GB.
Monitors C: drive free space and guarantees a minimum 20 GB safety buffer.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model\training_data\frontier_datasets")
MIN_FREE_SPACE_GB = 20.0

DATASETS = [
    {
        "repo_id": "Crownelius/GPT-5.6-Sol-Luna-Terra-Traces",
        "subfolder_name": "gpt_5_6_sol_luna_terra",
        "desc": "GPT-5.6 Sol-Luna-Terra Traces (285 MB Parquet)",
        "ignore_patterns": []
    },
    {
        "repo_id": "r0b0tlab/qwen3.8-max-glm5.2-kimi-k3-distillation",
        "subfolder_name": "qwen3_8_glm5_2_kimi_k3",
        "desc": "Qwen3.8-Max / GLM-5.2 / Kimi-K3 Distillation & Unsloth recipes (789 MB Parquet)",
        "ignore_patterns": []
    },
    {
        "repo_id": "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
        "subfolder_name": "fable_5_1_max",
        "desc": "Fable-5.1 Max-Reasoning Filtered 10000x (2.42 GB JSONL)",
        "ignore_patterns": []
    }
]

def get_c_drive_free_gb() -> float:
    total, used, free = shutil.disk_usage("C:")
    return free / (1024 ** 3)

def check_disk_safety():
    free_gb = get_c_drive_free_gb()
    print(f"[DISK CHECK] Free Space on C:: {free_gb:.2f} GB")
    if free_gb < MIN_FREE_SPACE_GB:
        raise RuntimeError(
            f"ABORT: Free disk space on C: ({free_gb:.2f} GB) is below safety margin ({MIN_FREE_SPACE_GB} GB)!"
        )

def get_dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)

def main():
    print("=" * 70)
    print("QUILLAN-RONIN FRONTIER DATASET DOWNLOADER")
    print(f"Destination: {BASE_DIR}")
    print("=" * 70)

    check_disk_safety()
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for i, ds in enumerate(DATASETS, 1):
        target_dir = BASE_DIR / ds["subfolder_name"]
        print(f"\n[{i}/{len(DATASETS)}] Starting: {ds['repo_id']}")
        print(f"      Description: {ds['desc']}")
        print(f"      Target Dir:  {target_dir}")

        check_disk_safety()

        ds_start = time.time()
        try:
            snapshot_download(
                repo_id=ds["repo_id"],
                repo_type="dataset",
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=ds["ignore_patterns"] if ds["ignore_patterns"] else None,
                resume_download=True
            )
            ds_elapsed = time.time() - ds_start
            size_mb = get_dir_size_mb(target_dir)
            print(f"[SUCCESS] {ds['repo_id']} downloaded in {ds_elapsed:.1f}s. Stored size: {size_mb:.2f} MB")
        except Exception as e:
            print(f"[ERROR] Failed to download {ds['repo_id']}: {e}", file=sys.stderr)
            raise e

    total_elapsed = time.time() - start_time
    total_size_gb = get_dir_size_mb(BASE_DIR) / 1024
    final_free_gb = get_c_drive_free_gb()

    print("\n" + "=" * 70)
    print("ALL FRONTIER DATASETS DOWNLOADED SUCCESSFULLY")
    print(f"Total Download Time: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} mins)")
    print(f"Total Downloaded Size: {total_size_gb:.2f} GB")
    print(f"Remaining Free Space on C:: {final_free_gb:.2f} GB")
    print("=" * 70)

if __name__ == "__main__":
    main()
