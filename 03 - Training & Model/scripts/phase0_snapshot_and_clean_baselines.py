"""
=============================================================================
PHASE 0: CHECKPOINT SNAPSHOT LEDGER (SHA256) & CLEAN BASELINE COMPILATION
=============================================================================
Implements Hard Rules #1 & #2 from the Refined Agent Prompt:
1. Computes and logs SHA256 hashes, byte sizes, and paths for all untouched
   original sources, released GGUFs, and experimental WSP checkpoints into
   `C:\02_QUILLAN\exports\checkpoint_sha256_ledger.json`.
2. Creates untouched backup snapshots of the post-WSP-v5.0 `.pt` files before
   restoring their clean pre-WSP v1.1 weights via `compile_wsp_tripartite.py`.
3. Compiles Clean Unmodified Q4_K_M GGUFs directly from the untouched original
   base weights (with zero weight perturbations) so we can run a strict
   apples-to-apples Phase 0 Baseline Evaluation comparing:
     - Clean Untouched Base (Q4_K_M)
     - Released Checkpoint (F16 / Q8_0 / Q4_K_M)
     - Experimental WSP v5.0 (Q4_K_M)
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = r"C:\02_QUILLAN"
EXPORT_ROOT = os.path.join(ROOT_DIR, "exports")
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "checkpoints_oni", "sft_runs")
SNAP_BACKUP_DIR = os.path.join(EXPORT_ROOT, "snapshots_pre_phase0")

FLAGSHIP_SNAP = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c"
MINI_SNAP = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"

BLOB_FLAGSHIP_REL = r"C:\Users\Admin\.ollama\models\blobs\sha256-0fcaf0693b4af3e8a516a9bad1d1f64b6ed07f255bb8556e4a38964a9046fcea"
BLOB_MINI_REL = r"C:\Users\Admin\.ollama\models\blobs\sha256-1923271f8d4d5101e91b90d67fdd56da544a6258c614a4d1ce9cd098696062f9"
BLOB_DRAGON_REL = r"C:\Users\Admin\.ollama\models\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"

QUANTIZE_BIN = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-quantize.exe"
CONVERT_SCRIPT = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\convert_hf_to_gguf.py"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_clean_base_ggufs() -> Dict[str, str]:
    """
    Compiles Clean Unmodified Q4_K_M GGUF files from the untouched originals
    (0 weight modifications) so we can isolate pure quantization + runtime fix
    from WSP v5.0 weight perturbations.
    """
    print("\n>>> [Step 2/3] Compiling Clean Unmodified Base Q4_K_M GGUFs (Zero Weight Edits)...")
    clean_paths = {}

    # 1. Clean Mini 0.5B Q4_K_M directly from untouched BLOB_MINI_REL (F16 GGUF)
    clean_mini_q4 = os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-clean-base-q4_k_m.gguf")
    if not os.path.exists(clean_mini_q4):
        print("  Quantizing untouched Mini 0.5B F16 blob -> quillan-mini-0.5b-clean-base-q4_k_m.gguf...")
        subprocess.run([QUANTIZE_BIN, BLOB_MINI_REL, clean_mini_q4, "Q4_K_M"], check=True)
    clean_paths["MODEL_B_CLEAN_Q4"] = clean_mini_q4

    # 2. Clean Dragon 1B Q4_K_M directly from untouched BLOB_DRAGON_REL (Q8_0 GGUF)
    clean_dragon_q4 = os.path.join(EXPORT_ROOT, "quillan-dragon-1b-clean-base-q4_k_m.gguf")
    if not os.path.exists(clean_dragon_q4):
        print("  Quantizing untouched Dragon 1B Q8_0 blob -> quillan-dragon-1b-clean-base-q4_k_m.gguf...")
        subprocess.run([QUANTIZE_BIN, "--allow-requantize", BLOB_DRAGON_REL, clean_dragon_q4, "Q4_K_M"], check=True)
    clean_paths["MODEL_C_CLEAN_Q4"] = clean_dragon_q4

    # 3. Clean Flagship 1.2B Q4_K_M directly from untouched FLAGSHIP_SNAP (Llama-3.2-1B-Instruct)
    clean_flag_q4 = os.path.join(EXPORT_ROOT, "quillan-1b-clean-base-q4_k_m.gguf")
    if not os.path.exists(clean_flag_q4):
        print("  Converting untouched Flagship 1.2B HF snapshot -> quillan-1b-clean-base-q4_k_m.gguf...")
        tmp_f16 = os.path.join(EXPORT_ROOT, "quillan-1b-clean-base-tmp-f16.gguf")
        subprocess.run([sys.executable, CONVERT_SCRIPT, FLAGSHIP_SNAP, "--outfile", tmp_f16, "--outtype", "f16"], check=True)
        subprocess.run([QUANTIZE_BIN, tmp_f16, clean_flag_q4, "Q4_K_M"], check=True)
        if os.path.exists(tmp_f16):
            os.remove(tmp_f16)
    clean_paths["MODEL_A_CLEAN_Q4"] = clean_flag_q4

    return clean_paths


def main():
    t0 = time.time()
    print("=" * 96)
    print("PHASE 0: CHECKPOINT SNAPSHOT LEDGER (SHA256) & CLEAN BASELINE COMPILATION")
    print("=" * 96)

    os.makedirs(SNAP_BACKUP_DIR, exist_ok=True)

    # 1. Backup current WSP-v5.0 .pt and GGUF files before any revert/ablation
    print("\n>>> [Step 1/3] Backing up Post-WSP-v5.0 Checkpoints to snapshots_pre_phase0/...")
    for fname in [
        "quillan_1b_tripartite.pt",
        "quillan_mini_0.6b_tripartite.pt",
    ]:
        src = os.path.join(CKPT_DIR, fname)
        dst = os.path.join(SNAP_BACKUP_DIR, fname + ".wsp_v50_backup")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  [BACKUP] Saved {dst}")

    for gname in [
        "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf",
        "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf",
        "quillan-dragon-1b-wsp-v3-q4_k_m.gguf",
    ]:
        src = os.path.join(EXPORT_ROOT, gname)
        dst = os.path.join(SNAP_BACKUP_DIR, gname.replace("-wsp-v3-", "-wsp-v50-backup-"))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  [BACKUP] Saved {dst}")

    # 2. Build Clean Unmodified Base Q4_K_M GGUFs
    clean_paths = build_clean_base_ggufs()

    # 3. Compute SHA256 Ledger across all originals, released checkpoints, clean Q4_K_M, and WSP v5.0
    print("\n>>> [Step 3/3] Computing SHA256 Checksum Ledger across All Model Files...")
    tracked_files = [
        ("MODEL_A_ORIGINAL_SAFETENSORS", os.path.join(FLAGSHIP_SNAP, "model.safetensors")),
        ("MODEL_A_RELEASED_Q4_K_M", os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-q4_k_m.gguf")),
        ("MODEL_A_RELEASED_Q8_0", os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-q8_0.gguf")),
        ("MODEL_A_CLEAN_BASE_Q4_K_M", clean_paths["MODEL_A_CLEAN_Q4"]),
        ("MODEL_A_WSP_V50_Q4_K_M", os.path.join(EXPORT_ROOT, "quillan-1b-sovereign-wsp-v3-q4_k_m.gguf")),

        ("MODEL_B_ORIGINAL_SAFETENSORS", os.path.join(MINI_SNAP, "model.safetensors")),
        ("MODEL_B_RELEASED_OLLAMA_F16", BLOB_MINI_REL),
        ("MODEL_B_CLEAN_BASE_Q4_K_M", clean_paths["MODEL_B_CLEAN_Q4"]),
        ("MODEL_B_WSP_V50_Q4_K_M", os.path.join(EXPORT_ROOT, "quillan-mini-0.5b-wsp-v3-q4_k_m.gguf")),

        ("MODEL_C_RELEASED_OLLAMA_Q8_0", BLOB_DRAGON_REL),
        ("MODEL_C_CLEAN_BASE_Q4_K_M", clean_paths["MODEL_C_CLEAN_Q4"]),
        ("MODEL_C_WSP_V50_Q4_K_M", os.path.join(EXPORT_ROOT, "quillan-dragon-1b-wsp-v3-q4_k_m.gguf")),

        ("NATIVE_ONI_1B_PT_WSP_V50", os.path.join(CKPT_DIR, "quillan_1b_tripartite.pt")),
        ("NATIVE_ONI_MINI_PT_WSP_V50", os.path.join(CKPT_DIR, "quillan_mini_0.6b_tripartite.pt")),
    ]

    ledger: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoints": {}
    }

    for tag, fpath in tracked_files:
        if os.path.exists(fpath):
            sz_bytes = os.path.getsize(fpath)
            sha = compute_sha256(fpath)
            ledger["checkpoints"][tag] = {
                "path": fpath,
                "size_bytes": sz_bytes,
                "size_mb": round(sz_bytes / (1024 * 1024), 2),
                "sha256": sha,
            }
            print(f"  {tag:30s} | {sz_bytes/(1024*1024):8.2f} MB | sha256:{sha[:16]}...")

    ledger_path = os.path.join(EXPORT_ROOT, "checkpoint_sha256_ledger.json")
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    print(f"\n[OK] Saved SHA256 Checkpoint Ledger to: {ledger_path} in {time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()

