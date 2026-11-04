"""
=============================================================================
QUILLAN-RONIN: HUGGING FACE GGUF PUBLICATION ENGINE
=============================================================================
Publishes calibrated Sovereign GGUF binaries (Q4_K_M and Q8_0) directly
to Hugging Face Hub (CrashOverrideX/Quillan-Ronin) for global distribution.
=============================================================================
"""

import os
import sys
import time
from huggingface_hub import HfApi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ID = "CrashOverrideX/Quillan-Ronin"
EXPORTS_DIR = r"C:\02_QUILLAN\exports"

files_to_upload = [
    ("quillan-1b-sovereign-q4_k_m.gguf", "quillan-1b-sovereign-q4_k_m.gguf"),
    ("quillan-1b-sovereign-q8_0.gguf", "quillan-1b-sovereign-q8_0.gguf")
]

print("=" * 80)
print(f"HUGGING FACE GGUF PUBLICATION: {REPO_ID}")
print("=" * 80)

api = HfApi()
who = api.whoami()
print(f"Authenticated as: {who.get('name', 'Unknown')}\n")

for local_name, remote_name in files_to_upload:
    local_path = os.path.join(EXPORTS_DIR, local_name)
    if not os.path.exists(local_path):
        print(f"❌ File not found: {local_path}")
        continue
    
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f">>> Uploading {local_name} ({size_mb:.2f} MB) -> {REPO_ID}:{remote_name}...")
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_name,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Add Sovereign GGUF {local_name} ({size_mb:.1f} MB)"
    )
    elapsed = time.time() - t0
    rate = size_mb / max(0.1, elapsed)
    print(f"✅ Successfully uploaded {local_name} in {elapsed:.1f}s ({rate:.2f} MB/s)\n")

print("=" * 80)
print("🎉 ALL SOVEREIGN GGUF MODELS PUBLISHED TO HUGGING FACE SUCCESSFULLY!")
print("=" * 80)

