from pathlib import Path
from huggingface_hub import snapshot_download
DST = r"C:\02_QUILLAN\scripts\hf_code"
Path(DST).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="CrashOverrideX/Quillan-Ronin", local_dir=DST,
                  allow_patterns=["evo_moe.py", "evo_moe_glm.py",
                                  "modeling_quillan_oni.py", "train_oni.py",
                                  "reasoning_engine_oni.py"])
print("CODE2 DONE", flush=True)
