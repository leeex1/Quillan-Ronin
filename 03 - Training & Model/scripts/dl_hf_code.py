from pathlib import Path
from huggingface_hub import snapshot_download
DST = r"C:\02_QUILLAN\scripts\hf_code"
Path(DST).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="CrashOverrideX/Quillan-Ronin", local_dir=DST,
                  allow_patterns=["modeling_quillan_oni.py",
                                  "configuration_quillan_oni.py",
                                  "quillan_v5_4_oni.py"])
print("CODE DONE", flush=True)
