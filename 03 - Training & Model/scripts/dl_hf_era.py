from pathlib import Path
from huggingface_hub import list_repo_commits, hf_hub_download
commits = list_repo_commits("CrashOverrideX/Quillan-Ronin", repo_type="model")
cands = [c for c in commits if c.title.startswith("Upload quillan_v5_4_oni.py")]
for c in cands:
    print(f"cand {c.created_at} :: {c.commit_id[:8]}", flush=True)
target = [c for c in cands if "02:26:14" in str(c.created_at)][0]
print("era commit=" + target.commit_id, flush=True)
DST = r"C:\02_QUILLAN\scripts\hf_code_era"
Path(DST).mkdir(parents=True, exist_ok=True)
for f in ["quillan_v5_4_oni.py", "train_oni.py"]:
    p = hf_hub_download("CrashOverrideX/Quillan-Ronin", f,
                        revision=target.commit_id, local_dir=DST)
    print(f"got {f} -> {p}", flush=True)
print("ERA DONE", flush=True)
