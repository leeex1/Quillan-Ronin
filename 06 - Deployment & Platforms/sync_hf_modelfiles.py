import os
import time
from huggingface_hub import HfApi

api = HfApi()

repos = [
    ("CrashOverrideX/Quillan-Ronin", r"C:\02_QUILLAN\Modelfile", "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf"),
    ("CrashOverrideX/Quillan-Ronin-Mini", r"C:\02_QUILLAN\Modelfile.mini", "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf"),
    ("CrashOverrideX/Quillan-Dragon", r"C:\02_QUILLAN\Modelfile.dragon", "quillan-dragon-1b-wsp-v8-q4_k_m.gguf")
]

for repo_id, local_file, gguf_rel_name in repos:
    print(f"\n>>> Processing {repo_id}...")
    with open(local_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace the absolute Windows path with relative path for global Hugging Face portability
    lines = content.splitlines(keepends=True)
    new_lines = []
    for line in lines:
        if line.startswith("FROM C:/02_QUILLAN/exports/") or line.startswith("FROM exports/"):
            new_lines.append(f"FROM ./{gguf_rel_name}\n")
        else:
            new_lines.append(line)
    
    hf_modelfile_content = "".join(new_lines)
    tmp_path = os.path.join(r"C:\02_QUILLAN", f"Modelfile.hf.{repo_id.split('/')[-1]}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(hf_modelfile_content)
    
    print(f"Uploading {tmp_path} -> {repo_id}:Modelfile...")
    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="Modelfile",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update Modelfile with 2M context, canonical <think> reasoning protocol, and relative GGUF path"
    )
    print(f"Successfully synced Modelfile to {repo_id}!")

print("\nAll Hugging Face Modelfiles synchronized successfully!")
