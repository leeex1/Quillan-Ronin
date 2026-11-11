import os
from huggingface_hub import HfApi

api = HfApi()

UPLOADS = [
    {
        "repo_id": "CrashOverrideX/Quillan-Ronin",
        "files": [
            ("C:/02_QUILLAN/exports/quillan-1b-sovereign-wsp-v8-q4_k_m.gguf", "quillan-1b-sovereign-wsp-v8-q4_k_m.gguf"),
            ("C:/02_QUILLAN/Modelfile", "Modelfile"),
        ]
    },
    {
        "repo_id": "CrashOverrideX/Quillan-Ronin-Mini",
        "files": [
            ("C:/02_QUILLAN/exports/quillan-mini-0.5b-wsp-v8-q4_k_m.gguf", "quillan-mini-0.5b-wsp-v8-q4_k_m.gguf"),
            ("C:/02_QUILLAN/Modelfile.mini", "Modelfile"),
        ]
    },
    {
        "repo_id": "CrashOverrideX/Quillan-Dragon",
        "files": [
            ("C:/02_QUILLAN/exports/quillan-dragon-1b-wsp-v8-q4_k_m.gguf", "quillan-dragon-1b-wsp-v8-q4_k_m.gguf"),
            ("C:/02_QUILLAN/Modelfile.dragon", "Modelfile"),
        ]
    }
]

def main():
    for item in UPLOADS:
        repo_id = item["repo_id"]
        print(f"=== Uploading to {repo_id} ===")
        for local_path, path_in_repo in item["files"]:
            print(f"Uploading {local_path} -> {path_in_repo}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Deploy WSP v8 GGUF artifact with 2M context ({path_in_repo})"
            )
            print(f"Done: {path_in_repo}")

if __name__ == "__main__":
    main()
