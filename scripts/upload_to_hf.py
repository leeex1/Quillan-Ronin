#!/usr/bin/env python3
"""Upload Sovereign GGUF artifacts and 2M Modelfiles to Hugging Face Hub.

Dynamically resolves the latest promoted v23 GGUFs (falling back cleanly to
v20 / v15 / v18b if v23 is not yet on disk), matching build_full_sovereign_modelfiles.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from huggingface_hub import HfApi

ROOT_DIR = Path(r"C:\02_QUILLAN")
EXPORTS_DIR = ROOT_DIR / "exports"

SPECS = [
    {
        "key": "flagship",
        "repo_id": "CrashOverrideX/Quillan-Ronin",
        "v23_glob": "quillan-1b*wsp-v23*-q8_0.gguf",
        "fallback_gguf": "exports/quillan-1b-sovereign-wsp-v20-q8_0.gguf",
        "modelfile": "Modelfile",
    },
    {
        "key": "mini",
        "repo_id": "CrashOverrideX/Quillan-Ronin-Mini",
        "v23_glob": "quillan-mini*wsp-v23*-q8_0.gguf",
        "fallback_gguf": "exports/quillan-mini-0.5b-wsp-v15-q8_0.gguf",
        "modelfile": "Modelfile.mini",
    },
    {
        "key": "dragon",
        "repo_id": "CrashOverrideX/Quillan-Dragon",
        "v23_glob": "quillan-dragon*wsp-v23*-q8_0.gguf",
        "fallback_gguf": "exports/quillan-dragon-1b-wsp-v18b-q8_0.gguf",
        "modelfile": "Modelfile.dragon",
    },
]


def resolve_gguf(spec: dict) -> Path:
    candidates = [
        c
        for c in EXPORTS_DIR.glob(spec["v23_glob"])
        if c.is_file()
        and c.stat().st_size > 0
        and "tmp" not in c.name.lower()
        and "stageb" not in c.name.lower()
    ]
    if candidates:
        candidates.sort(key=lambda c: c.stat().st_mtime, reverse=True)
        return candidates[0]
    fb = ROOT_DIR / spec["fallback_gguf"]
    if not fb.exists() or fb.stat().st_size <= 0:
        raise FileNotFoundError(f"GGUF not found for {spec['key']}: {fb}")
    return fb


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Sovereign GGUFs & Modelfiles to HF Hub.")
    parser.add_argument("--dry-run", action="store_true", help="Verify paths without uploading.")
    parser.add_argument("--model", choices=["all", "flagship", "mini", "dragon"], default="all")
    args = parser.parse_args()

    api = HfApi()
    for spec in SPECS:
        if args.model != "all" and spec["key"] != args.model:
            continue
        gguf_path = resolve_gguf(spec)
        mf_path = ROOT_DIR / spec["modelfile"]
        if not mf_path.exists() or mf_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Modelfile not found: {mf_path}")

        repo_id = spec["repo_id"]
        print(
            f"[{spec['key']}] repo={repo_id} | gguf={gguf_path.name} ({gguf_path.stat().st_size:,} B) "
            f"| modelfile={mf_path.name} ({mf_path.stat().st_size:,} B)"
        )
        if args.dry_run:
            continue

        for local_path, path_in_repo in [(gguf_path, gguf_path.name), (mf_path, "Modelfile")]:
            print(f"Uploading {local_path} -> {repo_id}:{path_in_repo}...")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Deploy Sovereign WSP artifact with 2M context ({path_in_repo})",
            )
            print(f"Done: {repo_id}:{path_in_repo}")


if __name__ == "__main__":
    main()
