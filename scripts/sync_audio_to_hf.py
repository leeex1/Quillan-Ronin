#!/usr/bin/env python3
"""Sync local MP3s to Hugging Face dataset (canonical audio source).

HF repo: CrashOverrideX/quillan-audio-media (dataset type)
Usage:
  python scripts/sync_audio_to_hf.py [--dry-run]
Requires: huggingface_hub, HF_TOKEN env var (write token)
Website streams from HF CDN with Range seeks -> zero GitHub LFS bandwidth.
GitHub repo keeps only HTML/JS/CSS + track manifest, never MP3 binaries.
"""

import argparse, os, sys
from pathlib import Path

LOCAL_SOURCES = ["assets/audio", "audio"]
HF_REPO = "CrashOverrideX/quillan-audio-media"


def collect() -> list[tuple[Path, str]]:
    items = []
    for src in LOCAL_SOURCES:
        p = Path(src)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*")):
            if (
                f.suffix.lower() in {".mp3", ".flac", ".m4a", ".ogg", ".wav"}
                and f.is_file()
            ):
                # HF path mirrors local repo-relative path
                items.append((f, f.as_posix()))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    items = collect()
    total = sum(f.stat().st_size for f, _ in items)
    print(f"Found {len(items)} audio files, {total / 1e6:.1f} MB")
    if args.dry_run:
        for f, rel in items:
            print(f"  would upload {f} -> {rel} ({f.stat().st_size / 1e6:.1f} MB)")
        return
    tok = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN_PATH")
    )
    if not tok:
        print(
            "Missing HF_TOKEN. Create one at https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        sys.exit(1)
    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    for f, rel in items:
        print(f"Uploading {rel} ...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=rel,
            repo_id=HF_REPO,
            repo_type="dataset",
        )
    print(f"Done. HF CDN base: https://huggingface.co/datasets/{HF_REPO}/resolve/main/")


if __name__ == "__main__":
    main()
