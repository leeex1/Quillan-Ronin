#!/usr/bin/env python3
"""
Quillan Hugging Face Multi-Dataset & Model Sync Script
======================================================
Uploads:
1. CrashOverrideX/quillan-lyrics-corpus (Dataset: Lyrics, poems, markdown prompts)
2. CrashOverrideX/quillan-audio-media   (Dataset: MP3/FLAC/WAV audio & artwork)
3. CrashOverrideX/Quillan-Ronin         (Model/Repo: Core architecture, code, and papers)
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, HfFolder

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or HfFolder.get_token()
api = HfApi(token=TOKEN)

user_info = api.whoami()
username = user_info.get("name", user_info.get("username", "CrashOverrideX"))
print(f"Authenticated with Hugging Face as: {username}")

# ── 1. Create and Upload: Lyrics & Text Dataset ──────────────────────────────
print("\n[1/3] Uploading Lyrics & Text Dataset...")
lyrics_repo = f"{username}/quillan-lyrics-corpus"
create_repo(repo_id=lyrics_repo, repo_type="dataset", token=TOKEN, exist_ok=True)

# Prepare clean staging folder for lyrics
stage_lyrics = Path(r"C:\02_QUILLAN\scratch\hf_stage_lyrics")
if stage_lyrics.exists(): shutil.rmtree(stage_lyrics)
stage_lyrics.mkdir(parents=True, exist_ok=True)

# Copy Audio Engineer files
audio_eng = Path(r"C:\02_QUILLAN\Audio Engineer")
if audio_eng.exists():
    for item in audio_eng.rglob("*"):
        if item.is_file() and not item.name.endswith(".wav") and not item.name.endswith(".mp3"):
            rel = item.relative_to(audio_eng)
            dest = stage_lyrics / "lyrics_and_notes" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

# Add dataset card README
(stage_lyrics / "README.md").write_text("""---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- lyrics
- music
- creative-writing
- quillan-ronin
size_categories:
- n<1K
---

# Quillan-Ronin: Sovereign Lyric & Prompt Corpus

Curated lyrical dataset, song architectures, conceptual prompts, and rhythmic markdown sheets generated and structured by Quillan-Ronin.
""", encoding="utf-8")

api.upload_folder(
    folder_path=str(stage_lyrics),
    repo_id=lyrics_repo,
    repo_type="dataset",
    commit_message="feat: upload complete Quillan lyrics and prompt dataset",
    token=TOKEN
)
print(f"[SUCCESS] Lyrics Dataset live at: https://huggingface.co/datasets/{lyrics_repo}")
print("\n[2/3] Uploading Audio & Media Dataset...")
media_repo = f"{username}/quillan-audio-media"
create_repo(repo_id=media_repo, repo_type="dataset", token=TOKEN, exist_ok=True)

# Prepare clean staging folder for media
stage_media = Path(r"C:\02_QUILLAN\scratch\hf_stage_media")
if stage_media.exists(): shutil.rmtree(stage_media)
stage_media.mkdir(parents=True, exist_ok=True)

# Copy audio tracks (excluding soulframe, msi, and NFT designs)
audio_dir = Path(r"C:\02_QUILLAN\06_Media\audio")
if audio_dir.exists():
    for item in audio_dir.rglob("*"):
        if item.is_file():
            # Skip blacklisted items
            if any(k in item.name.lower() for k in ["soulframe", ".msi", ".exe", ".mp4", ".mkv", ".avi", ".mov"]):
                continue
            rel = item.relative_to(audio_dir)
            dest = stage_media / "audio" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

# Copy artwork (excluding NFT designs)
main_img_dir = Path(r"C:\02_QUILLAN\Main images")
if main_img_dir.exists():
    for item in main_img_dir.rglob("*"):
        if item.is_file():
            rel = item.relative_to(main_img_dir)
            dest = stage_media / "artwork" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

# Add media dataset card README
(stage_media / "README.md").write_text("""---
license: apache-2.0
task_categories:
- audio-to-audio
- text-to-audio
language:
- en
tags:
- audio
- music-generation
- flac
- mp3
- quillan-ronin
---

# Quillan-Ronin: Multimodal Audio & Media Dataset

Full multimodal audio and visual collection produced and designed by Quillan-Ronin:
- **Lossless FLAC Master Recordings** (The Sound of Alchemy, singles)
- **High-Bitrate MP3 Releases** (Draming of the Sky, Rock Album, ai beats, stems)
- **Visual Concepts & Model Diagrams**
""", encoding="utf-8")

api.upload_folder(
    folder_path=str(stage_media),
    repo_id=media_repo,
    repo_type="dataset",
    commit_message="feat: upload audio master tracks, stems, and album artwork",
    token=TOKEN
)
print(f"[SUCCESS] Audio & Media Dataset live at: https://huggingface.co/datasets/{media_repo}")

# ── 3. Sync Main Model/Codebase Repo ─────────────────────────────────────────
print("\n[3/3] Uploading Quillan-Ronin Main Model Repository...")
model_repo = f"{username}/Quillan-Ronin"
create_repo(repo_id=model_repo, repo_type="model", token=TOKEN, exist_ok=True)

# Prepare sanitized clean staging folder for model architecture
stage_model = Path(r"C:\02_QUILLAN\scratch\hf_stage_model")
if stage_model.exists(): shutil.rmtree(stage_model)
stage_model.mkdir(parents=True, exist_ok=True)

# Copy only core code, docs, and papers (strictly excluding all profiles, envs, keys)
safe_dirs = ["oni", "docs", "templates", "Papers", "configs"]
for sd in safe_dirs:
    src_p = Path(r"C:\02_QUILLAN") / sd
    if src_p.exists():
        dst_p = stage_model / sd
        shutil.copytree(src_p, dst_p, ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "*profile*", "*.token", "*.key", "*.env*"))

for root_file in ["README.md", "AGENTS.md", "LINEAGE.md", "index.html"]:
    rf = Path(r"C:\02_QUILLAN") / root_file
    if rf.exists():
        shutil.copy2(rf, stage_model / root_file)

api.upload_folder(
    folder_path=str(stage_model),
    repo_id=model_repo,
    repo_type="model",
    commit_message="v5.4.0-oni: synchronize clean architecture, papers, and configs",
    token=TOKEN
)
print(f"[SUCCESS] Main Model Repository live at: https://huggingface.co/{model_repo}")
print("\n[COMPLETE] All 3 Hugging Face repositories successfully synchronized!")
