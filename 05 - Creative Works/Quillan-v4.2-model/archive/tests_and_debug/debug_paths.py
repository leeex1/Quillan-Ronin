#!/usr/bin/env python3
"""
Debug script to check data loading paths
"""
import os
import glob

data_dir = "."

print("=== Path Debug ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory: {data_dir}")
print(f"Absolute data directory: {os.path.abspath(data_dir)}")
print(f"Parent directory: {os.path.dirname(data_dir)}")
print(f"Parent absolute: {os.path.abspath(os.path.dirname(data_dir))}")

# Check song lyrics path
lyrics_dir = os.path.join(os.path.dirname(data_dir), "Songs Lyrics")
lyrics_abs = os.path.abspath(lyrics_dir)
print(f"\nSong lyrics directory: {lyrics_dir}")
print(f"Absolute lyrics path: {lyrics_abs}")
print(f"Lyrics directory exists: {os.path.exists(lyrics_abs)}")

if os.path.exists(lyrics_abs):
    lyrics_files = glob.glob(os.path.join(lyrics_abs, "*.md"))
    print(f"Lyrics .md files found: {len(lyrics_files)}")
    if lyrics_files:
        print(f"Sample files: {lyrics_files[:3]}")

# Check knowledge files path
knowledge_dir = os.path.join(os.path.dirname(data_dir), "Quillan Knowledge files")
knowledge_abs = os.path.abspath(knowledge_dir)
print(f"\nKnowledge directory: {knowledge_dir}")
print(f"Absolute knowledge path: {knowledge_abs}")
print(f"Knowledge directory exists: {os.path.exists(knowledge_abs)}")

if os.path.exists(knowledge_abs):
    knowledge_files = glob.glob(os.path.join(knowledge_abs, "*.md"))
    print(f"Knowledge .md files found: {len(knowledge_files)}")
    if knowledge_files:
        print(f"Sample files: {knowledge_files[:3]}")

# Check JSONL file
jsonl_path = os.path.join(data_dir, "Quillan_finetune_full_dataset.jsonl")
jsonl_abs = os.path.abspath(jsonl_path)
print(f"\nJSONL file: {jsonl_path}")
print(f"Absolute JSONL path: {jsonl_abs}")
print(f"JSONL file exists: {os.path.exists(jsonl_abs)}")

if os.path.exists(jsonl_abs):
    with open(jsonl_abs, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"JSONL lines: {len(lines)}")
        if lines:
            print(f"First line preview: {lines[0][:100]}...")

print("\n=== Debug Complete ===")
