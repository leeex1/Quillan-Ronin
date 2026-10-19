import os
import json
import glob
from pathlib import Path
import whisper
import tiktoken

enc = tiktoken.get_encoding("gpt2")
media_dir = Path(r"C:\02_QUILLAN\06_Media")
out_file = Path(r"C:\02_QUILLAN\training_data\media_transcripts_corpus.jsonl")

print("[*] Loading Whisper model (base)...")
model = whisper.load_model("base", device="cpu")

audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.mp4"]
media_files = []
for ext in audio_extensions:
    media_files.extend(list(media_dir.rglob(ext)))

print(f"[*] Found {len(media_files)} audio/video files in {media_dir}")

extracted_count = 0
with open(out_file, "w", encoding="utf-8") as out_f:
    for mf in media_files:
        try:
            print(f"[*] Transcribing {mf.name}...")
            result = model.transcribe(str(mf))
            text = result.get("text", "").strip()
            if text and len(text) > 30:
                tokens = enc.encode(f"Media Title / Transcript ({mf.name}):\n\n" + text)
                if len(tokens) >= 32:
                    out_f.write(json.dumps({"input_ids": tokens[:512], "labels": list(tokens[:512])}) + "\n")
                    extracted_count += 1
                    print(f"  [+] Transcribed {mf.name}: {len(tokens)} tokens")
        except Exception as e:
            print(f"[!] Error transcribing {mf.name}: {e}")

print(f"\n[COMPLETE] Transcribed {extracted_count} media files into {out_file}")
