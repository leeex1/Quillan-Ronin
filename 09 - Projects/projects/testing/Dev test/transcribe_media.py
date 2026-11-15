import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import tiktoken
import whisper

_MCP_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp")
if _MCP_DIR.exists() and str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from _shared.media_utils import decode_audio_pyav, resolve_ffmpeg_bin


def transcribe_media(
    media_dir: Optional[Path] = None,
    out_file: Optional[Path] = None,
    model_size: str = "tiny",
    max_files: Optional[int] = None,
    max_seconds: float = 30.0,
) -> dict:
    """Transcribe audio/video assets in media_dir using PyAV + Whisper."""
    resolve_ffmpeg_bin("ffmpeg")
    enc = tiktoken.get_encoding("gpt2")
    repo_root = Path(r"C:\02_QUILLAN")
    target_dir = Path(media_dir) if media_dir else (repo_root / "05 - Creative Works")
    target_out = (
        Path(out_file)
        if out_file
        else (repo_root / "09 - Projects" / "mcp" / "quillan_audio" / "sessions" / "media_transcripts_corpus.jsonl")
    )
    target_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size, device="cpu")

    audio_extensions = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.mp4")
    media_files = []
    if target_dir.exists():
        for ext in audio_extensions:
            media_files.extend(target_dir.rglob(ext))
    media_files = sorted(set(media_files))
    if max_files is not None and max_files > 0:
        media_files = media_files[:max_files]

    print(f"[*] Found {len(media_files)} audio/video files in {target_dir}")

    extracted_count = 0
    errors = []
    with open(target_out, "w", encoding="utf-8") as out_f:
        for mf in media_files:
            try:
                print(f"[*] Transcribing {mf.name}...")
                decoded = decode_audio_pyav(mf, max_seconds=max_seconds, target_sr=16000)
                result = model.transcribe(decoded["samples"], fp16=False)
                text = (result.get("text") or "").strip()
                if text:
                    tokens = enc.encode(f"Media Title / Transcript ({mf.name}):\n\n" + text)
                    out_f.write(
                        json.dumps(
                            {
                                "filename": mf.name,
                                "text": text,
                                "input_ids": tokens[:512],
                                "labels": list(tokens[:512]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    extracted_count += 1
                    print(f"  [+] Transcribed {mf.name}: {len(tokens)} tokens")
            except Exception as e:
                errors.append(f"{mf.name}: {e}")
                print(f"[!] Error transcribing {mf.name}: {e}")

    print(f"\n[COMPLETE] Transcribed {extracted_count} media files into {target_out}")
    return {"transcribed": extracted_count, "scanned": len(media_files), "output": str(target_out), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio/video media assets.")
    parser.add_argument("--media-dir", type=Path, default=None)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    transcribe_media(
        media_dir=args.media_dir,
        out_file=args.out_file,
        model_size=args.model_size,
        max_files=args.max_files,
        max_seconds=args.max_seconds,
    )
