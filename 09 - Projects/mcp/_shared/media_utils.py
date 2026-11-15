#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Omni-Modal Utilities for Quillan MCP Servers
===================================================
Provides:
- Dynamic FFmpeg/FFprobe discovery (PATH, env, WinGet Gyan.FFmpeg packages)
- Canonicalized safe path validation inside ALLOWED_ROOT
- PyAV-based fast audio decoding, spectral centroid, RMS/peak, and onset BPM
- PyAV + PIL-based video/image inspection and frame/audio extraction
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_ALLOWED_ROOT = Path(os.environ.get("QUILLAN_BASE", r"C:\02_QUILLAN")).resolve()


def resolve_ffmpeg_bin(binary_name: str = "ffmpeg") -> Optional[Path]:
    """Discover ffmpeg/ffprobe dynamically and prepend its directory to PATH."""
    found = shutil.which(binary_name)
    if found:
        return Path(found)

    exe = f"{binary_name}.exe" if os.name == "nt" else binary_name
    candidates: List[Path] = []

    for env_key in ("FFMPEG_BIN", "FFMPEG_PATH", "FFPROBE_BIN"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        p = Path(raw)
        candidates.append(p / exe if p.is_dir() else (p.parent / exe if p.name.lower() != exe.lower() else p))

    local_app = os.environ.get("LOCALAPPDATA", "")
    if local_app:
        winget_root = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            candidates.extend(sorted(winget_root.glob(f"Gyan.FFmpeg*/*/bin/{exe}"), reverse=True))
            candidates.extend(sorted(winget_root.glob(f"*ffmpeg*/**/bin/{exe}"), reverse=True))

    candidates.append(Path(r"C:\ffmpeg\bin") / exe)

    for cand in candidates:
        if cand.exists() and cand.is_file():
            bin_dir = str(cand.parent)
            cur_path = os.environ.get("PATH", "")
            if bin_dir.lower() not in cur_path.lower():
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{cur_path}" if cur_path else bin_dir
            return cand
    return None


def validate_safe_file(
    raw_path: str,
    allowed_root: Optional[Path] = None,
    must_exist: bool = True,
) -> Path:
    """Resolve path safely, rejecting symlinks/traversal escaping allowed_root."""
    if not raw_path or not isinstance(raw_path, str):
        raise ValueError("Path must be a non-empty string.")
    root = (allowed_root or DEFAULT_ALLOWED_ROOT).resolve()
    p = Path(raw_path)
    resolved = p.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"Path '{resolved}' escapes allowed root '{root}'.")
    if must_exist:
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Path is not a regular file: {resolved}")
    return resolved


def estimate_bpm_numpy(samples: Any, sr: int) -> float:
    """Estimate BPM via vectorized short-time energy onset flux autocorrelation."""
    import numpy as np

    if len(samples) < max(sr // 2, 2048):
        return 0.0
    hop = max(256, sr // 100)  # ~10ms hop -> ~100 Hz onset rate
    n_frames = len(samples) // hop
    if n_frames < 16:
        return 0.0
    framed = samples[: n_frames * hop].reshape(n_frames, hop)
    energy = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    onset = onset - np.mean(onset)
    if float(np.max(np.abs(onset))) < 1e-7:
        return 0.0

    fps = sr / float(hop)
    min_lag = max(1, int(fps * 60.0 / 200.0))  # 200 BPM
    max_lag = min(len(onset) - 1, int(fps * 60.0 / 50.0))  # 50 BPM
    if max_lag <= min_lag:
        return 0.0

    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    window = ac[min_lag : max_lag + 1]
    if len(window) == 0 or float(np.max(window)) <= 0.0:
        return 0.0
    best_lag = min_lag + int(np.argmax(window))
    return round(60.0 * fps / float(best_lag), 1)


def compute_spectral_centroid_numpy(samples: Any, sr: int) -> float:
    """Compute mean spectral centroid in Hz using vectorized FFT."""
    import numpy as np

    n_fft = 2048
    hop = 1024
    if len(samples) < n_fft:
        pad = np.zeros(n_fft, dtype=np.float32)
        pad[: len(samples)] = samples
        samples = pad
    n_frames = max(1, 1 + (len(samples) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + (np.arange(n_frames)[:, None] * hop)
    frames = samples[idx] * np.hanning(n_fft)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))[None, :]
    denom = np.sum(mag, axis=1) + 1e-12
    centroids = np.sum(mag * freqs, axis=1) / denom
    return round(float(np.mean(centroids)), 1)


def decode_audio_pyav(
    path: Path,
    max_seconds: float = 30.0,
    target_sr: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode audio stream (including audio inside video containers) via PyAV."""
    import av
    import numpy as np

    resolve_ffmpeg_bin("ffmpeg")
    chunks: List[np.ndarray] = []
    with av.open(str(path)) as container:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError(f"No audio stream found in '{path.name}'.")
        stream = audio_streams[0]
        orig_sr = int(stream.rate or 44100)
        out_sr = int(target_sr or orig_sr)
        channels = int(stream.channels or 1)
        codec = str(stream.codec_context.name or "unknown")
        dur_sec = float(container.duration) / 1_000_000.0 if container.duration else 0.0
        max_samples = int(out_sr * max_seconds) if max_seconds > 0 else 0
        collected = 0
        resampler = av.AudioResampler(format="flt", layout="mono", rate=out_sr)

        for frame in container.decode(stream):
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                if len(arr):
                    chunks.append(arr)
                    collected += len(arr)
            if max_samples and collected >= max_samples:
                break

        for out_frame in resampler.resample(None):
            arr = out_frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
            if len(arr):
                chunks.append(arr)
                collected += len(arr)

    if not chunks:
        raise ValueError(f"Decoded 0 audio samples from '{path.name}'.")

    samples = np.concatenate(chunks)
    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]
    if dur_sec <= 0.0:
        dur_sec = round(len(samples) / float(out_sr), 3)

    rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
    rms_db = round(20.0 * float(np.log10(max(rms, 1e-9))), 2)
    peak = round(float(np.max(np.abs(samples))), 4)
    cent = compute_spectral_centroid_numpy(samples, out_sr)
    bpm = estimate_bpm_numpy(samples, out_sr)

    return {
        "samples": samples,
        "sample_rate": out_sr,
        "source_sample_rate": orig_sr,
        "channels": channels,
        "codec": codec,
        "duration_sec": round(dur_sec, 3),
        "analyzed_sec": round(len(samples) / float(out_sr), 3),
        "rms_db": rms_db,
        "peak": peak,
        "brightness_centroid": cent,
        "bpm": bpm,
    }


def inspect_image_file(path: Path) -> Dict[str, Any]:
    """Inspect an image file using Pillow and return verified properties."""
    from PIL import Image, ImageStat

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        stat = ImageStat.Stat(rgb)
        return {
            "path": str(path),
            "format": im.format,
            "mode": im.mode,
            "width": im.width,
            "height": im.height,
            "file_size_bytes": path.stat().st_size,
            "mean_rgb": [round(float(v), 2) for v in stat.mean],
            "stddev_rgb": [round(float(v), 2) for v in stat.stddev],
        }


def inspect_video_file(path: Path, sample_frames: int = 3) -> Dict[str, Any]:
    """Inspect a video container via PyAV and decode up to sample_frames RGB keyframes."""
    import av
    import numpy as np

    resolve_ffmpeg_bin("ffmpeg")
    frame_shapes: List[Tuple[int, int]] = []
    frame_means: List[List[float]] = []
    with av.open(str(path)) as container:
        vstreams = [s for s in container.streams if s.type == "video"]
        astreams = [s for s in container.streams if s.type == "audio"]
        if not vstreams:
            raise ValueError(f"No video stream found in '{path.name}'.")
        vs = vstreams[0]
        dur_sec = float(container.duration) / 1_000_000.0 if container.duration else 0.0
        fps = float(vs.average_rate) if vs.average_rate else 0.0
        info: Dict[str, Any] = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "duration_sec": round(dur_sec, 3),
            "video_codec": str(vs.codec_context.name or "unknown"),
            "width": int(vs.codec_context.width or 0),
            "height": int(vs.codec_context.height or 0),
            "fps": round(fps, 3),
            "has_audio": bool(astreams),
            "audio_codec": str(astreams[0].codec_context.name) if astreams else None,
            "stream_count": len(container.streams),
        }
        for i, frame in enumerate(container.decode(vs)):
            rgb = frame.to_ndarray(format="rgb24")
            frame_shapes.append((int(rgb.shape[1]), int(rgb.shape[0])))
            frame_means.append([round(float(c), 2) for c in rgb.mean(axis=(0, 1))])
            if i + 1 >= max(1, sample_frames):
                break
        info["decoded_sample_frames"] = len(frame_shapes)
        info["sample_frame_mean_rgb"] = frame_means
    return info
