#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 QUILLAN-RONIN v5.3.1 MULTIMODAL ALIGNMENT PIPELINE
Trains the image, audio, and video decoders/projector using FFmpeg pipelines.
Fits easily in 4GB VRAM by freezing the 519M parameter text backbone.
"""
import argparse
import os
import sys
import json
import math
import shutil
import logging
import subprocess
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT.parent.parent
REPO_ROOT = TRAINING_ROOT.parent
MCP_DIR = REPO_ROOT / "09 - Projects" / "mcp"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "multimodal_train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
LOG = logging.getLogger(__name__)

# Force UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for _p in (str(ROOT), str(TRAINING_ROOT), str(MCP_DIR)):
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign
from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_fused_optimizer import QuillanFusedOptimizer

try:
    from _shared.media_utils import resolve_ffmpeg_bin
except Exception:
    def resolve_ffmpeg_bin(binary_name: str = "ffmpeg"):
        found = shutil.which(binary_name)
        if found:
            return Path(found)
        exe = f"{binary_name}.exe" if os.name == "nt" else binary_name
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            winget = Path(local_app) / "Microsoft" / "WinGet" / "Packages"
            if winget.exists():
                for cand in sorted(winget.glob(f"Gyan.FFmpeg*/*/bin/{exe}"), reverse=True):
                    if cand.exists():
                        os.environ["PATH"] = f"{cand.parent}{os.pathsep}{os.environ.get('PATH', '')}"
                        return cand
        return None

_RESOLVED_FFMPEG = resolve_ffmpeg_bin("ffmpeg")
FFMPEG_PATH = str(_RESOLVED_FFMPEG) if _RESOLVED_FFMPEG else "ffmpeg"

# ── Hyperparameters ────────────────────────────────────────────────────────────
LR_MUON = 1e-4
LR_ADAMW = 5e-5
MAX_STEPS = 100
LOG_EVERY = 5
SAVE_EVERY = 25

# ── FFmpeg Media Pipeline ──────────────────────────────────────────────────────
def get_safe_path(relpath: str) -> Path:
    """Resolve and validate media paths inside C:\\02_QUILLAN (or explicit Downloads if present)."""
    raw = Path(relpath)
    allowed_roots = [REPO_ROOT.resolve(), (Path.home() / "Downloads").resolve()]
    candidates = [raw] if raw.is_absolute() else [
        REPO_ROOT / relpath,
        REPO_ROOT / "05 - Creative Works" / relpath,
        REPO_ROOT / "06_Media" / relpath,
        Path.home() / relpath,
    ]
    for cand in candidates:
        try:
            resolved = cand.resolve()
            if any(resolved == ar or ar in resolved.parents for ar in allowed_roots) and resolved.exists():
                return resolved
        except Exception:
            continue
    return (REPO_ROOT / "non_existent_safe_fallback").resolve()


def _resolve_ffmpeg_cmd() -> str:
    found = resolve_ffmpeg_bin("ffmpeg")
    return str(found) if found else FFMPEG_PATH


def load_image_ffmpeg(file_path: Path) -> torch.Tensor:
    """Decode and resize an image to 256x256 RGB24 [-1, 1] using FFmpeg (PIL fallback)."""
    ffmpeg_cmd = _resolve_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd, "-y", "-i", str(file_path),
        "-vf", "scale=256:256", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    expected_bytes = 256 * 256 * 3
    if proc.returncode == 0 and len(proc.stdout) >= expected_bytes:
        data = bytearray(proc.stdout[:expected_bytes])
        t = torch.frombuffer(data, dtype=torch.uint8).float().reshape(256, 256, 3).permute(2, 0, 1)
    else:
        from PIL import Image
        import numpy as np
        with Image.open(file_path) as im:
            arr = np.array(im.convert("RGB").resize((256, 256)), dtype=np.uint8)
        t = torch.from_numpy(arr).float().permute(2, 0, 1)
    return (t / 127.5) - 1.0


def load_audio_ffmpeg(file_path: Path) -> torch.Tensor:
    """Decode and resample audio to 16000Hz mono [-1, 1] (1 sec) via FFmpeg (PyAV fallback)."""
    ffmpeg_cmd = _resolve_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd, "-y", "-i", str(file_path), "-t", "1",
        "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    expected_bytes = 16000 * 2
    if proc.returncode == 0 and proc.stdout:
        data = bytearray(proc.stdout)
        if len(data) < expected_bytes:
            data.extend(b"\x00" * (expected_bytes - len(data)))
        t = torch.frombuffer(data[:expected_bytes], dtype=torch.int16).float().unsqueeze(0) / 32768.0
        return t
    import av
    import numpy as np
    chunks = []
    with av.open(str(file_path)) as c:
        st = next(s for s in c.streams if s.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        for frame in c.decode(st):
            for out_f in resampler.resample(frame):
                chunks.append(out_f.to_ndarray().reshape(-1))
            if sum(len(x) for x in chunks) >= 16000:
                break
    arr = np.concatenate(chunks) if chunks else np.zeros(16000, dtype=np.float32)
    if len(arr) < 16000:
        arr = np.pad(arr, (0, 16000 - len(arr)))
    return torch.from_numpy(arr[:16000].astype(np.float32)).unsqueeze(0)


def load_video_ffmpeg(file_path: Path) -> torch.Tensor:
    """Decode and sample 8 frames at 64x64 RGB24 [-1, 1] via FFmpeg (PyAV fallback)."""
    ffmpeg_cmd = _resolve_ffmpeg_cmd()
    cmd = [
        ffmpeg_cmd, "-y", "-i", str(file_path),
        "-vf", "scale=64:64,fps=8", "-vframes", "8",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    expected_bytes = 8 * 64 * 64 * 3
    if proc.returncode == 0 and proc.stdout:
        data = bytearray(proc.stdout)
        if len(data) < expected_bytes:
            data.extend(b"\x00" * (expected_bytes - len(data)))
        t = torch.frombuffer(data[:expected_bytes], dtype=torch.uint8).float()
        t = t.reshape(8, 64, 64, 3).permute(3, 0, 1, 2)
    else:
        import av
        import numpy as np
        frames = []
        with av.open(str(file_path)) as c:
            vs = next(s for s in c.streams if s.type == "video")
            for frame in c.decode(vs):
                im = frame.to_image().convert("RGB").resize((64, 64))
                frames.append(np.array(im, dtype=np.uint8))
                if len(frames) >= 8:
                    break
        while len(frames) < 8:
            frames.append(frames[-1] if frames else np.zeros((64, 64, 3), dtype=np.uint8))
        arr = np.stack(frames, axis=0)
        t = torch.from_numpy(arr).float().permute(3, 0, 1, 2)
    return (t / 127.5) - 1.0


def tensor_to_ingestion_features(tensor: torch.Tensor, modality: str, out_dim: int = 384) -> torch.Tensor:
    """Convert raw image [3,H,W], audio [1,N], or video [3,T,H,W] tensor into [1, S, 384] tokens for QuillanRoninSovereign."""
    if modality == "image":
        patches = F.adaptive_avg_pool2d(tensor.unsqueeze(0), (8, 8))  # [1, 3, 8, 8] -> 192
        flat = patches.flatten(1)
        rep = torch.cat([flat, flat], dim=-1)  # [1, 384]
        return rep.unsqueeze(1)
    if modality == "audio":
        pooled = F.adaptive_avg_pool1d(tensor.unsqueeze(0), out_dim)  # [1, 1, 384]
        return pooled
    if modality == "video":
        # tensor: [3, T, H, W] -> [T, 3, H, W]
        frames = tensor.permute(1, 0, 2, 3)
        patches = F.adaptive_avg_pool2d(frames, (8, 8)).flatten(1)  # [T, 192]
        rep = torch.cat([patches, patches], dim=-1).unsqueeze(0)  # [1, T, 384]
        return rep
    raise ValueError(f"Unsupported modality: {modality}")

# ── Main Training Loop ─────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LOG.info("=== Quillan Multimodal Alignment Pipeline ===")
    LOG.info(f"Device: {device}")

    # Tokenizer
    tok = QuillanBPETokenizer()
    tok.load(str(ROOT / "quillan_bpe_tokenizer.pkl"))

    # Model configuration
    cfg = QuillanArchConfig(text_only=False)
    model = QuillanRoninSovereign(cfg)

    # Load latest trained text checkpoint
    base_ckpt = CKPT_DIR / "quillan_v8_domain_final.pt"
    if not base_ckpt.exists():
        candidates = sorted(
            CKPT_DIR.glob("quillan_v8_domain_step_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        if candidates:
            base_ckpt = candidates[0]
            
    if base_ckpt.exists():
        sd = torch.load(base_ckpt, map_location='cpu', weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        LOG.info(f"Loaded base text weights from {base_ckpt.name} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        LOG.warning("No domain text checkpoint found! Training from seeded/init weights.")

    model = model.to(device).half() if device == "cuda" else model.to(device)
    model.train()

    # Freeze all text parameters, keep only multimodal projection & modality embeddings trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ["img_proj", "mod_emb"]):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    total_trainable = sum(p.numel() for p in trainable_params)
    LOG.info(f"Multimodal training mode: {total_trainable/1e6:.2f}M trainable parameters (text-backbone frozen)")

    # Optimizer
    opt = QuillanFusedOptimizer(
        model,
        lr_muon=LR_MUON,
        lr_adamw=LR_ADAMW,
        lr_min=1e-5,
        momentum=0.95,
        weight_decay=0.01,
        ns_steps=3,
        warmup=10,
        total_steps=MAX_STEPS,
        start_step=0
    )

    # Data stream
    balanced_file = ROOT / "training_data" / "domain_splits" / "domain_general_routing_balanced.jsonl"
    if not balanced_file.exists():
        LOG.error(f"Balanced dataset not found: {balanced_file}")
        return

    step = 0
    accum_loss = 0.0
    
    with open(balanced_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if step >= MAX_STEPS:
                break
            try:
                d = json.loads(line)
                meta = d.get("meta", {})
                source = meta.get("source")
                if source != "media_metadata":
                    continue
                
                relpath = meta.get("relpath", "")
                if not relpath:
                    continue

                safe_file_path = get_safe_path(relpath)
                if not safe_file_path.exists():
                    continue

                # Tokenize prompt metadata text
                text = d.get("text", "")
                tokens = tok.encode(text)[:128]
                if len(tokens) < 8:
                    continue

                txt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

                # Process based on modality type via QuillanRoninSovereign(txt, img=features)
                ext = safe_file_path.suffix.lower()
                loss = None
                mod_dtype = next(model.parameters()).dtype
                mod_feats = None

                if ext in [".png", ".jpg", ".jpeg", ".webp"]:
                    target = load_image_ffmpeg(safe_file_path).to(device, dtype=mod_dtype)
                    mod_feats = tensor_to_ingestion_features(target, "image")
                elif ext in [".mp3", ".wav", ".flac", ".m4a"]:
                    target = load_audio_ffmpeg(safe_file_path).to(device, dtype=mod_dtype)
                    mod_feats = tensor_to_ingestion_features(target, "audio")
                elif ext in [".mp4"]:
                    target = load_video_ffmpeg(safe_file_path).to(device, dtype=mod_dtype)
                    mod_feats = tensor_to_ingestion_features(target, "video")

                if mod_feats is not None and txt_tensor.shape[1] >= 2:
                    out = model(txt_tensor[:, :-1], img=mod_feats.to(device, dtype=mod_dtype))
                    logits = out[0] if isinstance(out, tuple) else out
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), txt_tensor[:, 1:].reshape(-1))

                if loss is not None:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step(integrity=1.0)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    accum_loss += loss.item()

                    if step % LOG_EVERY == 0:
                        avg_loss = accum_loss / max(1, step % LOG_EVERY if step > 0 else 1)
                        LOG.info(f"Step {step:>4d}/{MAX_STEPS} | loss {avg_loss:.4f} | file: {safe_file_path.name}")
                        accum_loss = 0.0

                    if step > 0 and step % SAVE_EVERY == 0:
                        ckpt_path = CKPT_DIR / f"quillan_v8_multimodal_step_{step}.pt"
                        trainable_sd = {k: v for k, v in model.state_dict().items() if "img_proj" in k or "mod_emb" in k}
                        torch.save({"state_dict": trainable_sd, "step": step}, ckpt_path)
                        LOG.info(f"Saved multimodal checkpoint: {ckpt_path.name}")

                    step += 1
            except Exception as e:
                LOG.warning(f"Skipping multimodal sample due to {type(e).__name__}: {e}")
                continue

    # Final Save
    final_path = CKPT_DIR / "quillan_v8_multimodal_final.pt"
    trainable_sd = {k: v for k, v in model.state_dict().items() if "img_proj" in k or "mod_emb" in k}
    torch.save({"state_dict": trainable_sd, "step": step}, final_path)
    LOG.info(f"Done. Final multimodal weights saved: {final_path.name}")

if __name__ == "__main__":
    main()
