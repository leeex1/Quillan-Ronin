#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 QUILLAN-RONIN v5.3.1 MULTIMODAL ALIGNMENT PIPELINE
Trains the image, audio, and video decoders/projector using FFmpeg pipelines.
Fits easily in 4GB VRAM by freezing the 519M parameter text backbone.
"""
import os
import sys
import json
import math
import logging
import subprocess
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

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

sys.path.insert(0, str(ROOT))
from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign
from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_fused_optimizer import QuillanFusedOptimizer

# ── Environment & Paths ────────────────────────────────────────────────────────
DOWNLOADS_DIR = Path("C:/Users/Admin/Downloads").resolve()
FFMPEG_PATH = r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
if not os.path.exists(FFMPEG_PATH):
    FFMPEG_PATH = "ffmpeg"  # Fallback to PATH

# ── Hyperparameters ────────────────────────────────────────────────────────────
LR_MUON = 1e-4
LR_ADAMW = 5e-5
MAX_STEPS = 100
LOG_EVERY = 5
SAVE_EVERY = 25

# ── FFmpeg Media Pipeline ──────────────────────────────────────────────────────
def get_safe_path(relpath: str) -> Path:
    """Resolve and validate relative paths, with fallback mappings for relocated folders."""
    norm_path = relpath.replace('\\', '/')
    base_user = Path("C:/Users/Admin").resolve()
    
    def validate(p: Path) -> Path:
        try:
            resolved = p.resolve()
            if not str(resolved).lower().startswith("c:\\users\\admin"):
                raise ValueError(f"Security boundary breach: {resolved}")
            return resolved
        except Exception:
            return base_user / "non_existent_safe_fallback"
    
    # Try 1: Relative to C:/Users/Admin
    p1 = (base_user / norm_path).resolve()
    if p1.exists():
        return validate(p1)
        
    # Try 2: If path starts with Downloads, check if it needs nested folder mapping
    if norm_path.startswith("Downloads/"):
        parts = norm_path.split('/')
        if len(parts) > 1:
            filename = parts[-1]
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.mp3', '.wav'] and 'audio' not in parts:
                p_audio = (base_user / "Downloads" / "audio" / "/".join(parts[1:])).resolve()
                if p_audio.exists():
                    return validate(p_audio)
            elif ext in ['.mp4'] and 'vids' not in parts:
                p_vids = (base_user / "Downloads" / "vids" / "/".join(parts[1:])).resolve()
                if p_vids.exists():
                    return validate(p_vids)
            elif ext in ['.png', '.jpg', '.jpeg'] and 'iamges' not in parts:
                p_img = (base_user / "Downloads" / "iamges" / "/".join(parts[1:])).resolve()
                if p_img.exists():
                    return validate(p_img)

    # Try 3: Map moved Quillan-Ronin directories (e.g. into Ronin-Saga-Neo-Eden)
    if "Quillan-Ronin/" in norm_path:
        norm_path_moved = norm_path.replace("Quillan-Ronin/", "Quillan-Ronin/Ronin-Saga-Neo-Eden/")
        p3 = (base_user / norm_path_moved).resolve()
        if p3.exists():
            return validate(p3)
            
    # Try 4: Check if file name matches any file in Downloads folders
    filename = norm_path.split('/')[-1]
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.png', '.jpg', '.jpeg']:
        p_fallback = Path(f"C:/Users/Admin/Downloads/iamges/{filename}").resolve()
        if p_fallback.exists():
            return validate(p_fallback)
    elif ext in ['.mp3', '.wav']:
        import glob
        matches = glob.glob(f"C:/Users/Admin/Downloads/audio/**/{filename}", recursive=True)
        if matches:
            return validate(Path(matches[0]))
    elif ext in ['.mp4']:
        p_fallback = Path(f"C:/Users/Admin/Downloads/vids/{filename}").resolve()
        if p_fallback.exists():
            return validate(p_fallback)

    return validate(p1)

def load_image_ffmpeg(file_path: Path) -> torch.Tensor:
    """Use FFmpeg to decode and resize an image to 256x256 RGB24."""
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(file_path),
        "-vf", "scale=256:256", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg image load failed: {proc.stderr.decode('utf-8', errors='replace')}")
    
    expected_bytes = 256 * 256 * 3
    data = proc.stdout
    if len(data) < expected_bytes:
        data = data + b"\x00" * (expected_bytes - len(data))
    elif len(data) > expected_bytes:
        data = data[:expected_bytes]
        
    t = torch.frombuffer(data, dtype=torch.uint8).float()
    t = t.reshape(256, 256, 3).permute(2, 0, 1)  # [3, 256, 256]
    t = (t / 127.5) - 1.0  # Normalize to [-1, 1]
    return t

def load_audio_ffmpeg(file_path: Path) -> torch.Tensor:
    """Use FFmpeg to decode and resample audio to 16000Hz s16le, 1 sec."""
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(file_path),
        "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg audio load failed: {proc.stderr.decode('utf-8', errors='replace')}")
    
    expected_bytes = 16000 * 2  # 16000 samples of 16-bit (2 bytes)
    data = proc.stdout
    if len(data) < expected_bytes:
        data = data + b"\x00" * (expected_bytes - len(data))
    elif len(data) > expected_bytes:
        data = data[:expected_bytes]
        
    t = torch.frombuffer(data, dtype=torch.int16).float()
    t = t.unsqueeze(0)  # [1, 16000]
    t = t / 32768.0  # Normalize to [-1, 1]
    return t

def load_video_ffmpeg(file_path: Path) -> torch.Tensor:
    """Use FFmpeg to decode and sample 8 frames at 64x64 RGB24."""
    cmd = [
        FFMPEG_PATH, "-y", "-i", str(file_path),
        "-vf", "scale=64:64,fps=8", "-vframes", "8",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg video load failed: {proc.stderr.decode('utf-8', errors='replace')}")
    
    expected_bytes = 8 * 64 * 64 * 3
    data = proc.stdout
    if len(data) < expected_bytes:
        data = data + b"\x00" * (expected_bytes - len(data))
    elif len(data) > expected_bytes:
        data = data[:expected_bytes]
        
    t = torch.frombuffer(data, dtype=torch.uint8).float()
    t = t.reshape(8, 64, 64, 3).permute(3, 0, 1, 2)  # [3, 8, 64, 64]
    t = (t / 127.5) - 1.0  # Normalize to [-1, 1]
    return t

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

    model = model.to(device).half()
    model.train()

    # Freeze all text parameters, keep only multimodal projection & decoders trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if any(x in name for x in ["image_decoder", "audio_decoder", "video_decoder", "img_proj"]):
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

                # Process based on modality type
                ext = safe_file_path.suffix.lower()
                loss = None
                
                if ext in [".png", ".jpg", ".jpeg"]:
                    target = load_image_ffmpeg(safe_file_path).to(device, dtype=torch.float16)
                    # Forward pass
                    out = model(txt_tensor, target_modality="image")
                    gen = out.get("image")
                    if gen is not None:
                        loss = F.l1_loss(gen, target.unsqueeze(0))
                        
                elif ext in [".mp3", ".wav"]:
                    target = load_audio_ffmpeg(safe_file_path).to(device, dtype=torch.float16)
                    # Forward pass
                    out = model(txt_tensor, target_modality="audio")
                    gen = out.get("audio")
                    if gen is not None:
                        loss = F.l1_loss(gen, target.unsqueeze(0))
                        
                elif ext in [".mp4"]:
                    target = load_video_ffmpeg(safe_file_path).to(device, dtype=torch.float16)
                    # Forward pass
                    out = model(txt_tensor, target_modality="video")
                    gen = out.get("video")
                    if gen is not None:
                        loss = F.l1_loss(gen, target.unsqueeze(0))

                if loss is not None:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    opt.step(integrity=1.0)
                    torch.cuda.empty_cache()
                    
                    accum_loss += loss.item()
                    
                    if step % LOG_EVERY == 0:
                        avg_loss = accum_loss / max(1, step % LOG_EVERY if step > 0 else 1)
                        LOG.info(f"Step {step:>4d}/{MAX_STEPS} | loss {avg_loss:.4f} | file: {safe_file_path.name}")
                        accum_loss = 0.0

                    if step > 0 and step % SAVE_EVERY == 0:
                        ckpt_path = CKPT_DIR / f"quillan_v8_multimodal_step_{step}.pt"
                        # Save only the trainable parameters to conserve disk space
                        trainable_sd = {k: v for k, v in model.state_dict().items() if any(x in k for x in ["image_decoder", "audio_decoder", "video_decoder", "img_proj"])}
                        torch.save({"state_dict": trainable_sd, "step": step}, ckpt_path)
                        LOG.info(f"Saved multimodal checkpoint: {ckpt_path.name}")

                    step += 1
            except Exception as e:
                continue

    # Final Save
    final_path = CKPT_DIR / "quillan_v8_multimodal_final.pt"
    trainable_sd = {k: v for k, v in model.state_dict().items() if any(x in k for x in ["image_decoder", "audio_decoder", "video_decoder", "img_proj"])}
    torch.save({"state_dict": trainable_sd, "step": step}, final_path)
    LOG.info(f"Done. Final multimodal weights saved: {final_path.name}")

if __name__ == "__main__":
    main()
