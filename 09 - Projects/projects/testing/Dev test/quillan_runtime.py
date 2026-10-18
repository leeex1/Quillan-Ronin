#!/usr/bin/env python3
"""Shared Quillan-Ronin local runtime utilities.

This module centralizes checkpoint selection, safe state-dict loading, compact
health metadata, and small generation helpers for the v5.3.1 custom runtime.
Original checkpoint files are never modified here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

CUSTOM_CONFIG_KWARGS: Dict[str, Any] = {
    "hidden_dim": 1280,
    "ffn_dim": 3456,
    "num_experts": 34,
    "vocab_size": 8192,
    "text_only": True,
}

# 16→34 expert migration map: old_index -> new_index
_OLD_TO_NEW_EXPERT = {
    0: 30, 1: 6, 2: 3, 3: 7, 4: 9, 5: 12, 6: 13, 7: 27,
    8: 15, 9: 1, 10: 24, 11: 4, 12: 19, 13: 32, 14: 8, 15: 5,
}
_NUM_OLD_EXPERTS = 16
_NUM_NEW_EXPERTS = 34

DEFAULT_CONTEXT_WINDOW = int(os.environ.get("QUILLAN_CONTEXT_WINDOW", "128"))
DEFAULT_BACKEND = os.environ.get("QUILLAN_BACKEND", "custom-v5")


class CheckpointLoadError(RuntimeError):
    """Raised when a checkpoint cannot be safely used for the target runtime."""


@dataclass(frozen=True)
class CheckpointDiagnostics:
    path: str
    size_bytes: int
    fingerprint: str
    total_keys: int
    loaded_keys: int
    ignored_teacher_keys: int
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    shape_mismatches: Tuple[str, ...]


@dataclass(frozen=True)
class LoadedRuntime:
    model: torch.nn.Module
    config: Any
    checkpoint: CheckpointDiagnostics
    device: str
    dtype: str
    parameter_count: int


def configure_logging(level: int = logging.INFO) -> None:
    """Install a simple logger if the application has not configured logging."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def resolve_device(preferred: Optional[str] = None) -> str:
    """Resolve a local runtime device without silently claiming unavailable CUDA."""
    if preferred:
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(device: str, preferred: Optional[str] = None) -> torch.dtype:
    """Choose the runtime dtype. CUDA defaults to fp16 for the 4GB GTX 1050 path."""
    value = (preferred or os.environ.get("QUILLAN_DTYPE") or ("fp16" if device == "cuda" else "fp32")).lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported QUILLAN_DTYPE value: {value}")


def candidate_checkpoints(base_dir: Path = BASE_DIR) -> List[Path]:
    """Return candidate checkpoints in recovery-preferred order."""
    env_path = os.environ.get("QUILLAN_CHECKPOINT")
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            # ── PRIMARY: Full 34-expert seeded checkpoint ────────────────────────
            base_dir / "checkpoints" / "quillan_v8_34_expert_seeded.pt",
            # ── SECONDARY: Complete checkpoint with SwiGLU gate (wgate) ─────────
            base_dir / "checkpoints" / "quillan_v8_ASCENSION_wgate_patched.pt",
            # ── Tertiary trained weights (ASCENSION v5.3.1) ─────────────────────
            base_dir / "checkpoints" / "quillan_v8_ASCENSION_step_50000.pt",
            base_dir / "checkpoints" / "quillan_v8_STABLE_step_50000.pt",
            base_dir / "quillan_v8_clean_base.pth",
            # ── Legacy / fallback names ──────────────────────────────────────────
            base_dir / "checkpoints" / "quillan_v5.3.1_student_inference.pt",
            base_dir / "quillan_v5.3.1_epoch_10_final.pth",
            base_dir / "quillan_v5.3.1_epoch_9_final.pth",
            base_dir / "quillan_v5.3.1_latest_extended.pth",
            base_dir / "quillan_v5.3.1_epoch_8_final.pth",
            base_dir / "quillan_v5.3.1_epoch_7_final.pth",
        ]
    )

    seen = set()
    unique: List[Path] = []
    for path in candidates:
        resolved = path if path.is_absolute() else base_dir / path
        key = str(resolved.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def partial_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a cheap deterministic fingerprint using size, mtime, and edge chunks."""
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(int(stat.st_mtime)).encode("ascii"))
    with path.open("rb") as handle:
        head = handle.read(chunk_size)
        digest.update(head)
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            digest.update(handle.read(chunk_size))
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a full SHA-256 for artifact inventory commands."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            maybe_state = payload.get(key)
            if isinstance(maybe_state, dict):
                return maybe_state
        if all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise CheckpointLoadError("Checkpoint does not contain a recognizable state_dict.")


def clean_student_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
    """Drop teacher EMA tensors and keep only student-runtime tensors."""
    clean: Dict[str, torch.Tensor] = {}
    ignored_teacher = 0
    for key, value in state_dict.items():
        if key.startswith("teacher."):
            ignored_teacher += 1
            continue
        clean[key] = value
    return clean, ignored_teacher


def _migrate_experts_16_to_34(
    checkpoint_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Expand a 16-expert checkpoint state dict to fit a 34-expert model."""
    migrated = {}
    # Start from model's empty state (34-expert shapes), fill in from checkpoint
    for key in model_state:
        model_t = model_state[key]
        if key in checkpoint_state:
            ckpt_t = checkpoint_state[key]
            if ckpt_t.shape == model_t.shape:
                migrated[key] = ckpt_t.clone()
            elif key in ("moe.w1", "moe.w2", "moe.wgate") and ckpt_t.shape[0] == 16:
                new_t = model_t.clone()
                for old_idx, new_idx in _OLD_TO_NEW_EXPERT.items():
                    new_t[new_idx] = ckpt_t[old_idx]
                migrated[key] = new_t
                LOGGER.info("Migrated %s: [16→34] expert weights remapped", key)
            elif key == "moe.router.weight" and ckpt_t.shape[0] == 16:
                new_t = model_t.clone()
                new_t[:_NUM_OLD_EXPERTS] = ckpt_t
                migrated[key] = new_t
                LOGGER.info("Migrated moe.router.weight: [16→34]")
            elif key == "moe.router.bias" and ckpt_t.shape[0] == 16:
                new_t = model_t.clone()
                new_t[:_NUM_OLD_EXPERTS] = ckpt_t
                migrated[key] = new_t
            elif "moe.router" in key and "lora" in key and ckpt_t.shape[-1] == 16 and model_t.shape[-1] == 34:
                new_t = model_t.clone()
                new_t[:, :_NUM_OLD_EXPERTS] = ckpt_t
                migrated[key] = new_t
                LOGGER.info("Migrated %s: [16→34]", key)
            elif "moe.expert_swarms." in key:
                parts = key.split(".")
                if len(parts) >= 4 and parts[2].isdigit():
                    old_e = int(parts[2])
                    new_key = key
                    if old_e < 16:
                        # Remap old swarm slot to new persona slot
                        new_e_idx = _OLD_TO_NEW_EXPERT.get(old_e, old_e)
                        new_key = f"moe.expert_swarms.{new_e_idx}.{parts[3]}"
                    migrated[new_key] = ckpt_t.clone()
                    if new_key != key:
                        LOGGER.info("Migrated swarm %s → %s", key, new_key)
            else:
                migrated[key] = ckpt_t.clone()
        else:
            # Key not in checkpoint: use model's init value
            migrated[key] = model_t.clone()
    return migrated


def validate_state_dict(
    model_state: Dict[str, torch.Tensor],
    checkpoint_state: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], List[str]]:
    """Compare model/checkpoint keys and tensor shapes before loading."""
    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        expected = tuple(model_state[key].shape)
        actual = tuple(checkpoint_state[key].shape)
        if expected != actual:
            mismatches.append(f"{key}: expected {expected}, got {actual}")
    return missing, unexpected, mismatches


def load_checkpoint_for_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool = True,
    map_location: str = "cpu",
) -> CheckpointDiagnostics:
    """Load a checkpoint after validating it against the model state dict."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(str(checkpoint_path))

    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)

    raw_state = _extract_state_dict(payload)
    clean_state, ignored_teacher = clean_student_state_dict(raw_state)
    model_state = model.state_dict()

    # Detect 16→34 expert migration
    needs_migration = False
    for key in ("moe.w1", "moe.w2", "moe.wgate"):
        if key in clean_state and key in model_state:
            if clean_state[key].shape[0] == 16 and model_state[key].shape[0] == 34:
                needs_migration = True
                break

    if needs_migration:
        LOGGER.info("Migrating 16→34 expert checkpoint...")
        clean_state = _migrate_experts_16_to_34(clean_state, model_state)

    missing, unexpected, mismatches = validate_state_dict(model_state, clean_state)

    # Grace keys: architectural additions added after training — initialised from scratch.
    # The SwiGLU gate projection (moe.wgate) was added to v5.3.1 after ASCENSION training.
    _GRACE_MISSING: frozenset = frozenset({"moe.wgate"})

    hard_missing = [k for k in missing if k not in _GRACE_MISSING]
    if strict and (hard_missing or unexpected or mismatches):
        raise CheckpointLoadError(
            "Checkpoint is incompatible with v5.3.1 runtime: "
            f"missing={len(missing)} (grace={len(missing) - len(hard_missing)}), "
            f"unexpected={len(unexpected)}, shape_mismatches={len(mismatches)}"
        )
    if missing:
        LOGGER.warning(
            "Checkpoint missing %d key(s) — using random init for: %s",
            len(missing), missing
        )

    model.load_state_dict(clean_state, strict=False)
    return CheckpointDiagnostics(
        path=str(checkpoint_path),
        size_bytes=checkpoint_path.stat().st_size,
        fingerprint=partial_fingerprint(checkpoint_path),
        total_keys=len(raw_state),
        loaded_keys=len(clean_state),
        ignored_teacher_keys=ignored_teacher,
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        shape_mismatches=tuple(mismatches),
    )


def select_compatible_checkpoint(
    model: torch.nn.Module,
    candidates: Optional[Sequence[Path]] = None,
) -> Tuple[Path, CheckpointDiagnostics]:
    """Pick the first checkpoint that loads without hard mismatches.

    A checkpoint with only grace-period missing keys (e.g. ``moe.wgate``) is
    accepted; those parameters are left at their random initialisation values.
    """
    errors: List[str] = []
    for path in candidates or candidate_checkpoints():
        if not path.exists():
            errors.append(f"{path}: not found")
            continue
        try:
            diagnostics = load_checkpoint_for_model(model, path, strict=True, map_location="cpu")
            LOGGER.info("Loaded checkpoint: %s", path.name)
            return path, diagnostics
        except Exception as exc:  # keep trying candidates; report all failures if none work
            errors.append(f"{path.name}: {exc}")
            continue
    raise CheckpointLoadError(
        "No compatible checkpoint found. Tried:\n" +
        "\n".join(f"  {e}" for e in errors)
    )


def build_custom_runtime(
    *,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    checkpoint_path: Optional[Path] = None,
) -> LoadedRuntime:
    """Instantiate the v5.3.1 custom model with verified trained weights."""
    configure_logging()
    resolved_device = resolve_device(device)
    resolved_dtype = dtype or resolve_dtype(resolved_device)

    from quillan_v8_saturated import QuillanArchConfig, QuillanRoninSovereign

    config = QuillanArchConfig(**CUSTOM_CONFIG_KWARGS)
    model = QuillanRoninSovereign(config)
    if checkpoint_path is not None:
        diagnostics = load_checkpoint_for_model(model, Path(checkpoint_path), strict=True, map_location="cpu")
    else:
        _, diagnostics = select_compatible_checkpoint(model)

    if resolved_dtype != torch.float32:
        model = model.to(dtype=resolved_dtype)
    model = model.to(resolved_device)
    model.eval()
    model.pre_quantize()

    params = sum(param.numel() for param in model.parameters())
    return LoadedRuntime(
        model=model,
        config=config,
        checkpoint=diagnostics,
        device=resolved_device,
        dtype=str(resolved_dtype).replace("torch.", ""),
        parameter_count=params,
    )


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    use_cache: bool = False,
    past_key_values=None,
) -> torch.Tensor:
    """Run the fast logits-only path, bypassing agentic side effects.
    
    When use_cache=True, also returns the KV-cache dict via model's return dict.
    """
    out = model(input_ids, return_hidden=True, use_cache=use_cache, past_key_values=past_key_values)
    if isinstance(out, dict) and "logits" in out:
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    raise RuntimeError("Model forward did not return logits.")


def sample_next_token(logits: torch.Tensor, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0) -> int:
    """Sample one token from final-step logits using bounded top-p/top-k filtering."""
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scores = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        values, _ = torch.topk(scores, min(top_k, scores.shape[-1]))
        scores = scores.masked_fill(scores < values[..., -1, None], -float("inf"))
    if 0 < top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores.masked_fill(remove, -float("inf")))
    probs = F.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_tokens(
    runtime: LoadedRuntime,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> Tuple[List[int], Dict[str, Any]]:
    """Generate tokens using KV-cache prefill + decode loop for low-VRAM hardware."""
    prompt_ids = tokenizer.encode(prompt)
    device = torch.device(runtime.device)

    with torch.inference_mode():
        # --- Prefill: process all prompt tokens at once ---
        window = prompt_ids[-context_window:]
        input_ids = torch.tensor([window], dtype=torch.long, device=device)
        out = forward_logits(runtime.model, input_ids, use_cache=True)
        logits = out["logits"][:, -1, :]
        past_kv = out.get("past_key_values")
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)

        generated_ids = list(window) + [next_id]

        # --- Decode loop: one token at a time with KV-cache ---
        for _ in range(max_new_tokens - 1):
            if next_id == getattr(tokenizer, "eos_token_id", 0):
                break
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = forward_logits(runtime.model, input_ids, use_cache=True, past_key_values=past_kv)
            logits = out["logits"][:, -1, :]
            past_kv = out.get("past_key_values")
            next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
            generated_ids.append(next_id)

    return generated_ids, {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": max(0, len(generated_ids) - len(prompt_ids)),
        "context_window": context_window,
    }


def runtime_health(runtime: Optional[LoadedRuntime]) -> Dict[str, Any]:
    """Return API-friendly runtime metadata."""
    if runtime is None:
        return {"status": "starting", "backend": DEFAULT_BACKEND}
    cuda = torch.cuda.is_available()
    gpu: Dict[str, Any] = {}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        gpu = {
            "name": props.name,
            "total_vram_bytes": props.total_memory,
            "allocated_vram_bytes": torch.cuda.memory_allocated(0),
            "reserved_vram_bytes": torch.cuda.memory_reserved(0),
        }
    return {
        "status": "healthy",
        "backend": DEFAULT_BACKEND,
        "device": runtime.device,
        "dtype": runtime.dtype,
        "parameter_count": runtime.parameter_count,
        "checkpoint": asdict(runtime.checkpoint),
        "config": CUSTOM_CONFIG_KWARGS,
        "gpu": gpu,
    }


def inspect_bitnet_safetensors(model_dir: Path = BASE_DIR / "Quillan-v4.2-model") -> Dict[str, Any]:
    """Inspect the 1B+ safetensors artifact without requiring Transformers."""
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    result: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "config_exists": config_path.exists(),
        "weights_exists": weights_path.exists(),
        "loader_status": "metadata_only",
    }
    if config_path.exists():
        result["config"] = json.loads(config_path.read_text(encoding="utf-8"))
    if not weights_path.exists():
        result["loader_status"] = "missing_weights"
        return result

    try:
        from safetensors import safe_open
    except ImportError:
        result["loader_status"] = "safetensors_not_installed"
        return result

    dtype_counts: Dict[str, int] = {}
    tensor_count = 0
    storage_elements = 0
    sample_tensors: List[Dict[str, Any]] = []
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            tensor = handle.get_tensor(key)
            dtype_name = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            storage_elements += tensor.numel()
            if len(sample_tensors) < 12:
                sample_tensors.append({"name": key, "shape": tuple(tensor.shape), "dtype": dtype_name})

    result.update(
        {
            "weights_size_bytes": weights_path.stat().st_size,
            "fingerprint": partial_fingerprint(weights_path),
            "tensor_count": tensor_count,
            "storage_elements": storage_elements,
            "dtype_counts": dtype_counts,
            "sample_tensors": sample_tensors,
            "loader_status": "metadata_ok_runtime_loader_needed",
        }
    )
    return result


def json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)
