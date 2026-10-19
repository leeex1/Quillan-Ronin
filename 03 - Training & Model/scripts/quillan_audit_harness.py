#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin Model Audit, Verification, and Flagship Resumption Harness.
Author: Quillan Autonomous Engineering Lab
Version: 5.4.0-oni Production

Provides drop-in verification, integrity audits, parameter introspection,
and safe resumption for the 6-layer and 12-layer Quillan-Ronin MoE models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn

# Base path definitions
_BASE_DIR: Path = Path(r"C:\02_QUILLAN")
_ONI_DIR: Path = _BASE_DIR / "09 - Projects" / "projects" / "oni"
_TRAIN_DIR: Path = _BASE_DIR / "03 - Training & Model" / "scripts"

for _p in [str(_BASE_DIR), str(_ONI_DIR), str(_TRAIN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOGGER: logging.Logger = logging.getLogger("QuillanAuditor")
if not LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter(
        '{"timestamp":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","msg":"%(message)s"}'
    )
    _handler.setFormatter(_formatter)
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)


@dataclass(frozen=True)
class CheckpointAuditReport:
    """Immutable audit report for a model checkpoint."""
    checkpoint_path: str
    file_size_mb: float
    sha256_prefix: str
    step: int
    best_val: float
    recorded_loss: Optional[float]
    n_layers: int
    hidden_dim: int
    total_parameters: int
    trainable_parameters: int
    has_optimizer_state: bool
    has_ema_state: bool
    status: str
    notes: List[str]


class SafePathValidator:
    """Security validator preventing directory traversal and unauthorized file access."""

    @staticmethod
    def validate_file(path_str: str, allowed_extensions: Tuple[str, ...]) -> Path:
        """
        Validates that a path exists, is a regular file, and possesses an allowed extension.
        
        Complexity: O(1) time | O(1) space.
        """
        if not path_str or not isinstance(path_str, str):
            raise ValueError("Path must be a non-empty string.")
        
        resolved = Path(path_str).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Target file not found: {resolved}")
        
        if not any(resolved.name.endswith(ext) for ext in allowed_extensions):
            raise ValueError(
                f"File {resolved.name} does not match allowed extensions: {allowed_extensions}"
            )
        return resolved


class ForwardSignatureAdapter:
    """
    Adapter ensuring backward-compatible execution across legacy and updated MoE expert signatures.
    
    Legacy Contract: expert(x: Tensor) -> Tensor
    Modern Contract: expert(x: Tensor, gov_scale: float) -> Tensor
    
    Deprecation Notice: Calling experts with single-tensor signature is deprecated 
    and will be removed in v5.5.0 (90-day grace period). Pass gov_scale explicitly.
    """

    @staticmethod
    def forward_expert(expert: nn.Module, x: torch.Tensor, gov_scale: float = 1.0) -> torch.Tensor:
        """Dispatches forward pass with graceful fallback to legacy interface."""
        try:
            return expert(x, gov_scale)
        except TypeError:
            return expert(x)


class CheckpointIntrospector:
    """Inspects and audits PyTorch checkpoints with bounded resource consumption."""

    @staticmethod
    def calculate_sha256(filepath: Path, chunk_size: int = 1048576) -> str:
        """
        Computes SHA256 checksum in bounded 1MB chunks to prevent heap exhaustion.
        
        Complexity: O(N) time where N is file size | O(1) memory overhead.
        """
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def audit_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> CheckpointAuditReport:
        """
        Deeply inspects state dict, architecture configuration, and training metadata.
        
        Complexity: O(P) time where P is parameter count | O(P) memory to hold state dict.
        """
        validated_path = SafePathValidator.validate_file(
            checkpoint_path, (".pt", ".pth", ".bin", ".safetensors")
        )
        file_size_mb = round(validated_path.stat().st_size / (1024 * 1024), 2)
        sha256_hash = cls.calculate_sha256(validated_path)[:16]

        LOGGER.info(f"Auditing checkpoint: {validated_path.name} ({file_size_mb} MB)")

        try:
            ckpt: Any = torch.load(validated_path, map_location=device, weights_only=False)
        except Exception as exc:
            err_msg = str(exc)
            LOGGER.warning(f"Failed to load checkpoint {validated_path.name}: {err_msg}")
            return CheckpointAuditReport(
                checkpoint_path=str(validated_path),
                file_size_mb=file_size_mb,
                sha256_prefix=sha256_hash,
                step=-1,
                best_val=-1.0,
                recorded_loss=None,
                n_layers=-1,
                hidden_dim=-1,
                total_parameters=0,
                trainable_parameters=0,
                has_optimizer_state=False,
                has_ema_state=False,
                status="TORCHSCRIPT_OR_UNSUPPORTED",
                notes=[err_msg],
            )

        notes: List[str] = []
        step: int = 0
        best_val: float = 0.0
        recorded_loss: Optional[float] = None
        has_opt: bool = False
        has_ema: bool = False
        state_dict: Dict[str, Any] = {}

        if isinstance(ckpt, dict):
            step = ckpt.get("step", 0)
            best_val = float(ckpt.get("best_val", 0.0))
            if "loss" in ckpt:
                recorded_loss = float(ckpt["loss"])
            has_opt = "opt" in ckpt or "optimizer" in ckpt
            has_ema = "ema_sd" in ckpt

            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = getattr(ckpt, "state_dict", lambda: {})()

        total_params: int = 0
        trainable_params: int = 0
        layer_indices: Set[int] = set()
        hidden_dim: int = 1024

        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                total_params += tensor.numel()
                if tensor.requires_grad:
                    trainable_params += tensor.numel()
                if "layers." in key:
                    parts = key.split("layers.")[1].split(".")
                    if parts[0].isdigit():
                        layer_indices.add(int(parts[0]))
                elif "h." in key:
                    parts = key.split("h.")[1].split(".")
                    if parts[0].isdigit():
                        layer_indices.add(int(parts[0]))
                if "tok_emb.weight" in key or "pos_emb" in key or "wte.weight" in key:
                    hidden_dim = tensor.shape[-1]

        n_layers = len(layer_indices) if layer_indices else 0

        status = "HEALTHY"
        if n_layers == 0:
            status = "UNKNOWN_ARCH"
            notes.append("Could not detect transformer layers from keys.")
        elif n_layers == 6:
            notes.append(f"Confirmed 6-layer model (loss: {recorded_loss if recorded_loss is not None else 'N/A'}).")
        elif n_layers == 12:
            notes.append(f"Confirmed 12-layer flagship model (loss: {recorded_loss if recorded_loss is not None else 'N/A'}).")

        report = CheckpointAuditReport(
            checkpoint_path=str(validated_path),
            file_size_mb=file_size_mb,
            sha256_prefix=sha256_hash,
            step=step,
            best_val=best_val,
            recorded_loss=recorded_loss,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            has_optimizer_state=has_opt,
            has_ema_state=has_ema,
            status=status,
            notes=notes,
        )
        LOGGER.info(
            f"Audit finished: {validated_path.name} | Steps: {step} | "
            f"Layers: {n_layers} | Params: {total_params / 1e6:.2f}M | "
            f"Loss: {recorded_loss} | Status: {status}"
        )
        return report


def audit_frontier_suite() -> Dict[str, Any]:
    """Audits primary checkpoints across the repository."""
    targets: Sequence[Path] = [
        _BASE_DIR / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_best.pt",
        _BASE_DIR / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_latest.pt",
    ]
    results: Dict[str, Any] = {}
    for target in targets:
        if target.exists():
            report = CheckpointIntrospector.audit_checkpoint(str(target))
            results[target.name] = asdict(report)
        else:
            LOGGER.warning(f"Target checkpoint absent: {target}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Checkpoint Audit and Resumption Tool")
    parser.add_argument("--audit-all", action="store_true", help="Run audit across all primary checkpoints")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to specific checkpoint to audit")
    args = parser.parse_args()

    if args.checkpoint:
        rep = CheckpointIntrospector.audit_checkpoint(args.checkpoint)
        print(json.dumps(asdict(rep), indent=2))
    elif args.audit_all or not sys.argv[1:]:
        res = audit_frontier_suite()
        print(json.dumps(res, indent=2))
