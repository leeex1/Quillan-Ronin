#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process inference engine for the 815M Quillan-Ronin-Mini custom checkpoint.

Duty: load ``exports/quillan-mini-sovereign-bitnet158.safetensors`` into a real
``QuillanRoninSovereign`` (int8 * per-tensor bitnet_scale dequantisation, exact
530/530 strict keys), then drive the FORGE-repaired incremental KV generation
path (``model.generate`` + ``QuillanKVCache``, chunked bounded prefill).

CONTEXT CONTRACT (non-negotiable, board #855/#861/#895/#896):
  * declared num_ctx / max_position_embeddings = 2,097,152 (2M) -- preserved.
  * physically allocatable ACTIVE KV window   = 262,144 (256k).
  * an over-long prompt is REJECTED, never truncated or silently shortened.
  * no Ollama/llama.cpp backend substitution; this is the custom architecture.

No existing server, model, or exporter file is modified by this module.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import safetensors.torch
import torch
from tokenizers import Tokenizer

# ── local imports (model + adapter live in the canonical training tree) ──────
CORE_MODEL_DIR = Path(r"C:\02_QUILLAN\03 - Training & Model")
SCRIPTS_DIR = Path(r"C:\02_QUILLAN\scripts")
for _p in (str(CORE_MODEL_DIR), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quillan_v8_saturated import (  # noqa: E402
    DECLARED_MAX_POSITIONS,
    MAX_DIRECT_CAUSAL_QUERIES,
    QuillanArchConfig,
    QuillanKVCache,
    QuillanRoninSovereign,
)

# ── verified artefact identity (board #894) ──────────────────────────────────
DEFAULT_CHECKPOINT = Path(
    r"C:\02_QUILLAN\exports\quillan-mini-sovereign-bitnet158.safetensors"
)
DEFAULT_TOKENIZER = Path(
    r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct"
    r"\snapshots\7ae557604adf67be50417f59c2c2f167def9a775\tokenizer.json"
)
EXPECTED_SHA256_PREFIX = "53e60d33"

MODEL_NAME = "quillan-ronin-mini"
MODEL_TAG = "quillan-ronin-mini:latest"

# Source dims of the verified donor-derived checkpoint (verify_mighty_mouse.py:41).
HIDDEN_DIM = 896
FFN_DIM = 4864
VOCAB_SIZE = 151936
NUM_LAYERS = 24
NUM_EXPERTS = 34
EGGROLL_RANK = 64

# Declared product capacity is the model constant; nothing here lowers it.
DECLARED_NUM_CTX: int = int(DECLARED_MAX_POSITIONS)  # 2_000_000

# 256k ACTIVE window: >= the full ~171k system prompt (#896), so it is never
# truncated. 131072 (the class default) is TOO SMALL for that prompt.
ACTIVE_WINDOW_TOKENS = 262_144

# Qwen2.5 eos (151645 <|im_end|>, 151643 <|endoftext|>) plus the arch's own
# legacy 50256 sentinel used inside model.generate().
EOS_TOKEN_IDS: Tuple[int, ...] = (151645, 151643, 50256)


class ContextWindowError(ValueError):
    """Raised when a request cannot fit the ACTIVE window. Never truncate."""


class CheckpointError(RuntimeError):
    """Raised when the checkpoint does not match its verified identity."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class GenerateResult:
    model: str
    response: str
    done: bool = True
    done_reason: str = "length"
    context: List[int] = field(default_factory=list)
    created_at: str = ""
    total_duration_ns: int = 0
    generation_duration_ns: int = 0
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_tokens_per_second: float = 0.0
    warnings: List[str] = field(default_factory=list)
    context_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "created_at": self.created_at or _iso_now(),
            "response": self.response,
            "done": self.done,
            "done_reason": self.done_reason,
            "context": self.context,
            # Ollama-shaped fields. Prefill and decode run in ONE model.generate()
            # call, so prompt_eval_duration is not separately measurable; it is
            # reported as 0 and the exact combined duration is exposed below.
            "total_duration": self.total_duration_ns,
            "load_duration": 0,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": 0,
            "eval_count": self.eval_count,
            "eval_duration": self.generation_duration_ns,
            "eval_tokens_per_second": round(self.eval_tokens_per_second, 4),
            "warnings": self.warnings,
            "context_policy": self.context_policy,
        }


class QuillanServingEngine:
    """Loads the verified checkpoint once; owns the KV cache and tokenizer."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        tokenizer_path: Path = DEFAULT_TOKENIZER,
        active_window: int = ACTIVE_WINDOW_TOKENS,
        device: str = "cpu",
        verify_checksum: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.tokenizer_path = Path(tokenizer_path)
        self.active_window = int(active_window)
        self.device = str(device)
        if not self.checkpoint.exists():
            raise CheckpointError(f"checkpoint missing: {self.checkpoint}")
        if not self.tokenizer_path.exists():
            raise CheckpointError(f"tokenizer missing: {self.tokenizer_path}")
        if self.active_window > DECLARED_NUM_CTX:
            raise ValueError("active window cannot exceed declared 2M positions")

        t0 = time.perf_counter()
        self.sha256 = _sha256_file(self.checkpoint) if verify_checksum else ""
        if self.sha256 and not self.sha256.startswith(EXPECTED_SHA256_PREFIX):
            raise CheckpointError(
                f"sha256 {self.sha256} does not match verified prefix "
                f"{EXPECTED_SHA256_PREFIX} (board #894)"
            )
        self.checkpoint_bytes = self.checkpoint.stat().st_size

        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._load_model()
        self.load_duration_ns = int((time.perf_counter() - t0) * 1e9)

    # ── checkpoint -> live model ────────────────────────────────────────────
    def _load_model(self) -> None:
        packed = safetensors.torch.load_file(str(self.checkpoint), device="cpu")
        self.packed_tensor_count = len(packed)
        scales = {
            k[: -len(".bitnet_scale")]: v
            for k, v in packed.items()
            if k.endswith(".bitnet_scale")
        }
        raw = {k: v for k, v in packed.items() if not k.endswith(".bitnet_scale")}
        self.int8_tensor_count = sum(1 for v in raw.values() if v.dtype == torch.int8)
        self.bitnet_scale_count = len(scales)

        sd: Dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            if value.dtype == torch.int8:
                if key not in scales:
                    raise CheckpointError(f"missing bitnet_scale for {key}")
                sd[key] = value.float() * scales[key].float()
            else:
                sd[key] = value.float()
        del packed, raw, scales
        gc.collect()

        cfg = QuillanArchConfig(
            text_only=True,
            multimodal=False,
            hidden_dim=HIDDEN_DIM,
            ffn_dim=FFN_DIM,
            vocab_size=VOCAB_SIZE,
            num_experts=NUM_EXPERTS,
            num_experts_active=4,
            top_k=4,
            use_lora=True,
            device=self.device,
            eggroll_rank=EGGROLL_RANK,
            max_position_embeddings=DECLARED_NUM_CTX,
            kv_cache_capacity=self.active_window,
            kv_cache_dtype="fp32",  # fp16 cache would mismatch fp32 BitLinear matmuls
            prefill_chunk_size=512,
        )
        self.model = QuillanRoninSovereign(cfg).to(self.device).eval()
        missing, unexpected = self.model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            raise CheckpointError(
                f"strict load failed: missing={len(missing)} unexpected={len(unexpected)}"
            )
        self.loaded_keys = len(sd)
        del sd
        gc.collect()

        self.param_count = int(sum(p.numel() for p in self.model.parameters()))
        if self.param_count != 815_396_270:
            raise CheckpointError(
                f"param count {self.param_count} != verified 815,396,270"
            )
        self.kv_bytes_per_token = int(
            self.model.kv_cache_bytes_per_token(torch.float32)
        )
        self.active_kv_bytes = self.kv_bytes_per_token * self.active_window
        self.declared_kv_bytes = self.kv_bytes_per_token * DECLARED_NUM_CTX

    # ── tokenisation ────────────────────────────────────────────────────────
    def encode(self, text: str) -> List[int]:
        return [
            int(t) for t in self.tokenizer.encode(text, add_special_tokens=False).ids
        ]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    # ── guards ──────────────────────────────────────────────────────────────
    def _guard(self, prompt_ids: Sequence[int], num_predict: int) -> None:
        needed = len(prompt_ids) + max(0, int(num_predict))
        if needed > self.active_window:
            raise ContextWindowError(
                f"request needs {needed} positions but ACTIVE KV window is "
                f"{self.active_window}. Declared num_ctx stays {DECLARED_NUM_CTX}; "
                f"the prompt is NOT truncated. Split the request or raise "
                f"ACTIVE_WINDOW_TOKENS only if host RAM permits "
                f"({self.kv_bytes_per_token} B/token)."
            )

    def _context_policy(self) -> Dict[str, Any]:
        return {
            "declared_num_ctx": DECLARED_NUM_CTX,
            "active_window_tokens": self.active_window,
            "full_declared_kv_gib": round(self.declared_kv_bytes / 1024**3, 3),
            "active_kv_gib": round(self.active_kv_bytes / 1024**3, 3),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "prompt_truncated": False,
            "backend": "pytorch-custom-arch",
            "max_direct_causal_queries": MAX_DIRECT_CAUSAL_QUERIES,
        }

    # ── public API: non-streaming ───────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        num_predict: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop: Optional[Sequence[str]] = None,
        chunk_size: Optional[int] = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> GenerateResult:
        num_predict = max(0, int(num_predict))
        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt produced zero tokens")
        self._guard(prompt_ids, num_predict)

        t0 = time.perf_counter()
        tokens, _cache = self.model.generate(
            prompt_ids,
            max_tokens=num_predict,
            temp=float(temperature),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            return_cache=True,
            chunk_size=chunk_size,
        )
        gen_ns = int((time.perf_counter() - t0) * 1e9)

        stop_ids = set(EOS_TOKEN_IDS)
        cut = len(tokens)
        reason = "length"
        for idx, tok in enumerate(tokens):
            if tok in stop_ids:
                cut, reason = idx, "stop"
                break
        if cut == len(tokens) and stop:
            text = self.decode(tokens)
            for needle in stop:
                pos = text.find(needle)
                if pos >= 0:
                    cut = self._prefix_token_cut(tokens, text[:pos])
                    reason = "stop"
                    break
        emitted = tokens[:cut]
        response = self.decode(emitted)
        seconds = gen_ns / 1e9
        return GenerateResult(
            model=MODEL_TAG,
            response=response,
            done=True,
            done_reason=reason,
            context=list(prompt_ids) + list(emitted),
            created_at=_iso_now(),
            total_duration_ns=gen_ns + self.load_duration_ns,
            generation_duration_ns=gen_ns,
            prompt_eval_count=len(prompt_ids),
            eval_count=len(emitted),
            eval_tokens_per_second=(len(emitted) / seconds) if seconds > 0 else 0.0,
            warnings=list(warnings or []),
            context_policy=self._context_policy(),
        )

    def _prefix_token_cut(self, tokens: Sequence[int], prefix: str) -> int:
        best = len(tokens)
        for k in range(len(tokens), 0, -1):
            if self.decode(tokens[:k]).startswith(prefix):
                best = k
                break
        return best

    # ── public API: token streaming (cache-reuse incremental) ───────────────
    def stream_generate(
        self,
        prompt: str,
        num_predict: int = 64,
        temperature: float = 0.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        stop: Optional[Sequence[str]] = None,
        chunk_size: Optional[int] = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> Iterator[Dict[str, Any]]:
        num_predict = max(0, int(num_predict))
        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt produced zero tokens")
        self._guard(prompt_ids, num_predict)

        prompt_ids = list(prompt_ids)
        cache: Optional[QuillanKVCache] = None
        emitted: List[int] = []
        pieces: List[str] = []
        started = time.perf_counter()
        reason = "length"
        stop_ids = set(EOS_TOKEN_IDS)

        for _ in range(num_predict):
            continuation, cache = self.model.generate(
                prompt_ids,
                max_tokens=1,
                temp=float(temperature),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                cache=cache,
                return_cache=True,
                chunk_size=chunk_size,
            )
            if not continuation:
                break
            tok = int(continuation[0])
            emitted.append(tok)
            prompt_ids.append(tok)
            piece = self.decode([tok])
            pieces.append(piece)
            yield {
                "model": MODEL_TAG,
                "created_at": _iso_now(),
                "response": piece,
                "done": False,
            }
            joined = "".join(pieces)
            if tok in stop_ids or any(s and s in joined for s in (stop or [])):
                reason = "stop"
                break

        gen_ns = int((time.perf_counter() - started) * 1e9)
        seconds = gen_ns / 1e9
        yield {
            "model": MODEL_TAG,
            "created_at": _iso_now(),
            "response": "",
            "done": True,
            "done_reason": reason,
            "context": prompt_ids,
            "total_duration": gen_ns,
            "eval_count": len(emitted),
            "eval_duration": gen_ns,
            "eval_tokens_per_second": round(len(emitted) / seconds, 4)
            if seconds
            else 0.0,
            "warnings": list(warnings or []),
            "context_policy": self._context_policy(),
        }

    # ── observability ───────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": MODEL_TAG,
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint_bytes,
            "sha256": self.sha256,
            "params": self.param_count,
            "loaded_keys": self.loaded_keys,
            "packed_tensors": self.packed_tensor_count,
            "int8_tensors": self.int8_tensor_count,
            "bitnet_scale_tensors": self.bitnet_scale_count,
            "device": self.device,
            "torch_version": torch.__version__,
            "load_duration_ns": self.load_duration_ns,
            "context_policy": self._context_policy(),
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "license": "sovereign-quillan-ronin",
            "modelfile": (
                "FROM exports/quillan-mini-sovereign-bitnet158.safetensors\n"
                "ARCHITECTURE QuillanRoninSovereign (custom, PyTorch-native)\n"
                f"PARAMETER num_ctx {DECLARED_NUM_CTX}  # DECLARED; not physically allocatable\n"
                f"PARAMETER active_window_tokens {self.active_window}  # ACTUAL KV allocation\n"
                "PARAMETER prompt_truncation false\n"
            ),
            "details": {
                "format": "safetensors",
                "family": "quillan-ronin-mini",
                "parameter_size": "815M",
                "quantization_level": "bitnet1.58-int8+per-tensor-scale",
                "num_ctx_declared": DECLARED_NUM_CTX,
                "active_window_tokens": self.active_window,
            },
            "parameters": (
                f"num_ctx {DECLARED_NUM_CTX}\n"
                f"active_window_tokens {self.active_window}\n"
                "prompt_truncation false\n"
            ),
            "digest": f"sha256:{self.sha256}",
            "context_policy": self._context_policy(),
        }


def build_engine_from_env() -> QuillanServingEngine:
    """Host/port/context knobs are read by server.py; engine uses verified defaults."""
    return QuillanServingEngine()


__all__ = [
    "ACTIVE_WINDOW_TOKENS",
    "DECLARED_NUM_CTX",
    "MODEL_NAME",
    "MODEL_TAG",
    "ContextWindowError",
    "CheckpointError",
    "QuillanServingEngine",
    "GenerateResult",
    "build_engine_from_env",
]
