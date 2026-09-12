#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.4-ONI — SOVEREIGN MASTER ORCHESTRATOR
=============================================================================
Unified orchestration layer connecting:
- Canonical v5.4-ONI architecture (QuillanRoninOni)
- Hardened autoregressive decoding (SovereignInferenceEngine)
- OpenAI-compatible REST server (api.py)
- Throne Deliberation pipeline (34 Council Experts + Quality Exit Gates)
=============================================================================
"""

import os
import sys
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

if not torch.cuda.is_available():
    torch.set_num_threads(min(2, os.cpu_count() or 2))
    torch.set_num_interop_threads(min(2, os.cpu_count() or 2))
try:
    import psutil
    p = psutil.Process()
    if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

# Ensure scripts directory and canonical oni directory are on sys.path
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ONI_DIR = Path(r"C:\02_QUILLAN\09 - Projects\projects\oni")
for _p in [str(_SCRIPTS_DIR), str(_ONI_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from sovereign_inference_engine import SovereignInferenceEngine, SovereignTokenizer, SamplingParams

LOGGER = logging.getLogger("quillan.orchestrator")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

_ORCHESTRATOR_INSTANCE: Optional["SovereignMasterOrchestrator"] = None
_ORCHESTRATOR_LOCK = threading.Lock()


class SovereignMasterOrchestrator:
    """
    Central brain orchestrating token generation, Throne deliberation,
    and inference execution across the Quillan-Ronin v5.4-ONI architecture.
    """
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = SovereignTokenizer("gpt2")
        self.checkpoint_path = checkpoint_path or self._discover_latest_checkpoint()
        self.engine: Optional[SovereignInferenceEngine] = None
        self._init_engine()

    def _discover_latest_checkpoint(self) -> Optional[str]:
        """Auto-detects highest quality checkpoint from standard repository locations."""
        candidates = [
            Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_frontier_v2_best.pt"),
            Path(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_frontier_v2_latest.pt"),
            Path(r"C:\02_QUILLAN\checkpoints\production_export\quillan_ronin_v531_sovereign_production.pt"),
            Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_weights.pt"),
            Path(r"C:\02_QUILLAN\checkpoints\checkpoints_oni\quillan_oni_latest.pt"),
        ]
        for c in candidates:
            if c.is_file():
                LOGGER.info("Discovered canonical checkpoint: %s", c)
                return str(c)
        LOGGER.warning("No pretrained checkpoint found; initializing randomly initialized sovereign model.")
        return None

    def _init_engine(self):
        def model_factory() -> QuillanRoninOni:
            cfg = QuillanOniConfig(
                n_layer=6,
                hidden_dim=1024,
                max_seq_len=512,
                router_mode="dense_pull",
                num_experts=34,
            )
            return QuillanRoninOni(cfg)

        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            try:
                LOGGER.info("Loading sovereign inference engine from: %s", self.checkpoint_path)
                self.engine = SovereignInferenceEngine.load_from_checkpoint(
                    model_factory=model_factory,
                    checkpoint_path=self.checkpoint_path,
                    device=self.device,
                    strict=False,
                )
            except Exception as e:
                LOGGER.error("Failed to load checkpoint (%s). Falling back to direct model initialization: %s", self.checkpoint_path, e)
                model = model_factory()
                self.engine = SovereignInferenceEngine(model=model, tokenizer=self.tokenizer, device=self.device)
        else:
            model = model_factory()
            self.engine = SovereignInferenceEngine(model=model, tokenizer=self.tokenizer, device=self.device)

    def generate_full_sovereign_response(
        self,
        user_query: str,
        params: Optional[SamplingParams] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Formats user query into sovereign prompt template and generates response
        through autoregressive decoding with repetition penalties and stop-token filtering.
        """
        if params is None:
            params = SamplingParams(
                max_new_tokens=max_tokens if max_tokens is not None else 256,
                temperature=temperature if temperature is not None else 0.65,
                top_k=top_k if top_k is not None else 40,
                top_p=top_p if top_p is not None else 0.85,
                repetition_penalty=kwargs.get("repetition_penalty", 1.20),
                frequency_penalty=kwargs.get("frequency_penalty", 0.30),
            )

        # Standard Sovereign Prompt Template
        formatted_prompt = f"<|user|>\n{user_query.strip()}\n<|assistant|>\n"
        
        response = self.engine.generate(formatted_prompt, params=params)
        
        # Clean response string: strip trailing stop strings and whitespace
        clean_text = response.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        return clean_text

    def deliberate(self, user_query: str, max_rounds: int = 2) -> Dict[str, Any]:
        """
        Executes deep Throne Deliberation (34-council arbitration mesh + quality gates).
        """
        formatted_prompt = f"<|user|>\n{user_query.strip()}\n<|assistant|>\n"
        tokens = self.tokenizer.encode(formatted_prompt)
        res = self.engine.model.deliberate(tokens, max_rounds=max_rounds)
        gen_tokens = res.get("tokens", [])
        text = self.tokenizer.decode(gen_tokens).split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        return {
            "response": text,
            "trace": res.get("trace", {}),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "model_version": "v5.4-ONI",
            "checkpoint": self.checkpoint_path,
            "device": str(self.device),
            "num_experts": 34,
            "architecture": "Quillan-Ronin Omni-Fractal Sovereign",
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Telemetry health status check."""
        status = self.get_status()
        status["healthy"] = True
        return status

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models in OpenAI-compatible format."""
        return [
            {
                "id": "quillan-ronin-v5.4-oni",
                "object": "model",
                "created": 1777000000,
                "owned_by": "quillan-foundation",
                "description": "34-Expert MoE Sovereign Model (479M params, 9-Vector Prism, 4-Stage Output)",
            }
        ]


def get_master_orchestrator() -> SovereignMasterOrchestrator:
    """Thread-safe singleton accessor for SovereignMasterOrchestrator."""
    global _ORCHESTRATOR_INSTANCE
    if _ORCHESTRATOR_INSTANCE is None:
        with _ORCHESTRATOR_LOCK:
            if _ORCHESTRATOR_INSTANCE is None:
                _ORCHESTRATOR_INSTANCE = SovereignMasterOrchestrator()
    return _ORCHESTRATOR_INSTANCE


if __name__ == "__main__":
    orchestrator = get_master_orchestrator()
    print("[+] Sovereign Master Orchestrator initialized successfully!")
    print(f"[+] Status: {orchestrator.get_status()}")
    test_out = orchestrator.generate_full_sovereign_response("Hello, Quillan!")
    print(f"[+] Sample generation:\n{test_out}")
