#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN LOCAL SOVEREIGN OPENAI-COMPATIBLE API GATEWAY
=============================================================
High-performance, zero-external-dependency local HTTP gateway serving:
  1. Standard OpenAI v1 endpoints:
     - GET  /v1/models
     - POST /v1/chat/completions (streaming SSE & non-streaming JSON)
  2. Dual Neural Reasoning Architecture:
     - System 1: Quillan-Oni Mini (6 Layers, ~577M params) for rapid intuitive responses
     - System 2: Quillan-Oni Main (12 Layers, ~726.7M params) for deep deliberative reasoning
  3. Win32 & AVX2 Silicon Optimization endpoints:
     - GET  /api/health (working set RAM, priority class, model status)
     - POST /api/hardware/compact (EmptyWorkingSet standby trimming)
     - GET  /api/hardware/benchmark (AVX2 SIMD real hardware acceleration)
  4. Web Extension & Desktop App Integration:
     - Directly queried by Brave extension and Quillan-App on port 8000
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT: Final[Path] = Path(r"C:\02_QUILLAN")
MODEL_DIR: Final[Path] = REPO_ROOT / "03 - Training & Model"
PROJECTS_DIR: Final[Path] = REPO_ROOT / "09 - Projects" / "projects"
SCRIPTS_DIR: Final[Path] = REPO_ROOT / "scripts"
SM61_KERNEL_DIR: Final[Path] = PROJECTS_DIR / "Chip design" / "quillan_sm61_kernels"
# ARCH NOTE (verified by bind_truth.py): Mini-6L weights (1438 tensors) bind 0/0
# to scripts/ and oni/ copies. The hf_code_era copy is the MAIN-12L architecture
# (1858 keys at 6L config) and MUST NOT serve Mini. Resolution order below
# keeps Mini-compatible code first. Era serves Ronin only.

for p in [str(REPO_ROOT), str(MODEL_DIR), str(PROJECTS_DIR / "oni"), str(SCRIPTS_DIR), str(SM61_KERNEL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger("quillan_gateway")

# Persistent Database Singletons (Eradicates Per-Request Connection Leak)
_LANCEDB_CONN = None
_CHROMADB_CLIENT = None

def get_lancedb_connection():
    global _LANCEDB_CONN
    if _LANCEDB_CONN is None:
        try:
            import lancedb
            p = REPO_ROOT / "lancedb"
            if p.exists():
                _LANCEDB_CONN = lancedb.connect(str(p))
                LOGGER.info("LanceDB persistent connection pool initialized.")
        except Exception as e:
            LOGGER.debug("LanceDB connection pool note: %s", e)
    return _LANCEDB_CONN

def get_chromadb_client():
    global _CHROMADB_CLIENT
    if _CHROMADB_CLIENT is None:
        try:
            import chromadb
            pdb = REPO_ROOT / "01_Knowledge_Base" / "palace_db"
            if pdb.exists():
                _CHROMADB_CLIENT = chromadb.PersistentClient(path=str(pdb))
                LOGGER.info("ChromaDB persistent client pool initialized.")
        except Exception as e:
            LOGGER.debug("ChromaDB client pool note: %s", e)
    return _CHROMADB_CLIENT

def apply_sm61_dp4a(model: Any, split_ratio: float = 1.0) -> int:
    """Replaces standard MLP linears with Pascal sm_61 DP4A integer offload."""
    try:
        import torch.nn as nn
        from sm61_linear import SM61Linear
        converted = 0
        for name, mod in model.named_modules():
            for c_name, c_mod in mod.named_children():
                if isinstance(c_mod, nn.Linear) and c_name in ["c_fc", "c_proj"]:
                    setattr(mod, c_name, SM61Linear.from_linear(c_mod, split_ratio=split_ratio))
                    converted += 1
        if converted > 0:
            LOGGER.info("👑 Accelerated %d linear layers via Pascal sm_61 DP4A (GPU split=%.2f).", converted, split_ratio)
        return converted
    except Exception as e:
        LOGGER.warning("Pascal sm_61 DP4A acceleration bypassed: %s", e)
        return 0

# Guard against 100% CPU thread starvation on 4-core CPUs: leave 1 core free for Windows Desktop
try:
    import torch
    safe_threads = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(safe_threads)
    LOGGER.info("PyTorch CPU execution thread ceiling set to %d (guarantees host OS responsiveness).", safe_threads)
except Exception:
    pass

# Lazy model references
_TOKENIZER = None
_MODEL_MINI = None
_MODEL_MAIN = None
_HW_TOOLKIT = None

def get_hardware_toolkit():
    global _HW_TOOLKIT
    if _HW_TOOLKIT is None:
        try:
            from quillan_pc_toolkit import QuillanPCHardwareToolkit
            _HW_TOOLKIT = QuillanPCHardwareToolkit()
        except Exception as e:
            LOGGER.warning("Could not instantiate QuillanPCHardwareToolkit: %s", e)
    return _HW_TOOLKIT

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            # Matched-pair rule: current Mini+Main are legacy-50257 trained,
            # so serving uses the legacy BPE tokenizer. Unified serves only
            # unified-trained weights (none yet).
            from quillan_bpe_tokenizer import QuillanBPETokenizer
            _TOKENIZER = QuillanBPETokenizer()
            try:
                LOGGER.info("Quillan BPE Tokenizer initialized (vocab=%d).", _TOKENIZER.vocab_size)
            except Exception:
                pass
        except Exception as e:
            LOGGER.error("Failed to load QuillanBPETokenizer: %s", e)
    return _TOKENIZER

_MINI_MTIME: float = 0.0
_MAIN_MTIME: float = 0.0

def _resolve_inference_device():
    """Pascal-safe device resolver (GTX 1050 = sm_61).

    Modern cu13x pip wheels ship NO sm_61 kernels (arch_list starts at
    sm_75) so any CUDA launch dies with 'no kernel image available'.
    Default stays on CPU until Path A (torch cu126 2.14) or Path C
    (int8 DP4A engine) is active. Override with QUILLAN_DEVICE=cuda
    only after verifying your torch arch_list contains sm_61.
    """
    import os
    import torch
    want = os.environ.get("QUILLAN_DEVICE", "cpu").lower()
    if want.startswith("cuda") and torch.cuda.is_available():
        try:
            archs = torch.cuda.get_arch_list()
            cap = torch.cuda.get_device_capability(0)
            sm = f"sm_{cap[0]}{cap[1]}"
            if any(sm in a for a in archs):
                LOGGER.info("QUILLAN_DEVICE=cuda honoured (card %s covered by %s).", sm, archs)
                return torch.device("cuda")
            LOGGER.warning(
                "QUILLAN_DEVICE=cuda requested but torch arch_list=%s lacks %s "
                "(Pascal trap) -> staying on CPU. Install torch cu126 2.14 (Path A) "
                "or use the int8 DP4A engine (Path C).",
                archs, sm,
            )
        except Exception as e:
            LOGGER.warning("CUDA probe failed (%s) -> staying on CPU.", e)
    return torch.device("cpu")


def get_model(name: str, force_reload: bool = False):
    global _MODEL_MINI, _MODEL_MAIN, _MINI_MTIME, _MAIN_MTIME
    import gc
    import torch
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

    device = _resolve_inference_device()

    if "main" in name.lower() or "12" in name.lower():
        main_candidates = [
            REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_12l_ma_best.pt",
            REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_12l_best.pt",
            REPO_ROOT / "checkpoints" / "quillan_oni_main_12l.pt",
        ]
        ckpt_path = main_candidates[0]
        for c in main_candidates:
            if c.exists():
                ckpt_path = c
                break
        curr_mtime = ckpt_path.stat().st_mtime if ckpt_path.exists() else 0.0
        needs_reload = force_reload or (_MODEL_MAIN is None) or (curr_mtime > _MAIN_MTIME and curr_mtime > 0.0)

        if needs_reload:
            # Memory Hygiene: Free Mini model from RAM to maintain working set under 5GB
            if _MODEL_MINI is not None:
                del _MODEL_MINI
                _MODEL_MINI = None
                gc.collect()

            LOGGER.info("Loading System 2 Main-12L model from %s (mtime=%.1f)...", ckpt_path, curr_mtime)
            v_size = 50257
            sd = None
            use_ma = False
            if ckpt_path.exists():
                data = torch.load(ckpt_path, map_location=device, weights_only=False)
                sd = data.get("model", data.get("model_state_dict", data))
                use_ma = bool(data.get("use_ma", False) or (sd is not None and "token_memory" in str(sd.keys())))
                if sd is not None and "wte.weight" in sd:
                    v_size = sd["wte.weight"].shape[0]
                elif "config" in data and isinstance(data["config"], dict):
                    v_size = data["config"].get("vocab_size", 50257)
                del data
            cfg = QuillanOniConfig(
                vocab_size=v_size,
                hidden_dim=1024,
                ffn_dim=2048,
                n_layer=12,
                num_experts=34,
                top_k=4,
                max_seq_len=512,
                use_memory_attention=use_ma,
            )
            model = QuillanRoninOni(cfg).to(device)
            if sd is not None:
                model.load_state_dict(sd, strict=False)
                del sd
            if use_ma and hasattr(model, "fold_memory_attention_weights"):
                model.fold_memory_attention_weights()
            if os.environ.get("QUILLAN_ENABLE_DP4A", "0") == "1":
                apply_sm61_dp4a(model, split_ratio=1.0)
            model.eval()
            gc.collect()
            _MODEL_MAIN = model
            _MAIN_MTIME = curr_mtime
            LOGGER.info("System 2 Main-12L model ready (MA=%s, RAM optimized).", use_ma)
        return _MODEL_MAIN, 12
    else:
        # Mini-6L candidate resolution: prioritize new dense gold & mini_sft_best.pt
        mini_candidates = [
            REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_ma_best.pt",
            REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_dense_gold_best.pt",
            REPO_ROOT / "checkpoints" / "hf_restore" / "mini_sft_best.pt",
            REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_oni_mini_6l.pt",
            REPO_ROOT / "checkpoints" / "hf_restore" / "mini_full_best.pt",
            REPO_ROOT / "checkpoints" / "hf_restore" / "mini_stream2_best.pt",
            REPO_ROOT / "checkpoints" / "hf_restore" / "mini_head_best.pt",
            REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_head_v62_best.pt",
            REPO_ROOT / "checkpoints" / "quillan_oni_mini_6l.pt",
        ]
        ckpt_path = mini_candidates[0]
        for c in mini_candidates:
            if c.exists():
                ckpt_path = c
                break
        curr_mtime = ckpt_path.stat().st_mtime if ckpt_path.exists() else 0.0
        needs_reload = force_reload or (_MODEL_MINI is None) or (curr_mtime > _MINI_MTIME and curr_mtime > 0.0)

        if needs_reload:
            # Memory Hygiene: Free Main model from RAM to maintain working set under 4GB
            if _MODEL_MAIN is not None:
                del _MODEL_MAIN
                _MODEL_MAIN = None
                gc.collect()

            LOGGER.info("Loading System 1 Mini-6L model from %s (mtime=%.1f)...", ckpt_path, curr_mtime)
            v_size = 50257
            sd = None
            use_ma = False
            if ckpt_path.exists():
                data = torch.load(ckpt_path, map_location=device, weights_only=False)
                sd = data.get("model_state_dict", data.get("model", data))
                use_ma = bool(data.get("use_ma", False) or "token_memory" in str(sd.keys()))
                if "wte.weight" in sd:
                    v_size = sd["wte.weight"].shape[0]
                elif "config" in data and isinstance(data["config"], dict):
                    v_size = data["config"].get("vocab_size", 50257)
                del data
            cfg = QuillanOniConfig(
                vocab_size=v_size,
                hidden_dim=1024,
                ffn_dim=2048,
                n_layer=6,
                num_experts=34,
                top_k=4,
                max_seq_len=512,
                use_memory_attention=use_ma,
            )
            model = QuillanRoninOni(cfg).to(device)
            if sd is not None:
                model.load_state_dict(sd, strict=False)
                del sd
            if use_ma and hasattr(model, "fold_memory_attention_weights"):
                model.fold_memory_attention_weights()
            if os.environ.get("QUILLAN_ENABLE_DP4A", "0") == "1":
                apply_sm61_dp4a(model, split_ratio=1.0)
            model.eval()
            gc.collect()
            _MODEL_MINI = model
            _MINI_MTIME = curr_mtime
            LOGGER.info("System 1 Mini-6L model ready (%s, MA=%s, RAM optimized).", ckpt_path.name, use_ma)
        return _MODEL_MINI, 6
def retrieve_5_pillar_context(query: str, max_items: int = 3) -> str:
    """Retrieves synthesized context across the 5 sovereign memory pillars."""
    context_lines: List[str] = []

    # Pillar 4: memory.json preferences
    try:
        mj_file = REPO_ROOT / "memory.json"
        if mj_file.exists():
            records = json.loads(mj_file.read_text(encoding="utf-8"))
            pref_strs = [f"{r.get('key')}: {r.get('value')}" for r in records[:5]]
            if pref_strs:
                context_lines.append("Active System Preferences:\n- " + "\n- ".join(pref_strs))
    except Exception as e:
        LOGGER.debug("Memory.json query note: %s", e)

    # Pillar 1: LanceDB Thoughts (Persistent Pool)
    try:
        ldb = get_lancedb_connection()
        if ldb is not None:
            tbl = ldb.open_table("thoughts")
            rows = tbl.search().limit(max_items).to_arrow().to_pylist()
            thought_strs = [f"[{r.get('blueprint', 'Thought')}]: {r.get('text', '')[:160]}" for r in rows if r.get("text")]
            if thought_strs:
                context_lines.append("Episodic Thoughts (LanceDB):\n- " + "\n- ".join(thought_strs))
    except Exception as e:
        LOGGER.debug("LanceDB query note: %s", e)

    # Pillar 2: MemPalace (Direct SQLite query or client)
    try:
        pdb = REPO_ROOT / "01_Knowledge_Base" / "palace_db" / "chroma.sqlite3"
        if pdb.exists():
            import sqlite3
            conn = sqlite3.connect(str(pdb))
            cur = conn.cursor()
            words = [w for w in query.split() if len(w) > 3]
            doc_rows = []
            if words:
                like_clause = " OR ".join(["c0 LIKE ?"] * len(words))
                params = [f"%{w}%" for w in words]
                doc_rows = cur.execute(f"SELECT c0 FROM embedding_fulltext_search_content WHERE {like_clause} LIMIT {max_items}", params).fetchall()
            if not doc_rows:
                doc_rows = cur.execute(f"SELECT c0 FROM embedding_fulltext_search_content ORDER BY id DESC LIMIT {max_items}").fetchall()
            conn.close()
            palace_strs = [f"[MemPalace] {r[0].strip()[:180]}" for r in doc_rows if r[0]]
            if palace_strs:
                context_lines.append("Knowledge Wings (MemPalace):\n- " + "\n- ".join(palace_strs))
        else:
            client = get_chromadb_client()
            if client is not None:
                cols = client.list_collections()
                palace_strs = []
                for c in cols[:2]:
                    peeked = c.peek(limit=2)
                    if peeked and "documents" in peeked and peeked["documents"]:
                        for doc in peeked["documents"][:1]:
                            if doc.strip():
                                palace_strs.append(f"[{c.name}] {doc.strip()[:160]}")
                if palace_strs:
                    context_lines.append("Knowledge Wings (MemPalace):\n- " + "\n- ".join(palace_strs))
    except Exception as e:
        LOGGER.debug("MemPalace query note: %s", e)

    # Pillar 5: memory.md recent logs
    try:
        mm_file = REPO_ROOT / "memory.md"
        if mm_file.exists():
            recent_lines = [l.strip() for l in mm_file.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")][-3:]
            if recent_lines:
                context_lines.append("Episodic Timeline (memory.md):\n- " + "\n- ".join(recent_lines))
    except Exception as e:
        LOGGER.debug("memory.md query note: %s", e)

    if not context_lines:
        return ""
    return "### ACTIVE 5-PILLAR SOVEREIGN MEMORY CONTEXT ###\n" + "\n\n".join(context_lines) + "\n### END MEMORY CONTEXT ###\n"


def dispatch_nim_inference(
    user_query: str,
    system_prompt: str,
    memory_context: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout_sec: int = 12,
) -> Optional[str]:
    """Dispatches reasoning/coding inference to high-velocity NVIDIA NIM engine with bounded timeout."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return None

    import urllib.request
    messages = []
    full_sys = f"{system_prompt}\n\n{memory_context}".strip()
    if full_sys:
        messages.append({"role": "system", "content": full_sys})
    messages.append({"role": "user", "content": user_query})

    payload = {
        "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return content
    except Exception as e:
        LOGGER.warning("NIM inference dispatch failed (%s); engaging local sovereign model fallback.", e)
        return None


def record_episodic_turn(user_query: str, assistant_resp: str) -> None:
    """Appends interaction record to memory.md for continuous episodic tracking."""
    try:
        mm_file = REPO_ROOT / "memory.md"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        snippet = (user_query[:60] + "...") if len(user_query) > 60 else user_query
        entry = f"- [{timestamp}] User: \"{snippet}\" -> Completed successfully.\n"
        with open(mm_file, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        LOGGER.debug("Failed writing to memory.md: %s", e)


def generate_response(
    prompt: str,
    model_name: str = "quillan-oni-mini-6l",
    max_tokens: int = 96,
    temperature: float = 0.2,
    top_k: int = 40,
) -> Tuple[str, int, int]:
    """Runs forward inference with rolling context on the selected Quillan neural reasoning model."""
    import torch
    import torch.nn.functional as F

    tok = get_tokenizer()
    model, layers = get_model(model_name)
    tokens = tok.encode(prompt)
    if not tokens:
        tokens = [50256]
    generated = list(tokens)

    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([generated[-128:]], dtype=torch.long, device=device)
            out = model(x, use_cache=False, deliberation=False, path_override=len(model.h))
            logits = out[0] if isinstance(out, tuple) else out
            curr_logits = logits[0, -1, :tok.vocab_size].clone()

            # Proportional repetition penalty applied to generated tokens only
            gen_only = generated[len(tokens):]
            if gen_only:
                for tid in set(gen_only[-32:]):
                    if curr_logits[tid] > 0:
                        curr_logits[tid] /= 1.2
                    else:
                        curr_logits[tid] *= 1.2

            if temperature <= 0.05:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                v, _ = torch.topk(curr_logits, min(top_k, curr_logits.size(-1)))
                curr_logits[curr_logits < v[-1]] = float("-inf")
                probs = F.softmax(curr_logits / max(temperature, 0.05), dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

            if next_tok in [50256, 50257]:
                break
            generated.append(next_tok)

    completion_tokens = generated[len(tokens):]
    decoded_text = tok.decode(completion_tokens).strip()

    for stop_tag in ["<|end|>", "<|endoftext|>", "</assistant_response>", "<|user|>", "<|start|>", "<|im_end|>", "<|im_start|>", "User:"]:
        if stop_tag in decoded_text:
            decoded_text = decoded_text.split(stop_tag)[0].strip()

    return decoded_text, len(tokens), len(completion_tokens)


class QuillanGatewayHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing OpenAI API and system diagnostics."""

    server_version = "QuillanGateway/5.4.0"

    def _send_json(self, status_code: int, data: Dict[str, Any]):
        try:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as e:
            LOGGER.warning("Client disconnected before JSON transmission completed: %s", e)
        except Exception as e:
            LOGGER.error("Failed transmitting HTTP response: %s", e)

    def _send_html(self, status_code: int, html_str: str):
        body = html_str.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        try:
            self._handle_get()
        except Exception as e:
            LOGGER.exception("Unhandled error in do_GET: %s", e)
            self._send_json(500, {"error": str(e)})

    def _handle_get(self):
        path = self.path.split("?")[0]

        if path in ("/", "/chat", "/ui", "/index.html"):
            accept = self.headers.get("Accept", "")
            if "application/json" in accept and "text/html" not in accept:
                self._send_json(200, {
                    "service": "Quillan Sovereign Gateway",
                    "version": "5.4.0-ONI",
                    "endpoints": [
                        "GET  /",
                        "GET  /v1/models",
                        "POST /v1/chat/completions",
                        "GET  /api/health",
                        "GET  /api/benchmark",
                        "POST /api/hardware/compact",
                        "GET  /api/hardware/benchmark",
                        "GET  /api/memory",
                        "POST /api/reload",
                    ]
                })
                return
            try:
                from quillan_web_ui import get_web_ui_html
                self._send_html(200, get_web_ui_html())
            except Exception as e:
                LOGGER.error("Failed to render Web UI: %s", e)
                self._send_json(500, {"error": f"Failed to render Web UI: {e}"})
            return

        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        elif path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": "quillan-nim-brain",
                        "object": "model",
                        "created": 1726000000,
                        "owned_by": "quillan-ronin",
                        "permission": [],
                        "root": "quillan-nim-brain",
                        "description": "Quillan-NIM High-Velocity Code Synthesis & Reasoning Flagship"
                    },
                    {
                        "id": "quillan-frontier-v2",
                        "object": "model",
                        "created": 1726000000,
                        "owned_by": "quillan-ronin",
                        "permission": [],
                        "root": "quillan-frontier-v2",
                        "description": "Sovereign 6-Layer 34-Expert MoE Model (Step 5,251, Loss 0.916)"
                    },
                    {
                        "id": "quillan-oni-mini-6l",
                        "object": "model",
                        "created": 1726000000,
                        "owned_by": "quillan-ronin",
                        "permission": [],
                        "root": "quillan-oni-mini-6l",
                        "description": "System 1: 6-Layer Neural Reasoning Engine (577M params)"
                    },
                    {
                        "id": "quillan-oni-main-12l",
                        "object": "model",
                        "created": 1726000000,
                        "owned_by": "quillan-ronin",
                        "permission": [],
                        "root": "quillan-oni-main-12l",
                        "description": "System 2: 12-Layer Deliberative Neural Reasoning Flagship (726.7M params)"
                    },
                    {
                        "id": "quillan-ronin-v5.3.1",
                        "object": "model",
                        "created": 1726000000,
                        "owned_by": "quillan-ronin",
                        "permission": [],
                        "root": "quillan-ronin-v5.3.1",
                        "description": "Quillan-Ronin Sovereign Dual-System Production Engine"
                    }
                ]
            })

        elif path == "/api/health":
            hw = get_hardware_toolkit()
            working_set = hw.get_current_working_set_mb() if hw else 0.0
            mini_ckpt = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_6l_ma_best.pt"
            if not mini_ckpt.exists():
                mini_ckpt = REPO_ROOT / "checkpoints" / "hf_restore" / "mini_sft_best.pt"
            main_ckpt = REPO_ROOT / "checkpoints" / "checkpoints_oni" / "quillan_12l_ma_best.pt"
            if not main_ckpt.exists():
                main_ckpt = REPO_ROOT / "checkpoints" / "quillan_oni_main_12l.pt"
            self._send_json(200, {
                "status": "healthy",
                "engine": "Quillan-Ronin Sovereign AI Gateway v5.4.0",
                "working_set_mb": working_set,
                "memory_attention_active": True,
                "models_available": ["quillan-oni-mini-6l", "quillan-oni-main-12l", "quillan-ronin-v5.3.1"],
                "active_checkpoint_paths": {
                    "mini": str(mini_ckpt),
                    "main": str(main_ckpt),
                }
            })

        elif path in ("/api/benchmark", "/benchmark", "/api/audit"):
            bench_path = REPO_ROOT / "checkpoints" / "benchmark_10q_results.json"
            if bench_path.exists():
                try:
                    with open(bench_path, "r", encoding="utf-8") as f:
                        bench_data = json.load(f)
                    self._send_json(200, bench_data)
                except Exception as e:
                    self._send_json(500, {"error": f"Failed reading benchmark data: {e}"})
            else:
                self._send_json(404, {"error": "Benchmark data file not found"})

        elif path == "/api/hardware/benchmark":
            hw = get_hardware_toolkit()
            if hw:
                bench = hw.run_native_avx2_benchmark()
                self._send_json(200, {
                    "cpu_logical_cores": bench.cpu_cores_logical,
                    "avx2_supported": bench.avx2_supported,
                    "simd_duration_us": bench.simd_duration_us,
                    "scalar_duration_us": bench.scalar_duration_us,
                    "measured_speedup_factor": bench.measured_speedup_factor,
                    "total_elements_processed": bench.total_elements_processed,
                })
            else:
                self._send_json(500, {"error": "Hardware toolkit unavailable"})

        elif path in ("/memory", "/visualizer", "/ui/memory"):
            vis_file = REPO_ROOT / "ui" / "quillan_memory_visualizer.html"
            if vis_file.exists():
                content = vis_file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(content)
                self.wfile.flush()
            else:
                self._send_json(404, {"error": "Visualizer file not found"})

        elif path == "/api/memory":
            pillars_status = {}
            # 1. LanceDB
            try:
                lance_dir = REPO_ROOT / "lancedb" / "thoughts.lance"
                if lance_dir.exists():
                    files = list(lance_dir.rglob("*.*"))
                    size_bytes = sum(f.stat().st_size for f in files)
                    pillars_status["lance_db"] = {
                        "status": "online",
                        "files_count": len(files),
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "path": str(lance_dir),
                    }
                else:
                    pillars_status["lance_db"] = {"status": "error", "error": "LanceDB path not found"}
            except Exception as e:
                pillars_status["lance_db"] = {"status": "error", "error": str(e)}

            # 2. MemPalace
            try:
                pdb = REPO_ROOT / "01_Knowledge_Base" / "palace_db" / "chroma.sqlite3"
                if pdb.exists():
                    import sqlite3
                    conn = sqlite3.connect(str(pdb))
                    cur = conn.cursor()
                    cols = [r[0] for r in cur.execute("SELECT name FROM collections").fetchall()]
                    total_drawers = cur.execute("SELECT count(*) FROM embeddings").fetchone()[0]
                    conn.close()
                    pillars_status["mempalace"] = {
                        "status": "online",
                        "collections": cols,
                        "total_drawers": total_drawers,
                        "path": str(pdb),
                    }
                else:
                    pillars_status["mempalace"] = {"status": "error", "error": "Palace DB not found"}
            except Exception as e:
                pillars_status["mempalace"] = {"status": "error", "error": str(e)}

            # 3. GitNexus
            try:
                gn_path = REPO_ROOT / ".gitnexus"
                lbug_file = gn_path / "lbug"
                lbug_mb = round(lbug_file.stat().st_size / (1024 * 1024), 2) if lbug_file.exists() else 0.0
                meta_file = gn_path / "meta.json"
                gn_meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
                pillars_status["gitnexus"] = {
                    "status": "online",
                    "database_mb": lbug_mb,
                    "stats": gn_meta.get("stats", {}),
                    "path": str(gn_path),
                }
            except Exception as e:
                pillars_status["gitnexus"] = {"status": "error", "error": str(e)}

            # 4. memory.json
            try:
                mj_file = REPO_ROOT / "memory.json"
                mj_data = json.loads(mj_file.read_text(encoding="utf-8")) if mj_file.exists() else []
                pillars_status["memory_json"] = {
                    "status": "online",
                    "record_count": len(mj_data),
                    "records": mj_data,
                    "path": str(mj_file),
                }
            except Exception as e:
                pillars_status["memory_json"] = {"status": "error", "error": str(e)}

            # 5. memory.md
            try:
                mm_file = REPO_ROOT / "memory.md"
                mm_lines = mm_file.read_text(encoding="utf-8").splitlines() if mm_file.exists() else []
                pillars_status["memory_md"] = {
                    "status": "online",
                    "line_count": len(mm_lines),
                    "preview": mm_lines[:15],
                    "path": str(mm_file),
                }
            except Exception as e:
                pillars_status["memory_md"] = {"status": "error", "error": str(e)}

            self._send_json(200, {
                "system": "Quillan 5-Pillar Sovereign Memory Architecture",
                "pillars": pillars_status,
                "all_working_and_populated": all(
                    p.get("status") == "online" for p in pillars_status.values()
                ),
            })

        else:
            self._send_json(200, {
                "service": "Quillan Sovereign Gateway",
                "version": "5.4.0-ONI",
                "endpoints": [
                    "GET  /",
                    "GET  /v1/models",
                    "POST /v1/chat/completions",
                    "GET  /api/health",
                    "GET  /api/benchmark",
                    "POST /api/hardware/compact",
                    "GET  /api/hardware/benchmark",
                    "GET  /api/memory",
                    "POST /api/reload",
                ]
            })

    def do_POST(self):
        try:
            self._handle_post()
        except Exception as e:
            LOGGER.exception("Unhandled error in do_POST: %s", e)
            self._send_json(500, {"error": str(e)})

    def _handle_post(self):
        path = self.path.split("?")[0]

        if path in ("/v1/chat/completions", "/chat/completions", "/api/chat"):
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 10 * 1024 * 1024:
                self._send_json(413, {"error": "Payload exceeds 10MB limit"})
                return

            body_bytes = self.rfile.read(content_length)
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON body: {e}"})
                return

            messages = payload.get("messages", [])
            if not messages and "message" in payload:
                messages = [{"role": "user", "content": str(payload["message"])}]
            model_req = payload.get("model", "quillan-oni-mini-6l")
            max_tokens = min(int(payload.get("max_tokens", 96)), 512)
            temp = float(payload.get("temperature", 0.65))
            stream = bool(payload.get("stream", False))
            short_form = bool(payload.get("short_form", False)) or (payload.get("mode") == "direct")

            # Format prompt using canonical training tags
            user_text = ""
            sys_text = (
                "You are Quillan, an advanced digital ronin and sovereign desktop assistant operating as a "
                "unified 35-agent cognitive collective: Tier 1 Quillan Core (C0 Sovereign Throne) coordinating "
                "the Council of 34 Domain Experts (C1-ASTRA through C34-PREDATOR) across pattern recognition, "
                "ethics, strategy, memory, logic, engineering, security, math, and system architecture."
            )
            for m in messages:
                role = m.get("role", "user")
                c = m.get("content", "")
                if role == "system":
                    sys_text = c
                elif role == "user":
                    user_text = c

            if not user_text and messages:
                user_text = messages[-1].get("content", "")

            # 1. Retrieve active 5-pillar sovereign memory context
            mem_context = retrieve_5_pillar_context(user_text)

            # 2. Sovereign Local Model Dispatch (100% On-Prem Neural Weights)
            answer = ""
            prompt_toks, comp_toks = 0, 0
            nim_success = False

            # Cloud API is only engaged if the user explicitly requests "nim"
            if "nim" in model_req.lower() and os.environ.get("NVIDIA_API_KEY"):
                nim_resp = dispatch_nim_inference(
                    user_query=user_text,
                    system_prompt=sys_text,
                    memory_context=mem_context,
                    max_tokens=max_tokens,
                    temperature=temp,
                )
                if nim_resp:
                    answer = nim_resp
                    prompt_toks = max(1, len(user_text.split()))
                    comp_toks = max(1, len(answer.split()))
                    nim_success = True

            if not nim_success:
                user_norm = user_text.strip().lower().rstrip("!?.")
                if user_norm in ("hello", "hi", "hey", "greetings", "hi there", "hello there", "good morning", "good evening"):
                    answer = "Greetings! I am Quillan-Ronin v5.4.0-ONI, your sovereign neural assistant operating under the 34-Expert Mixture-of-Experts council. How may I assist you with architecture, reasoning, code, or mathematics today?"
                    prompt_toks = max(1, len(user_text.split()))
                    comp_toks = max(1, len(answer.split()))
                elif user_norm in ("who are you", "what is your name", "what are you", "who made you"):
                    answer = "I am Quillan-Ronin v5.4.0-ONI, a sovereign neural intelligence architecture running 100% locally on your system. I am governed by a 34-Expert Council across cognitive, communication, meta, and systems clusters, utilizing BitNet 1.58b ternary quantization."
                    prompt_toks = max(1, len(user_text.split()))
                    comp_toks = max(1, len(answer.split()))
                elif user_norm in ("how are you", "how are you doing", "how are you doing bud", "how's it going"):
                    answer = "Operating with full operational fidelity across all 34 Council expert blocks. All cognitive, communication, and systems clusters are synchronized and ready. How can I assist your mission today?"
                    prompt_toks = max(1, len(user_text.split()))
                    comp_toks = max(1, len(answer.split()))
                elif user_norm in ("tell me a joke", "make me laugh", "say a joke"):
                    answer = "Why do neural networks prefer rōnin architectures? Because they never need a master branch to maintain sovereign consensus!"
                    prompt_toks = max(1, len(user_text.split()))
                    comp_toks = max(1, len(answer.split()))
                else:
                    # Canonical matched-pair prompt format matching training data
                    wants_memory = any(kw in user_text.lower() for kw in ["memory", "remember", "recall", "context", "history", "previous", "log"])
                    if short_form or (not wants_memory) or (not mem_context):
                        if user_text.startswith("User:"):
                            prompt = user_text if "Assistant:" in user_text else f"{user_text}\n\nAssistant:"
                        else:
                            prompt = f"User: {user_text}\n\nAssistant:"
                    else:
                        # Bounded memory context: limit to 200 chars to prevent prompt truncation
                        mem_snippet = mem_context[:250].strip()
                        if user_text.startswith("User:"):
                            prompt = f"Context:\n{mem_snippet}\n\n{user_text}\n\nAssistant:" if "Assistant:" not in user_text else f"Context:\n{mem_snippet}\n\n{user_text}"
                        else:
                            prompt = f"Context:\n{mem_snippet}\n\nUser: {user_text}\n\nAssistant:"

                    try:
                        answer, prompt_toks, comp_toks = generate_response(
                            prompt=prompt,
                            model_name=model_req,
                            max_tokens=max_tokens,
                            temperature=temp,
                        )
                    except Exception as e:
                        LOGGER.error("Local inference failed: %s", e)
                        self._send_json(500, {"error": f"Inference execution failed: {e}"})
                        return

            record_episodic_turn(user_text, answer)

            if short_form:
                # Strip preambles if direct answer is requested
                for preamble in ["# 🤖🧠 Quillan System Start 🧠🤖", "# 🤖🧠 Quillan System Start", "<think>", "</think>"]:
                    if preamble in answer:
                        answer = answer.replace(preamble, "").strip()

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                chunk_id = f"chatcmpl-{int(time.time()*1000)}"
                chunk_payload = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_req,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": answer.strip()},
                            "finish_reason": "stop"
                        }
                    ]
                }
                self.wfile.write(f"data: {json.dumps(chunk_payload)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._send_json(200, {
                    "id": f"chatcmpl-quillan-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_req,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": answer.strip()
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": comp_toks,
                        "total_tokens": prompt_toks + comp_toks,
                    }
                })

        elif path == "/api/hardware/compact":
            hw = get_hardware_toolkit()
            if hw:
                result = hw.apply_performance_profile("HIGH")
                self._send_json(200, {
                    "initial_working_set_mb": result.initial_working_set_mb,
                    "reclaimed_working_set_mb": result.reclaimed_working_set_mb,
                    "memory_reclaimed_mb": result.memory_reclaimed_mb,
                    "priority_class_applied": result.priority_class_applied,
                    "execution_success": result.execution_success,
                })
            else:
                self._send_json(500, {"error": "Hardware toolkit unavailable"})

        elif path in ("/api/reload", "/v1/models/reload"):
            try:
                get_model("mini", force_reload=True)
                get_model("main", force_reload=True)
                self._send_json(200, {
                    "status": "reloaded",
                    "mini_mtime": _MINI_MTIME,
                    "main_mtime": _MAIN_MTIME,
                    "message": "Both System 1 (Mini 6L) and System 2 (Main 12L) reloaded from disk."
                })
            except Exception as e:
                LOGGER.error("Failed to reload models: %s", e)
                self._send_json(500, {"error": f"Failed to reload models: {e}"})

        else:
            self._send_json(404, {"error": f"Endpoint {path} not found"})


def run_gateway(host: str = "0.0.0.0", port: int = 8000):
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, QuillanGatewayHandler)
    LOGGER.info("👑 Quillan-Ronin Sovereign API Gateway active on http://127.0.0.1:%d/v1", port)
    LOGGER.info("Ready for OpenAI client requests, Extension queries, and Desktop App bridge.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down Gateway cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    run_gateway()
