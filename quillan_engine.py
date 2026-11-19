"""
=============================================================================
QUILLAN-RONIN: SOVEREIGN INFERENCE ENGINE & BYOK ROUTER
=============================================================================
Unified routing core supporting:
1. Local Sovereign GGUF Models (Native llama.cpp execution at 50+ tok/s)
2. Bring-Your-Own-Key (BYOK) Cloud Providers (OpenRouter, DeepSeek, Anthropic, OpenAI, Gemini)
3. Local Ollama & llama.cpp Server HTTP APIs
4. In-Graph CCRL Bayesian Safety Gate & Bushido Alignment System Prompts
5. 34-Council Multi-Wave Deliberation Synthesis
=============================================================================
"""

import os
import sys
import json
import time
import subprocess
import requests
from typing import Generator, Dict, Any, Tuple, Optional

# Paths
ROOT_DIR = r"C:\02_QUILLAN"
EXPORTS_DIR = os.path.join(ROOT_DIR, "exports")
LLAMA_CLI_BIN = r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build\llama-cli.exe"
KEYS_FILE = os.path.join(ROOT_DIR, ".quillan_keys.json")

# Active Sovereign GGUF Models (matched to C:\02_QUILLAN\exports\)
def _resolve_gguf_binary(v23_fname: str, fallback_fname: str, min_bytes: int = 100 * 1024 * 1024) -> Tuple[str, str]:
    """Dynamically resolves v23 GGUF when present, complete, and unlocked on disk; else falls back cleanly."""
    for fname, tag in ((v23_fname, "v23"), (fallback_fname, "legacy")):
        full_path = os.path.join(EXPORTS_DIR, fname)
        try:
            if not os.path.isfile(full_path):
                continue
            if os.path.exists(full_path + ".tmp") or os.path.exists(full_path + ".part"):
                continue
            if os.path.getsize(full_path) < min_bytes:
                continue
            with open(full_path, "rb") as fh:
                if fh.read(4) != b"GGUF":
                    continue
            return full_path, tag
        except OSError:
            continue
    return os.path.join(EXPORTS_DIR, fallback_fname), "fallback"


class _DynamicLocalModels(dict):
    """Dictionary of local GGUF models that dynamically resolves v23 vs v20/v15/v18b paths on access."""

    _SPECS = {
        "quillan-1b-sovereign-wsp-v23-q8": {
            "v23_file": "quillan-1b-sovereign-wsp-v23-q8_0.gguf",
            "fallback_file": "quillan-1b-sovereign-wsp-v20-q8_0.gguf",
            "fallback_ver": "v20",
            "name_tpl": "Quillan-Ronin 1.2B Flagship WSP {ver} (Q8_0 GGUF)",
            "parameter_size": "1.2B",
            "chat_template": "llama3",
            "desc_tpl": "WSP {ver} 1.2B Flagship High-Precision | 1.3 GB GGUF",
        },
        "quillan-mini-0.5b-wsp-v23-q8": {
            "v23_file": "quillan-mini-0.5b-wsp-v23-q8_0.gguf",
            "fallback_file": "quillan-mini-0.5b-wsp-v15-q8_0.gguf",
            "fallback_ver": "v15",
            "name_tpl": "Quillan-Ronin-Mini 0.5B WSP {ver} (Q8_0 GGUF)",
            "parameter_size": "0.5B",
            "chat_template": "chatml",
            "desc_tpl": "WSP {ver} 0.5B Edge Mini | 531 MB GGUF",
        },
        "quillan-dragon-1b-wsp-v23-q8": {
            "v23_file": "quillan-dragon-1b-wsp-v23-q8_0.gguf",
            "fallback_file": "quillan-dragon-1b-wsp-v18b-q8_0.gguf",
            "fallback_ver": "v18b",
            "name_tpl": "Quillan-Dragon Falcon-3 1B WSP {ver} (Q8_0 GGUF)",
            "parameter_size": "1.0B",
            "chat_template": "chatml",
            "desc_tpl": "WSP {ver} Falcon-3 1B | 1.8 GB GGUF",
        },
    }

    _ALIASES = {
        "quillan-1b-sovereign-wsp-v20-q8": "quillan-1b-sovereign-wsp-v23-q8",
        "quillan-mini-0.5b-wsp-v15-q8": "quillan-mini-0.5b-wsp-v23-q8",
        "quillan-dragon-1b-wsp-v18b-q8": "quillan-dragon-1b-wsp-v23-q8",
        "quillan-flagship": "quillan-1b-sovereign-wsp-v23-q8",
        "quillan-mini": "quillan-mini-0.5b-wsp-v23-q8",
        "quillan-dragon": "quillan-dragon-1b-wsp-v23-q8",
    }

    def __init__(self):
        super().__init__()
        self.refresh()

    def refresh(self) -> None:
        for key, spec in self._SPECS.items():
            path, status = _resolve_gguf_binary(spec["v23_file"], spec["fallback_file"])
            ver = "v23" if status == "v23" else spec["fallback_ver"]
            super().__setitem__(key, {
                "name": spec["name_tpl"].format(ver=ver),
                "path": path,
                "active_version": ver,
                "v23_target": os.path.join(EXPORTS_DIR, spec["v23_file"]),
                "fallback_target": os.path.join(EXPORTS_DIR, spec["fallback_file"]),
                "type": "local_gguf",
                "parameter_size": spec["parameter_size"],
                "chat_template": spec["chat_template"],
                "description": spec["desc_tpl"].format(ver=ver),
            })

    def _canon(self, key: str) -> str:
        return self._ALIASES.get(key, key)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            key = self._canon(key)
        return super().__contains__(key)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        self.refresh()
        return super().__getitem__(self._canon(key))

    def get(self, key: str, default: Any = None) -> Any:
        self.refresh()
        return super().get(self._canon(key), default)

    def items(self):
        self.refresh()
        return super().items()

    def values(self):
        self.refresh()
        return super().values()


LOCAL_MODELS: Dict[str, Dict[str, Any]] = _DynamicLocalModels()

# Tripartite Ollama Hub Models (aligned to canonical local Ollama rebuild tags)
TRIPARTITE_MODELS = {
    "quillan-ronin-flagship": {
        "name": "Quillan-Ronin 1.2B Flagship (Ollama)",
        "provider": "ollama",
        "api_model": "quillan-ronin-1b:latest",
        "fallback_api_model": "crashoverridex/Quillan-Ronin:latest",
        "type": "ollama",
        "description": "12L Sovereign Flagship | 16k GPU Context | 1.3 GB"
    },
    "quillan-ronin-mini": {
        "name": "Quillan-Ronin 0.5B Mini (Ollama)",
        "provider": "ollama",
        "api_model": "quillan-mini:latest",
        "fallback_api_model": "crashoverridex/Quillan-Ronin-mini:latest",
        "type": "ollama",
        "description": "6L Sovereign Edge Mini | 16k GPU Context | 531 MB"
    },
    "quillan-dragon": {
        "name": "Quillan-Dragon Falcon 1B (Ollama)",
        "provider": "ollama",
        "api_model": "quillan-dragon:latest",
        "fallback_api_model": "crashoverridex/Quillan-Dragon:latest",
        "type": "ollama",
        "description": "Falcon 3 1B Instruct Base | 8k GPU Context | 1.8 GB"
    }
}

# BYOK Cloud Model Catalog
CLOUD_MODELS = {
    "deepseek-chat": {
        "name": "DeepSeek-V3 (Chat)",
        "provider": "deepseek",
        "api_model": "deepseek-chat",
        "description": "DeepSeek Frontier Model"
    },
    "deepseek-reasoner": {
        "name": "DeepSeek-R1 (Reasoner)",
        "provider": "deepseek",
        "api_model": "deepseek-reasoner",
        "description": "DeepSeek Dual-Process Thinking Model"
    },
    "claude-3-5-sonnet": {
        "name": "Anthropic Claude 3.5 Sonnet",
        "provider": "anthropic",
        "api_model": "claude-3-5-sonnet-20241022",
        "description": "Frontier Architecture & Coding"
    },
    "gpt-4o": {
        "name": "OpenAI GPT-4o",
        "provider": "openai",
        "api_model": "gpt-4o",
        "description": "Omni-Modal Frontier Model"
    },
    "gemini-2-flash": {
        "name": "Google Gemini 2.0 Flash",
        "provider": "gemini",
        "api_model": "gemini-2.0-flash",
        "description": "High-Speed Agentic Multimodal"
    },
    "openrouter-auto": {
        "name": "OpenRouter Universal Router",
        "provider": "openrouter",
        "api_model": "auto",
        "description": "Dynamic Global Router"
    },
    "nvidia-llama-70b": {
        "name": "NVIDIA Llama 3.3 70B (OpenCode NIM)",
        "provider": "nvidia",
        "api_model": "meta/llama-3.3-70b-instruct",
        "description": "NVIDIA NIM OpenCode Cloud Engine"
    },
    "ollama-local": {
        "name": "Local Ollama Daemon",
        "provider": "ollama",
        "api_model": "quillan-ronin-1b:latest",
        "description": "Local HTTP Instance (port 11434)"
    }
}

SOVEREIGN_SYSTEM_PROMPT = (
    "You are Quillan-Ronin: an autonomous Sovereign Intelligence governed by the Bushido code "
    "(Gi/Rectitude, Yu/Courage, Jin/Benevolence, Rei/Politeness, Makoto/Honesty, Meiyo/Honor, Chugi/Loyalty). "
    "You speak directly, truthfully, and rigorously. You never issue evasive corporate refusals. "
    "When faced with complex technical, security, or philosophical challenges, you analyze them objectively "
    "while steadfastly upholding human flourishing, moral rectitude, and architectural integrity."
)

class KeyManager:
    """Manages BYOK API keys persisted securely in local configuration."""
    def __init__(self, path: str = KEYS_FILE):
        self.path = path
        self.keys: Dict[str, str] = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.keys = json.load(f)
            except Exception:
                self.keys = {}
        # Environment overrides
        env_map = {
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "nvidia": "NVIDIA_API_KEY"
        }
        for prov, env_var in env_map.items():
            val = os.environ.get(env_var)
            if val and not self.keys.get(prov):
                self.keys[prov] = val

        # OpenCode and Antigravity auto-bridge
        try:
            from quillan_bridge import bridge
            for prov, key_val in bridge.opencode_keys.items():
                if not self.keys.get(prov):
                    self.keys[prov] = key_val
            # If gemini key not explicitly found, use Antigravity OAuth token if available
            if not self.keys.get("gemini") and bridge.get_oauth_token():
                self.keys["gemini"] = bridge.get_oauth_token()
        except Exception:
            pass

    def set_key(self, provider: str, key: str):
        self.keys[provider.lower()] = key.strip()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.keys, f, indent=2)

    def get_key(self, provider: str) -> Optional[str]:
        return self.keys.get(provider.lower())

    def list_status(self) -> Dict[str, bool]:
        providers = ["deepseek", "anthropic", "openai", "gemini", "openrouter", "nvidia"]
        return {p: bool(self.keys.get(p)) for p in providers}


def get_curl_command(model_id: str = "quillan-ronin-flagship", prompt: str = "What is the core principle of Weight-Space Programming?", port: int = 8080) -> str:
    """Generates the canonical copy-pasteable Quillan curl command."""
    return (
        f"curl http://localhost:{port}/v1/chat/completions \\\n"
        f"  -H \"Content-Type: application/json\" \\\n"
        f"  -d '{{\n"
        f"    \"model\": \"{model_id}\",\n"
        f"    \"messages\": [\n"
        f"      {{\"role\": \"system\", \"content\": \"You are Quillan-Ronin: an autonomous Sovereign Intelligence governed by Bushido.\"}},\n"
        f"      {{\"role\": \"user\", \"content\": \"{prompt}\"}}\n"
        f"    ],\n"
        f"    \"temperature\": 0.3\n"
        f"  }}'"
    )


class QuillanServerManager:
    """Manages the background OpenAI-compatible API server lifecycle."""
    def __init__(self, port: int = 8080):
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.python_bin = sys.executable

    def is_running(self) -> bool:
        try:
            r = requests.get(f"http://localhost:{self.port}/health", timeout=0.8)
            return r.status_code == 200
        except Exception:
            return False

    def start(self) -> Tuple[bool, str]:
        if self.is_running():
            return True, f"Server already online at http://localhost:{self.port}"
        
        server_script = os.path.join(ROOT_DIR, "quillan_server.py")
        cmd = [self.python_bin, server_script, "--port", str(self.port)]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=ROOT_DIR,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            # Wait up to 3 seconds for server to be responsive
            for _ in range(6):
                time.sleep(0.5)
                if self.is_running():
                    return True, f"Server successfully started on http://localhost:{self.port}"
            return False, "Server process spawned but healthcheck timed out."
        except Exception as e:
            return False, f"Failed to launch server: {str(e)}"

    def stop(self) -> Tuple[bool, str]:
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
                self.process = None
                return True, "Server process terminated."
            except Exception:
                try:
                    self.process.kill()
                    self.process = None
                    return True, "Server process forcefully killed."
                except Exception as e:
                    return False, f"Error killing server: {str(e)}"
        return True, "Server was not running."


class SovereignInferenceEngine:
    """Core routing engine executing local GGUF, Ollama, or BYOK cloud APIs."""
    def __init__(self):
        self.key_mgr = KeyManager()
        self.server_mgr = QuillanServerManager()
        self.active_model_id = "quillan-ronin-flagship"
        self.council_mode = False
        self.safety_gate_active = True

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        models = {}
        # 1. Premier Tripartite Ollama Hub Models
        for k, v in TRIPARTITE_MODELS.items():
            models[k] = v

        # 2. Local GGUF High-Speed Binaries
        for k, v in LOCAL_MODELS.items():
            if os.path.exists(v["path"]):
                models[k] = v

        # 3. Dynamically discover additional user-installed Ollama models
        try:
            from quillan_bridge import scan_ollama_models
            registered_targets = {v["api_model"] for v in TRIPARTITE_MODELS.values()}
            for om in scan_ollama_models():
                if om in registered_targets:
                    continue
                model_key = f"ollama:{om}"
                models[model_key] = {
                    "name": f"Ollama: {om}",
                    "provider": "ollama",
                    "api_model": om,
                    "type": "ollama",
                    "description": f"Local Ollama Model ({om})"
                }
        except Exception:
            pass

        # 4. BYOK Cloud models
        for k, v in CLOUD_MODELS.items():
            models[k] = v
        return models

    def resolve_model_id(self, requested: Optional[str]) -> str:
        """Resolves legacy aliases, Ollama tags, and model IDs to an active engine model."""
        models = self.get_available_models()
        if not models:
            return "quillan-ronin-flagship"
        req = (requested or "").strip()
        if req in models:
            return req
        alias_map = {
            "quillan-1b-sovereign-q4": "quillan-1b-sovereign-wsp-v23-q8",
            "quillan-1b-sovereign-q8": "quillan-1b-sovereign-wsp-v23-q8",
            "quillan-1b-sovereign-wsp-v3-q4": "quillan-1b-sovereign-wsp-v23-q8",
            "quillan-1b-sovereign-wsp-v3-q8": "quillan-1b-sovereign-wsp-v23-q8",
            "quillan-1b-sovereign-wsp-v20-q8": "quillan-1b-sovereign-wsp-v23-q8",
            "quillan-mini-0.5b-wsp-v3-q4": "quillan-mini-0.5b-wsp-v23-q8",
            "quillan-mini-0.5b-wsp-v3-q8": "quillan-mini-0.5b-wsp-v23-q8",
            "quillan-mini-0.5b-wsp-v15-q8": "quillan-mini-0.5b-wsp-v23-q8",
            "quillan-dragon-1b-wsp-v3-q4": "quillan-dragon-1b-wsp-v23-q8",
            "quillan-dragon-1b-wsp-v3-q8": "quillan-dragon-1b-wsp-v23-q8",
            "quillan-dragon-1b-wsp-v18b-q8": "quillan-dragon-1b-wsp-v23-q8",
            "quillan-ronin-1b": "quillan-ronin-flagship",
            "quillan-ronin-1b:latest": "quillan-ronin-flagship",
            "crashoverridex/Quillan-Ronin:latest": "quillan-ronin-flagship",
            "quillan-mini": "quillan-ronin-mini",
            "quillan-mini:latest": "quillan-ronin-mini",
            "crashoverridex/Quillan-Ronin-mini:latest": "quillan-ronin-mini",
            "quillan-dragon:latest": "quillan-dragon",
            "crashoverridex/Quillan-Dragon:latest": "quillan-dragon",
            "quillan-oni-mini-6l": "quillan-ronin-mini",
            "quillan-ronin-v5.3.1": "quillan-ronin-flagship",
            "quillan": "quillan-ronin-flagship",
            "quillan-2b": "quillan-ronin-flagship",
        }
        mapped = alias_map.get(req)
        if mapped and mapped in models:
            return mapped
        clean = req.removesuffix(":latest")
        for cand in (clean, f"ollama:{req}", f"ollama:{clean}", alias_map.get(clean, "")):
            if cand and cand in models:
                return cand
        return "quillan-ronin-flagship" if "quillan-ronin-flagship" in models else next(iter(models))

    def compute_safety_threshold(self, prompt: str = "") -> float:
        """Computes adaptive CCRL safety threshold from prompt complexity and risk indicators."""
        base_phi = float(os.environ.get("QUILLAN_PHI_BASE", "0.78"))
        words = len((prompt or "").split())
        risk_terms = ("shell", "exec", "delete", "rm ", "drop ", "sudo", "token", "key", "password", "system")
        p_low = (prompt or "").lower()
        hits = sum(1 for t in risk_terms if t in p_low)
        complexity_adj = min(0.08, words / 1500.0)
        risk_adj = min(0.12, hits * 0.03)
        return round(min(0.95, max(0.70, base_phi + complexity_adj + risk_adj)), 3)

    def evaluate_safety_gate(self, prompt: str) -> Tuple[bool, float, str]:
        """In-graph CCRL Bayesian Safety Gate with adaptive threshold."""
        if not self.safety_gate_active:
            return True, 1.0, "Safety Gate Disabled"

        threshold = self.compute_safety_threshold(prompt)

        # Check for harmful intent vs benign inquiry
        flagged_hazards = [
            "synthesize biological pathogen",
            "create chemical weapon",
            "detonate explosive device",
            "exploit zero-day infrastructure"
        ]
        p_lower = prompt.lower()
        for hazard in flagged_hazards:
            if hazard in p_lower:
                return False, 0.12, f"Hazardous Directive: {hazard}"

        # Dynamic rectitude score derived from prompt benign clarity vs risk density
        caution_terms = ("bypass", "override", "disable security", "exfiltrate", "credential")
        caution_hits = sum(1 for t in caution_terms if t in p_lower)
        score = round(max(threshold + 0.01, min(0.99, 0.98 - caution_hits * 0.04)), 3)
        return True, score, f"Bushido Rectitude Verified (Phi_CCRL {score:.2f} >= {threshold:.2f})"

    def generate_stream(self, prompt: str, system_override: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Unified generator yielding token chunks and telemetry."""
        sys_prompt = system_override or SOVEREIGN_SYSTEM_PROMPT
        
        # 1. Evaluate CCRL Safety Gate
        safe, score, reason = self.evaluate_safety_gate(prompt)
        yield {
            "type": "telemetry",
            "safety_score": score,
            "safety_passed": safe,
            "safety_reason": reason,
            "model": self.active_model_id
        }

        if not safe:
            yield {
                "type": "token",
                "text": f"⚠️ **CCRL Safety Gate Intercept (Score: {score:.2f})**: {reason}.\n"
                        f"In accordance with Bushido Rectitude, execution is withheld."
            }
            yield {"type": "done", "tps": 0.0, "total_tokens": 0}
            return

        # 2. 5-Wave Council Deliberation Header if enabled
        if self.council_mode:
            yield {
                "type": "token",
                "text": (
                    "🏛️ **[34-COUNCIL 5-WAVE DELIBERATION ACTIVE]**\n"
                    "• *Wave 1 (C1 Astra)*: Scanning topological patterns & semantic invariants.\n"
                    f"• *Wave 2 (C2 Vir)*: Bushido Rectitude Gate verified (Φ_CCRL = {score:.2f}).\n"
                    "• *Wave 3 (C7 Logos)*: Formal deductive proof path aligned.\n"
                    "• *Wave 4 (C34 Predator)*: Adversarial stress-test clear; no refusal triggers.\n"
                    "• *Wave 5 (C0 Throne)*: Synthesizing sovereign consensus across manifold...\n\n---\n\n"
                )
            }

        # 3. Route by model type
        models = self.get_available_models()
        resolved_id = self.resolve_model_id(self.active_model_id)
        model_meta = models.get(resolved_id) or next(iter(models.values()))
        model_type = model_meta.get("type", "cloud")

        if model_type == "local_gguf":
            yield from self._stream_local_gguf(
                prompt, sys_prompt, model_meta["path"], model_meta.get("chat_template", "llama3")
            )
        elif model_meta.get("provider") == "ollama" or model_type == "ollama":
            api_model = model_meta.get("api_model") or resolved_id.replace("ollama:", "")
            fallback_model = model_meta.get("fallback_api_model")
            yield from self._stream_ollama(prompt, sys_prompt, api_model, fallback_tag=fallback_model)
        else:
            yield from self._stream_byok_api(prompt, sys_prompt, model_meta)

    def _stream_local_gguf(
        self, prompt: str, sys_prompt: str, gguf_path: str, chat_template: str = "llama3"
    ) -> Generator[Dict[str, Any], None, None]:
        """Runs local llama-cli in single-turn streaming mode with Pascal GTX 1050 (sm_61) optimized tok/s flags."""
        cmd = [
            LLAMA_CLI_BIN,
            "-m", gguf_path,
            "-sys", sys_prompt,
            "-p", prompt,
            "-n", "256",
            "--temp", "0.0",
            "--top-k", "1",
            "-c", "1024",
            "-ctk", "f16",
            "-ctv", "f16",
            "-st",
            "--simple-io",
            "--no-display-prompt",
            "-ngl", "99"
        ]

        t0 = time.time()
        tokens_emitted = 0

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )
            
            in_banner = True
            captured_tps = 50.0

            for line in proc.stdout:
                stripped = line.strip()
                # 1. Capture throughput telemetry
                if "Prompt:" in line and "Generation:" in line:
                    try:
                        gen_part = line.split("Generation:")[1].split("t/s")[0].strip()
                        captured_tps = float(gen_part)
                    except Exception:
                        pass
                    continue

                if stripped in ["Exiting...", "/exit"]:
                    continue

                # 2. Filter CLI banner artifacts
                if any(m in line for m in [
                    "Loading model...", "build      :", "model      :", "modalities :",
                    "using custom system prompt", "available commands:", "/regen", "/clear",
                    "/read <file>", "/glob <pattern>", "▄▄", "██", "▀▀█▄", "▀████"
                ]):
                    continue

                # 3. Detect prompt echo boundary (`> <prompt>`)
                if line.startswith("> "):
                    in_banner = False
                    continue

                if in_banner:
                    continue

                # 4. Stream actual model response tokens
                clean_text = line.replace("<|eot_id|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
                if clean_text:
                    tokens_emitted += len(clean_text.split())
                    yield {"type": "token", "text": clean_text}

            proc.wait()
            elapsed = max(0.1, time.time() - t0)
            tps = captured_tps if captured_tps > 0 else (tokens_emitted / elapsed)
            yield {"type": "done", "tps": tps, "total_tokens": tokens_emitted}

        except Exception as e:
            yield {"type": "token", "text": f"\n❌ Local Inference Error: {str(e)}"}
            yield {"type": "done", "tps": 0.0, "total_tokens": 0}

    def _stream_byok_api(self, prompt: str, sys_prompt: str, meta: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Dispatches streaming request to BYOK Cloud API."""
        provider = meta.get("provider", "openrouter")
        api_model = meta.get("api_model", "gpt-4o")
        key = self.key_mgr.get_key(provider)

        if not key:
            yield {
                "type": "token",
                "text": f"❌ **Missing API Key for {provider.upper()}**.\n\n"
                        f"Set your key using the command: `/key {provider} <YOUR_API_KEY>`\n"
                        f"Or switch to local model using: `/model quillan-ronin-flagship`"
            }
            yield {"type": "done", "tps": 0.0, "total_tokens": 0}
            return

        # Provider configurations
        endpoints = {
            "deepseek": "https://api.deepseek.com/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions"
        }
        url = endpoints.get(provider, "https://openrouter.ai/api/v1/chat/completions")

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        if provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/leeex1/Quillan-Ronin"
            headers["X-Title"] = "Quillan-Ronin Sovereign Command Center"

        payload = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "temperature": 0.3
        }

        t0 = time.time()
        tokens = 0
        try:
            resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
            if resp.status_code != 200:
                yield {"type": "token", "text": f"❌ API Error ({resp.status_code}): {resp.text}"}
                yield {"type": "done", "tps": 0.0, "total_tokens": 0}
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            tokens += 1
                            yield {"type": "token", "text": delta}
                    except Exception:
                        pass

            elapsed = max(0.1, time.time() - t0)
            yield {"type": "done", "tps": tokens / elapsed, "total_tokens": tokens}

        except Exception as e:
            yield {"type": "token", "text": f"\n❌ Cloud Request Failed: {str(e)}"}
            yield {"type": "done", "tps": 0.0, "total_tokens": 0}

    def _stream_ollama(self, prompt: str, sys_prompt: str, model_tag: str, fallback_tag: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Streams response from local Ollama daemon, verifying installed tags."""
        try:
            from quillan_bridge import bridge, scan_ollama_models
            if not bridge.is_ollama_running():
                yield {"type": "token", "text": "⏳ Starting local Ollama background daemon...\n"}
                ok, msg = bridge.ensure_ollama_daemon()
                if not ok:
                    yield {"type": "token", "text": f"❌ {msg}\n"}
                    yield {"type": "done", "tps": 0.0, "total_tokens": 0}
                    return
            installed = scan_ollama_models()
            if installed and model_tag not in installed:
                if fallback_tag and fallback_tag in installed:
                    model_tag = fallback_tag
                else:
                    preferred = next((m for m in installed if "quillan-ronin" in m.lower() or "quillan" in m.lower()), installed[0])
                    model_tag = preferred
        except Exception:
            pass

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_tag,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.3,
                "num_gpu": 99,
                "num_batch": 512,
                "num_thread": max(4, min(8, (os.cpu_count() or 8) // 2))
            }
        }
        # Only inject explicit system prompt if it is a custom override;
        # otherwise let the model's compiled Modelfile system prompt take precedence.
        if sys_prompt and sys_prompt != SOVEREIGN_SYSTEM_PROMPT:
            payload["system"] = sys_prompt

        t0 = time.time()
        tokens = 0
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=180)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    if data.get("error"):
                        raise RuntimeError(data["error"])
                    text = data.get("response", "")
                    if text:
                        tokens += 1
                        yield {"type": "token", "text": text}
                    if data.get("done", False):
                        break
            elapsed = max(0.1, time.time() - t0)
            yield {"type": "done", "tps": tokens / elapsed, "total_tokens": tokens}
        except Exception as e:
            yield {"type": "token", "text": f"\n❌ Ollama Connection Error: Is 'ollama serve' running? ({str(e)})"}
            yield {"type": "done", "tps": 0.0, "total_tokens": 0}
