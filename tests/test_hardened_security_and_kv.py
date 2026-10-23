"""
test_hardened_security_and_kv.py
Comprehensive regression tests for:
1. Computeruse execution hardening (binary allowlist, shell metacharacter rejection, shell=False enforcement)
2. Reasoning engine PRNG determinism and prompt-entropy seeding
3. Quillan-Ronin ONI forward pass KV-cache stepping and generation stability
"""

import os
import sys
import pytest
import torch

# Add paths to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

COMPUTERUSE_PATH = os.path.join(BASE_DIR, "09 - Projects", "projects", "Software Engineer")
if COMPUTERUSE_PATH not in sys.path:
    sys.path.insert(0, COMPUTERUSE_PATH)

REASONING_PATH = os.path.join(BASE_DIR, "06 - Deployment & Platforms", "Platforms", "GPT")
if REASONING_PATH not in sys.path:
    sys.path.insert(0, REASONING_PATH)

MODEL_PATH = os.path.join(BASE_DIR, "03 - Training & Model", "scripts")
if MODEL_PATH not in sys.path:
    sys.path.insert(0, MODEL_PATH)


class TestComputerUseHardening:
    def test_disallowed_binary_rejected(self):
        import Computeruse
        cu = Computeruse.ComputerUse()
        res1 = cu.exec("powershell -Command Get-Process")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res1
        assert "powershell" in res1

        res2 = cu.exec("whoami")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res2
        assert "whoami" in res2

    def test_shell_metacharacters_rejected(self):
        import Computeruse
        cu = Computeruse.ComputerUse()
        # Chaining with &
        res = cu.exec("git status & echo hacked")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res
        assert ("shell operators are forbidden" in res) or ("disallowed shell metacharacter" in res)

        # Chaining with ;
        res = cu.exec("python --version ; dir")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res
        assert ("shell operators are forbidden" in res) or ("disallowed shell metacharacter" in res)

        # Redirection with >
        res = cu.exec("python -c \"print('test')\" > output.txt")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res
        assert ("shell operators are forbidden" in res) or ("disallowed shell metacharacter" in res)

        # Subshell with `
        res = cu.exec("git `status`")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res
        assert ("shell operators are forbidden" in res) or ("disallowed shell metacharacter" in res)

    def test_allowed_binary_executes_safely(self):
        import Computeruse
        cu = Computeruse.ComputerUse()
        res = cu.exec("python --version")
        assert "Success: Executed" in res
        assert "Python" in res or "python" in res.lower()

    def test_custom_allowlist_via_env(self, monkeypatch):
        import Computeruse
        monkeypatch.setattr(Computeruse, "ALLOWED_EXEC_BINARIES", frozenset(["hostname", "calc"]))
        cu = Computeruse.ComputerUse()
        res = cu.exec("python --version")
        assert "Blocked by C13-WARDEN Perimeter Gate" in res
        assert "'python' is not in the exec allowlist" in res


class TestReasoningEngineDeterminism:
    def test_deterministic_seeded_trace(self):
        import reasoning_engine
        engine = reasoning_engine.ReasoningEngine(deterministic_seed=42)
        trace1 = engine.reason("Optimize the BitNet 1.58b STE kernel for AVX2")
        trace2 = engine.reason("Optimize the BitNet 1.58b STE kernel for AVX2")

        assert trace1["trace"] == trace2["trace"]
        assert trace1["entropy_score"] == trace2["entropy_score"]
        assert trace1["confidence"] == trace2["confidence"]

    def test_prompt_entropy_differentiation(self):
        import reasoning_engine
        engine = reasoning_engine.ReasoningEngine(deterministic_seed=42)
        trace_a = engine.reason("Prompt Alpha")
        trace_b = engine.reason("Prompt Beta")

        assert trace_a["trace"] != trace_b["trace"]


class TestModelKVCacheAndGeneration:
    def test_kv_cache_generation_lifecycle(self):
        from quillan_v5_4_oni import QuillanRoninOni, QuillanConfig

        cfg = QuillanConfig(vocab_size=80, hidden_dim=64, n_layer=2, n_head=4, max_seq_len=64)
        model = QuillanRoninOni(cfg)
        model.eval()

        prefix = [1, 2, 3]
        out_legacy = model._generate_legacy(
            prefix, max_tokens=4, temp=0.0, top_k=0, top_p=1.0,
            repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0
        )
        assert len(out_legacy) == len(prefix) + 4
        assert out_legacy[:3] == prefix

        out_sample = model._sample_continuation(
            prefix, max_tokens=4, temp=0.5, recirc=None, device=torch.device("cpu")
        )
        assert len(out_sample) > len(prefix)
        assert out_sample[:3] == prefix
        if len(out_sample) < len(prefix) + 4:
            # Stopped early upon generating eos_token_id
            assert out_sample[-1] == cfg.eos_token_id
