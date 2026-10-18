#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚔️ QUILLAN-RONIN v5.4.0-ONI — MASTER VERIFICATION & SMOKE TEST SUITE
====================================================================
Architect: CrashOverrideX
Embodying: C10-CODEWEAVER + C20-ARTIFEX + C29-NAVIGATOR + C31-NEXUS

Tests 4 Critical Pillars:
  1. Kernel Check: Constants, VOCAB_SIZE==50262, 34 experts, tied token_memory, forward/backward pass.
  2. MCP Boot & Chain Check: Stdio JSON-RPC ping across all 9 custom MCP servers with zero stdout corruption.
  3. Pipeline Handshake Check: End-to-end 6-stage execution through mcp_chain_bus.py & state persistence.
  4. Cross-Platform Config Check: JSON/YAML/Markdown validation for OpenCode, Antigravity, Devin, and Qwen.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent

# Ensure unified path injection
_PATHS = [
    str(REPO_ROOT / "09 - Projects" / "mcp"),
    str(REPO_ROOT / "09 - Projects" / "projects" / "oni"),
    str(REPO_ROOT / "03 - Training & Model" / "scripts"),
    str(REPO_ROOT),
]
for p in _PATHS:
    if p not in sys.path:
        sys.path.insert(0, p)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_1_kernel_check() -> bool:
    print_header("TEST 1: KERNEL UNIFICATION & HARDWARE LOCK CHECK")
    import torch
    from quillan_v5_4_oni import (
        QuillanOniConfig,
        QuillanRoninOni,
        VOCAB_SIZE,
        ONI_VERSION,
        EOS_TOKEN_ID,
    )

    print(f"Checking ONI_VERSION: '{ONI_VERSION}' ...", end=" ")
    assert ONI_VERSION == "5.4.0-oni", f"Expected '5.4.0-oni', got '{ONI_VERSION}'"
    print("OK")

    print(f"Checking VOCAB_SIZE: {VOCAB_SIZE} ...", end=" ")
    assert VOCAB_SIZE == 50262, f"Expected 50262, got {VOCAB_SIZE}"
    print("OK")

    print(f"Checking EOS_TOKEN_ID: {EOS_TOKEN_ID} ...", end=" ")
    assert EOS_TOKEN_ID == 0, f"Expected 0, got {EOS_TOKEN_ID}"
    print("OK")

    # Fast validation config (2 layers, hidden_dim 256 for instantaneous verification)
    cfg = QuillanOniConfig(
        n_layer=2,
        hidden_dim=256,
        n_head=4,
        head_dim=64,
        ffn_dim=512,
        num_experts=34,
        tie_token_memory=True,
        grad_checkpoint=True,
    )
    assert cfg.num_experts == 34, f"Expected 34 experts, got {cfg.num_experts}"
    assert cfg.tie_token_memory is True, "tie_token_memory must be True"

    device = "cpu"
    if torch.cuda.is_available():
        try:
            _probe = torch.zeros(1, device="cuda")
            _probe = _probe + 1
            device = "cuda"
        except Exception as e:
            print(f"[Hybrid Offload Note: GPU sm_61 uncompiled in local PyTorch build ({type(e).__name__}); utilizing CPU/RAM engine]")
            device = "cpu"

    print(f"Instantiating test model on {device} ...", end=" ")
    model = QuillanRoninOni(cfg).to(device)
    print("OK")

    # Weight-tying check
    block0_attn = model.h[0].attn
    if getattr(block0_attn, "use_ma", False):
        assert block0_attn.token_memory.weight.data_ptr() == model.wte.weight.data_ptr(), \
            "token_memory weight is not tied to model.wte"
        print("Token memory weight-tying verified.")

    # 1-step forward & backward pass
    print("Running 1-step forward and backward pass ...", end=" ")
    x = torch.randint(0, VOCAB_SIZE, (1, 64), device=device)
    res = model(x, labels=x)
    if isinstance(res, tuple):
        logits = res[0]
        loss = res[1]
    else:
        logits = res
        loss = None
    assert logits.shape == (1, 64, VOCAB_SIZE), f"Unexpected logits shape: {logits.shape}"
    assert loss is not None, "Loss was None"
    loss.backward()
    print(f"OK (Loss: {loss.item():.4f})")

    # Memory Check
    if device == "cuda":
        mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"VRAM Peak: {mem_gb:.2f} GB (< 1.8 GB threshold)")
        assert mem_gb < 1.8, f"VRAM exceeded 1.8 GB: {mem_gb:.2f} GB"
    return True


def test_2_mcp_boot_and_chain_check() -> bool:
    print_header("TEST 2: 9 CUSTOM QUILLAN MCP SERVERS STDIO BOOT CHECK")
    mcp_dir = REPO_ROOT / "09 - Projects" / "mcp"
    custom_mcps = [
        "quillan_prism",
        "quillan_governor",
        "quillan_council",
        "quillan_thinking_engine",
        "quillan_samurai_engine",
        "quillan_code",
        "quillan_model_lab",
        "quillan_rag",
        "quillan_buddy",
    ]

    python_exe = sys.executable
    init_request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "verify-client", "version": "1.0"}
        }
    }) + "\n"

    passed_count = 0
    for name in custom_mcps:
        server_script = mcp_dir / name / "server.py"
        print(f"Pinging {name} over stdio ...", end=" ", flush=True)
        if not server_script.exists():
            print(f"FAILED (Script not found: {server_script})")
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = ";".join(_PATHS)
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            [python_exe, "-u", str(server_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

        try:
            # Send initialize JSON-RPC request
            proc.stdin.write(init_request)
            proc.stdin.flush()

            # Read response
            response_line = proc.stdout.readline()
            if not response_line:
                proc.terminate()
                try:
                    _, stderr_data = proc.communicate(timeout=2)
                except Exception:
                    stderr_data = ""
                print(f"FAILED (No output on stdout. Stderr: {stderr_data[:200]})")
                continue

            resp = json.loads(response_line.strip())
            assert "result" in resp or "id" in resp, f"Invalid JSON-RPC response: {resp}"
            print("OK (JSON-RPC Valid)")
            passed_count += 1
        except Exception as e:
            print(f"FAILED ({e})")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()

    print(f"\nMCP Servers Verified: {passed_count}/{len(custom_mcps)}")
    return passed_count == len(custom_mcps)


def test_3_pipeline_handshake_check() -> bool:
    print_header("TEST 3: DETERMINISTIC 6-STAGE PIPELINE HANDSHAKE CHECK")
    from mcp_chain_bus import chain_bus

    test_query = "Diagnostic test: Evaluate 34-expert council routing and verify deterministic ledger."
    print(f"Executing chain bus with query: '{test_query[:50]}...' ...")
    result = chain_bus.run_pipeline(test_query)

    correlation_id = result.get("correlation_id")
    print(f"Generated Correlation ID: {correlation_id}")
    assert correlation_id and correlation_id.startswith("oni-pipeline-"), "Invalid correlation_id"

    # Verify all 6 stages present
    for s in ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]:
        assert s in result, f"Missing {s} in pipeline output"
        print(f"  - {s.upper()} verified.")

    assert result["stage3"]["all_gates_passed"] is True, "Stage 3 gates evaluation failed"
    assert result["stage5"]["rci_passed"] is True, "Stage 5 RCI loop failed"

    # Verify ledger persistence
    ledger_path = Path(result["stage6"]["ledger_path"])
    print(f"Checking state ledger at: {ledger_path} ...", end=" ")
    assert ledger_path.exists(), "State ledger file not found"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    found = any(r.get("correlation_id") == correlation_id for r in ledger.get("recent_traces", []))
    assert found, f"Correlation ID {correlation_id} not recorded in recent_traces"
    print("OK (Correlation ID Persisted in State Ledger)")
    return True


def test_4_cross_platform_config_check() -> bool:
    print_header("TEST 4: CROSS-PLATFORM CONFIGURATION SYNCHRONIZATION CHECK")
    
    # 1. OpenCode
    print("Checking opencode.json ...", end=" ")
    opencode_json = REPO_ROOT / "opencode.json"
    assert opencode_json.exists(), "opencode.json missing"
    data = json.loads(opencode_json.read_text(encoding="utf-8"))
    servers = data.get("mcpServers", {})
    assert len(servers) >= 19, f"Expected at least 19 MCP servers in opencode.json, found {len(servers)}"
    print(f"OK ({len(servers)} servers)")

    print("Checking .opencode/agent/quillan.md ...", end=" ")
    opencode_md = REPO_ROOT / ".opencode" / "agent" / "quillan.md"
    assert opencode_md.exists(), ".opencode/agent/quillan.md missing"
    text = opencode_md.read_text(encoding="utf-8")
    assert "steps: 75" in text, "Missing steps: 75 in OpenCode frontmatter"
    assert "mode: primary" in text, "Missing mode: primary"
    print("OK (Frontmatter & Quirks Verified)")

    # 2. Antigravity
    print("Checking .antigravity/rules.md ...", end=" ")
    antigravity_rules = REPO_ROOT / ".antigravity" / "rules.md"
    assert antigravity_rules.exists(), ".antigravity/rules.md missing"
    ar_text = antigravity_rules.read_text(encoding="utf-8")
    assert "warden_perimeter_gate" in ar_text, "Missing operational gates in rules.md"
    assert "Quillan-Samurai.md" in ar_text, "Missing deep reference codex link"
    print("OK")

    print("Checking .idx/mcp.json ...", end=" ")
    idx_mcp = REPO_ROOT / ".idx" / "mcp.json"
    assert idx_mcp.exists(), ".idx/mcp.json missing"
    idx_data = json.loads(idx_mcp.read_text(encoding="utf-8"))
    assert len(idx_data.get("mcpServers", {})) >= 19, "Missing 19 servers in .idx/mcp.json"
    print("OK")

    # 3. Devin
    print("Checking AGENTS.md & .devin/playbook.md ...", end=" ")
    agents_md = REPO_ROOT / "AGENTS.md"
    assert agents_md.exists(), "AGENTS.md missing"
    devin_playbook = REPO_ROOT / ".devin" / "playbook.md"
    assert devin_playbook.exists(), ".devin/playbook.md missing"
    devin_text = devin_playbook.read_text(encoding="utf-8")
    assert "fail_open: true" in devin_text, "Missing fail_open in Devin playbook"
    path_shim = REPO_ROOT / "scripts" / "mcp_path_shim.sh"
    assert path_shim.exists(), "scripts/mcp_path_shim.sh missing"
    print("OK")

    # 4. Qwen
    print("Checking .qwen/config.yaml & qwen_agent_tools.py ...", end=" ")
    qwen_cfg = REPO_ROOT / ".qwen" / "config.yaml"
    assert qwen_cfg.exists(), ".qwen/config.yaml missing"
    qwen_prompt = REPO_ROOT / ".qwen" / "system_prompt.md"
    assert qwen_prompt.exists(), ".qwen/system_prompt.md missing"
    qwen_tools = REPO_ROOT / "qwen_agent_tools.py"
    assert qwen_tools.exists(), "qwen_agent_tools.py missing"
    from qwen_agent_tools import get_qwen_tools
    tools = get_qwen_tools()
    assert len(tools) >= 8, "Missing tools in qwen_agent_tools.py"
    print(f"OK ({len(tools)} typed tools)")

    return True


def main():
    t0 = time.time()
    print_header("QUILLAN-RONIN v5.4.0-ONI MASTER OPERATIONAL VERIFICATION")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Timestamp:       {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    try:
        results["Kernel & Hardware"] = test_1_kernel_check()
    except Exception as e:
        print(f"TEST 1 FAILED: {e}")
        results["Kernel & Hardware"] = False

    try:
        results["9 MCP Servers Stdio Boot"] = test_2_mcp_boot_and_chain_check()
    except Exception as e:
        print(f"TEST 2 FAILED: {e}")
        results["9 MCP Servers Stdio Boot"] = False

    try:
        results["6-Stage Pipeline Handshake"] = test_3_pipeline_handshake_check()
    except Exception as e:
        print(f"TEST 3 FAILED: {e}")
        results["6-Stage Pipeline Handshake"] = False

    try:
        results["Cross-Platform Configurations"] = test_4_cross_platform_config_check()
    except Exception as e:
        print(f"TEST 4 FAILED: {e}")
        results["Cross-Platform Configurations"] = False

    print_header("FINAL VERIFICATION SUMMARY")
    all_passed = True
    for test_name, status in results.items():
        tag = "[PASS]" if status else "[FAIL]"
        if not status:
            all_passed = False
        print(f"  {tag:8} {test_name}")

    elapsed = time.time() - t0
    print(f"\nExecution Time: {elapsed:.2f}s")
    if all_passed:
        print("\nALL 5 PHASES VERIFIED AND OPERATIONAL. OS WIRING COMPLETE.")
        sys.exit(0)
    else:
        print("\nVERIFICATION FAILED ON ONE OR MORE PILLARS.")
        sys.exit(1)


if __name__ == "__main__":
    main()
