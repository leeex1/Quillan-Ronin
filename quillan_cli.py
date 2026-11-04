"""
=============================================================================
⚔️ QUILLAN-RONIN: SOVEREIGN CLI & DEVELOPER HARNESS (OPENCODE ALTERNATIVE)
=============================================================================
Unified CLI entrypoint:
- `quillan`               : Launch Interactive TUI Command Center (Kabuto Banner)
- `quillan "<task>"`      : Autonomous Headless Coding Agent Execution
- `quillan --serve`       : Background OpenAI-Compatible API Server (:8080)
- `quillan --curl`        : Display & Execute Quillan Curl Quickstart
- `quillan --cu <action>` : Execute Google AI Studio Computer Use GUI action
- `quillan --install`     : Install `quillan` onto Windows PATH (~/.local/bin)
- `quillan --bench`       : Instant Hardware Speed & Throughput Benchmark
=============================================================================
"""

import os
import sys
import json
import shutil
import argparse
from typing import List

# UTF-8 Encoding Safeguard for Windows Consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = r"C:\02_QUILLAN"
sys.path.insert(0, ROOT_DIR)

from quillan_agent import SovereignCodingAgent
from quillan_engine import SovereignInferenceEngine, get_curl_command
from quillan_computer_use import QuillanComputerUseAgent


def install_global_cli():
    """Installs quillan.bat into user's PATH (~/.local/bin) for global terminal access."""
    target_dir = r"C:\Users\Admin\.local\bin"
    os.makedirs(target_dir, exist_ok=True)
    target_bat = os.path.join(target_dir, "quillan.bat")
    
    bat_content = f"""@echo off
set PYTHON_BIN="C:\\Users\\Admin\\.unsloth\\studio\\unsloth_studio\\Scripts\\python.exe"
%PYTHON_BIN% "{os.path.join(ROOT_DIR, 'quillan_cli.py')}" %*
"""
    with open(target_bat, "w", encoding="utf-8") as f:
        f.write(bat_content)
    
    print("=" * 72)
    print("⚔️ QUILLAN-RONIN CLI INSTALLED GLOBALLY!")
    print(f"📍 Location: {target_bat}")
    print("🚀 You can now type 'quillan' from ANY terminal in ANY directory!")
    print("=" * 72)


def run_headless_agent(prompt: str, model_id: str = "quillan-1b-sovereign-q4"):
    """Runs autonomous coding agent directly in the terminal with live streaming."""
    agent = SovereignCodingAgent()
    agent.engine.active_model_id = model_id
    
    print("\n" + "=" * 72)
    print(f"⚔️ QUILLAN-RONIN SOVEREIGN AGENT // TASK DISPATCH")
    print(f"📁 Workspace: {agent.workspace_root}")
    print(f"🤖 Model: {model_id}")
    print(f"🎯 Objective: {prompt}")
    print("=" * 72 + "\n")

    for chunk in agent.run_task(prompt):
        c_type = chunk.get("type")
        if c_type == "status":
            print(f"[STATUS] {chunk.get('text')}\n")
        elif c_type == "tool_call":
            tool_name = chunk.get("tool")
            res = chunk.get("result", {})
            print(f"\n⚡ [TOOL EXECUTION: {tool_name}]")
            print(json.dumps(res, indent=2))
        elif c_type == "token":
            sys.stdout.write(chunk.get("text", ""))
            sys.stdout.flush()
        elif c_type == "done":
            tps = chunk.get("tps", 0.0)
            print(f"\n\n✓ Task Complete | Generation Speed: {tps:.1f} tok/s\n")


def main():
    parser = argparse.ArgumentParser(
        description="Quillan-Ronin: Autonomous Sovereign Coding Agent (OpenCode Alternative)",
        add_help=False
    )
    parser.add_argument("prompt", nargs="?", default=None, help="Autonomous task prompt to execute")
    parser.add_argument("--serve", "-s", action="store_true", help="Boot OpenAI API server on :8080")
    parser.add_argument("--curl", "-c", action="store_true", help="Print and execute Quillan Curl")
    parser.add_argument("--cu", nargs="*", help="Execute Google AI Studio Computer Use GUI action")
    parser.add_argument("--install", action="store_true", help="Install global 'quillan' command on PATH")
    parser.add_argument("--bench", action="store_true", help="Run speed benchmark")
    parser.add_argument("--model", "-m", type=str, default="quillan-1b-sovereign-q4", help="Model ID")
    parser.add_argument("--help", "-h", action="store_true", help="Show help")

    args = parser.parse_args()

    if args.install:
        install_global_cli()
        return

    if args.serve:
        import quillan_server
        quillan_server.run_server()
        return

    if args.curl:
        print("\n⚡ [THE QUILLAN CURL QUICKSTART]\n")
        print(get_curl_command(args.model))
        print("\nExecute with: curl http://localhost:8080/v1/chat/completions ...\n")
        return

    if args.cu is not None:
        agent = QuillanComputerUseAgent()
        if len(args.cu) >= 3 and args.cu[0] in ["click", "move"]:
            action = args.cu[0]
            x, y = int(args.cu[1]), int(args.cu[2])
            res = agent.execute_action(action, {"x": x, "y": y, "intent": "CLI Computer Use"})
            print(f"[COMPUTER USE] {action} at ({x}, {y}) -> {res}")
        else:
            decl = agent.get_tool_declaration()
            print("[COMPUTER USE SCHEMA]:", json.dumps(decl, indent=2))
        return

    if args.bench:
        run_headless_agent("Calculate the prime factorization of 2048 and recite the virtues of Bushido.", model_id=args.model)
        return

    if args.help:
        print("""
============================================================================
⚔️ QUILLAN-RONIN: SOVEREIGN AGENT CLI USAGE (OPENCODE ALTERNATIVE)
============================================================================
  quillan                     : Launch interactive TUI Command Center (Kabuto Banner)
  quillan "<task prompt>"     : Execute autonomous coding/reasoning task in terminal
  quillan --serve             : Boot background OpenAI-compatible API server (:8080)
  quillan --curl              : Display & test the Quillan Curl one-liner
  quillan --cu <action> <x> <y>: Execute Google AI Studio Computer Use GUI action
  quillan --install           : Install 'quillan' globally on Windows PATH
  quillan --bench             : Benchmark generation throughput (tok/s)
  quillan --model <model_id>  : Select model (quillan-1b-sovereign-q4, deepseek-chat, gpt-4o)
============================================================================
""")
        return

    # If prompt provided -> run headless agent
    if args.prompt:
        run_headless_agent(args.prompt, model_id=args.model)
        return

    # Default -> Launch interactive TUI Command Center
    import quillan_tui
    quillan_tui.app = quillan_tui.QuillanCommandCenter()
    quillan_tui.app.run()


if __name__ == "__main__":
    main()
