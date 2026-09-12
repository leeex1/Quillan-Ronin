#!/usr/bin/env python3
"""
Quillan Multi-Agent Sovereign Harness CLI
=========================================
Command-line dispatch and orchestration for the 10 Sovereign Agent Variants.
"""

import sys
import os
import argparse
from pathlib import Path

# Force UTF-8 on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.variants import AGENT_VARIANTS, list_variants
from harness.registry import default_registry

def print_banner():
    print("=" * 70)
    print(" 👑 QUILLAN-RONIN MULTI-AGENT SOVEREIGN HARNESS (10 VARIANTS)")
    print("=" * 70)

def cmd_list():
    print_banner()
    print(f"\n{'VARIANT':<12} | {'COUNCIL CHAMBER':<26} | {'ROLE':<30}")
    print("-" * 75)
    for v in list_variants():
        print(f"{v['name']:<12} | {v['council_chamber']:<26} | {v['role_title']:<30}")
        print(f"   ├─ Tools: {', '.join(v['authorized_tools'])}")
        print(f"   └─ Desc : {v['description']}\n")

def cmd_run_agent(agent_name: str, task: str):
    print_banner()
    agent_name = agent_name.lower().strip()
    if agent_name not in AGENT_VARIANTS:
        print(f"❌ Error: Unknown agent '{agent_name}'. Run with --list to see options.")
        return 1

    cfg = AGENT_VARIANTS[agent_name]
    print(f"\n[DISPATCHING] -> {cfg.name.upper()} ({cfg.role_title})")
    print(f"Council Chamber: {cfg.council_chamber}")
    print(f"Task: {task}")
    print("-" * 70)

    agent = default_registry.get(agent_name)
    resp = agent.run(task)

    print("\n" + "=" * 70)
    print(f" 🎯 [{resp.agent_name.upper()}] FINAL RESPONSE (in {resp.duration_sec:.2f}s, {resp.turns_taken} turns):")
    print("=" * 70)
    if resp.success:
        print(resp.final_answer)
        if resp.tools_invoked:
            print("\n[Audit - Tools Used]:")
            for t in resp.tools_invoked:
                print(f"  • Turn {t['turn']}: {t['tool']}({', '.join(t['args'])}) -> {t['result_preview']}")
        return 0
    else:
        print(f"❌ Execution failed: {resp.error}")
        return 1

def cmd_swarm(task: str):
    print_banner()
    print(f"\n[SWARM ORCHESTRATION INITIATED]")
    print(f"Task: {task}\n")
    
    # 1. Architect analyzes
    print(">>> 1/3: Architect analyzing system topology...")
    arch_resp = default_registry.dispatch("architect", f"Analyze and outline architectural strategy for: {task}")
    print(f"✅ Architect completed in {arch_resp.duration_sec:.2f}s\n")
    
    # 2. Coder formulates implementation
    print(">>> 2/3: Coder designing implementation based on architecture...")
    coder_prompt = f"Implement the solution for '{task}' adhering to this strategy:\n{arch_resp.final_answer[:1500]}"
    coder_resp = default_registry.dispatch("coder", coder_prompt)
    print(f"✅ Coder completed in {coder_resp.duration_sec:.2f}s\n")
    
    # 3. Security audits the implementation
    print(">>> 3/3: Security verifying implementation...")
    sec_prompt = f"Audit the following code/strategy for security flaws:\n{coder_resp.final_answer[:2000]}"
    sec_resp = default_registry.dispatch("security", sec_prompt)
    print(f"✅ Security audit completed in {sec_resp.duration_sec:.2f}s\n")

    print("=" * 70)
    print(" 🛡️ UNIFIED SWARM COUNCIL SYNTHESIS")
    print("=" * 70)
    print("### Architecture Plan:\n" + arch_resp.final_answer[:600] + "\n...")
    print("\n### Implementation:\n" + coder_resp.final_answer[:800] + "\n...")
    print("\n### Security Audit:\n" + sec_resp.final_answer[:600] + "\n...")
    return 0

def cmd_interactive():
    print_banner()
    current_agent = "governor"
    print(f"Interactive Harness Shell. Default agent: [{current_agent}]")
    print("Commands: /agent <name> (switch), /list (view all), /exit (quit)\n")
    
    while True:
        try:
            prompt = input(f"quillan-{current_agent}> ").strip()
            if not prompt:
                continue
            if prompt in ("/exit", "exit", "quit"):
                break
            if prompt.startswith("/agent "):
                new_agent = prompt.split(" ", 1)[1].strip().lower()
                if new_agent in AGENT_VARIANTS:
                    current_agent = new_agent
                    print(f"Switched active agent to: [{current_agent}] ({AGENT_VARIANTS[current_agent].role_title})")
                else:
                    print(f"Unknown agent '{new_agent}'. Available: {list(AGENT_VARIANTS.keys())}")
                continue
            if prompt == "/list":
                cmd_list()
                continue
            
            # Run task on current agent
            agent = default_registry.get(current_agent)
            print(f"Executing with {current_agent}...")
            resp = agent.run(prompt)
            print(f"\n[{current_agent}]:\n{resp.final_answer}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quillan Multi-Agent Sovereign Harness CLI")
    parser.add_argument("--list", action="store_true", help="List all 10 registered agent variants")
    parser.add_argument("--agent", type=str, default="", help="Select agent variant by name")
    parser.add_argument("--swarm", type=str, default="", help="Run 3-tier council swarm pipeline on task")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive multi-agent REPL shell")
    parser.add_argument("task", nargs="*", default=[], help="Task description for the agent")

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.swarm:
        sys.exit(cmd_swarm(args.swarm))
    elif args.interactive:
        cmd_interactive()
    elif args.agent:
        task_str = " ".join(args.task) if args.task else input("Enter task: ")
        sys.exit(cmd_run_agent(args.agent, task_str))
    else:
        # Default behavior: list variants and usage
        cmd_list()
        print("Usage examples:")
        print("  python quillan_harness.py --agent coder 'Check syntax of server.py'")
        print("  python quillan_harness.py --agent security 'Audit tools_extended.py'")
        print("  python quillan_harness.py --agent rag 'Search Bushido of Computation'")
        print("  python quillan_harness.py --swarm 'Design a high-throughput queue'")
        print("  python quillan_harness.py --interactive")

if __name__ == "__main__":
    main()
