#!/usr/bin/env python3
"""
Quillan Multi-Agent Sovereign Harness CLI
=========================================
Command-line dispatch and orchestration for the 34 Sovereign Council Chambers (C0 - C33).
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

from harness.variants import AGENT_VARIANTS, list_variants, get_agent
from harness.registry import default_registry

def print_banner():
    print("=" * 78)
    print(" 👑 QUILLAN-RONIN SOVEREIGN PARLIAMENT (34 COUNCIL CHAMBERS C0 - C33)")
    print("=" * 78)

def cmd_list():
    print_banner()
    variants = list_variants()
    
    lobes = [
        ("CORE COGNITIVE LOBE (C0 - C9)", variants[0:10]),
        ("EQUILIBRIUM & REGULATION LOBE (C10 - C19)", variants[10:20]),
        ("RESEARCH & CREATION LOBE (C20 - C29)", variants[20:30]),
        ("META-GOVERNANCE & COMPETITIVE STRATEGY (C30 - C33)", variants[30:34]),
    ]
    
    for lobe_name, members in lobes:
        print(f"\n┌── {lobe_name} " + "─" * (74 - len(lobe_name)))
        print(f"│ {'CHAMBER':<12} | {'PERSONA':<12} | {'ROLE':<42}")
        print("├" + "─" * 74)
        for m in members:
            print(f"│ {m['chamber']:<12} | {m['name']:<12} | {m['role_title']:<42}")
            print(f"│    Tools: {', '.join(m['authorized_tools'])}")
        print("└" + "─" * 74)

    print("\n💡 Invocation Examples:")
    print("   python quillan_harness.py --agent c9 'Write a Python unit test'")
    print("   python quillan_harness.py --agent vir 'Audit ethical implications'")
    print("   python quillan_harness.py --agent predator 'Analyze competitor vulnerabilities'")
    print("   python quillan_harness.py --council 'Should we enable speculative decoding?'\n")

def cmd_run_agent(agent_identifier: str, task: str):
    print_banner()
    agent_clean = agent_identifier.lower().strip()
    if agent_clean not in AGENT_VARIANTS:
        print(f"❌ Error: Unknown Council chamber or agent '{agent_identifier}'.")
        print("Run with --list to see all 34 available chambers.")
        return 1

    cfg = AGENT_VARIANTS[agent_clean]
    print(f"\n[CHAMBER CONVOKED] -> {cfg.council_chamber} ({cfg.role_title})")
    print(f"Persona: {cfg.name.upper()}")
    print(f"Authorized Tools: {', '.join(cfg.tool_whitelist)}")
    print(f"Task: {task}")
    print("-" * 78)

    agent = default_registry.get(agent_clean)
    resp = agent.run(task)

    print("\n" + "=" * 78)
    print(f" 🎯 [{cfg.council_chamber}] FINAL VERDICT (in {resp.duration_sec:.2f}s, {resp.turns_taken} turns):")
    print("=" * 78)
    if resp.success:
        print(resp.final_answer)
        if resp.tools_invoked:
            print("\n[Chamber Audit - Tools Used]:")
            for t in resp.tools_invoked:
                print(f"  • Turn {t['turn']}: {t['tool']}({', '.join(t['args'])}) -> {t['result_preview']}")
        return 0
    else:
        print(f"❌ Execution failed: {resp.error}")
        return 1

def cmd_council_deliberation(task: str):
    print_banner()
    print(f"\n[TOP-4 SPARSE COUNCIL DELIBERATION INITIATED]")
    print(f"Task: {task}\n")
    
    active_chambers = [
        ("c0", "C0-ASTRA (Pattern Recognition)"),
        ("c6", "C6-LOGOS (Logical Deductions)"),
        ("c1", "C1-VIR (Ethical Guardian)"),
        ("c9", "C9-CODEWEAVER (Execution & Implementation)"),
    ]

    deliberations = {}
    for cid, label in active_chambers:
        print(f">>> Activating Chamber: {label}...")
        resp = default_registry.dispatch(cid, task)
        deliberations[cid] = resp
        print(f"    ✓ {cid.upper()} resolved in {resp.duration_sec:.2f}s ({resp.turns_taken} turns)")

    print("\n" + "=" * 78)
    print(" 🏛️ SOVEREIGN COUNCIL SYNTHESIS")
    print("=" * 78)
    for cid, label in active_chambers:
        ans = deliberations[cid].final_answer.strip()
        first_p = ans.split("\n\n")[0] if "\n\n" in ans else ans[:300]
        print(f"\n### [{label}]:\n{first_p}\n")
    return 0

def cmd_interactive():
    print_banner()
    current_agent = "c0"
    cfg = AGENT_VARIANTS[current_agent]
    print(f"Interactive Sovereign Parliament. Active Chamber: [{cfg.council_chamber}]")
    print("Commands: /switch <c0..c33 or name>, /list (show all 34), /exit\n")
    
    while True:
        try:
            curr_cfg = AGENT_VARIANTS[current_agent]
            prompt = input(f"[{curr_cfg.council_chamber}]> ").strip()
            if not prompt:
                continue
            if prompt in ("/exit", "exit", "quit"):
                break
            if prompt.startswith("/switch "):
                target = prompt.split(" ", 1)[1].strip().lower()
                if target in AGENT_VARIANTS:
                    current_agent = target
                    target_cfg = AGENT_VARIANTS[current_agent]
                    print(f"Switched active chamber to: [{target_cfg.council_chamber}] ({target_cfg.role_title})")
                else:
                    print(f"Unknown chamber '{target}'. Run /list to see all 34.")
                continue
            if prompt == "/list":
                cmd_list()
                continue
            
            # Execute on current chamber
            agent = default_registry.get(current_agent)
            print(f"Deliberating with {curr_cfg.council_chamber}...")
            resp = agent.run(prompt)
            print(f"\n[{curr_cfg.council_chamber}]:\n{resp.final_answer}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quillan Sovereign Council Harness CLI (34 Members)")
    parser.add_argument("--list", action="store_true", help="List all 34 Sovereign Council Chambers")
    parser.add_argument("--agent", "--chamber", dest="agent", type=str, default="", help="Select chamber by ID (c0..c33) or name")
    parser.add_argument("--council", "--deliberate", dest="council", type=str, default="", help="Run Top-4 Council deliberation")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive Council REPL shell")
    parser.add_argument("task", nargs="*", default=[], help="Task description for the chamber")

    args = parser.parse_args()

    if args.list:
        cmd_list()
    elif args.council:
        sys.exit(cmd_council_deliberation(args.council))
    elif args.interactive:
        cmd_interactive()
    elif args.agent:
        task_str = " ".join(args.task) if args.task else input("Enter task: ")
        sys.exit(cmd_run_agent(args.agent, task_str))
    else:
        cmd_list()

if __name__ == "__main__":
    main()
