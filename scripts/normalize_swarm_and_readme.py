#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize all swarm numbers across the codebase and docs to 9 Billion.
"""
import os
import shutil

ROOT = r"C:\02_QUILLAN"

# 1. Fix agents c20-c34 and quillan.agent.md
agents_dir = os.path.join(ROOT, "01 - Core Architecture", "agents")
for f in os.listdir(agents_dir):
    if f.endswith(".agent.md"):
        p = os.path.join(agents_dir, f)
        with open(p, "r", encoding="utf-8", errors="ignore") as fp:
            txt = fp.read()
        if "7,000" in txt or "7000" in txt:
            txt_fixed = txt.replace("**Micro-Swarm Configuration (7,000 Agents):**", "**Micro-Swarm Configuration (272M Agents / 9B Total Swarm):**")
            txt_fixed = txt_fixed.replace("7,000 Agents", "272,000,000 Agents (9 Billion Total Swarm)")
            txt_fixed = txt_fixed.replace("7,000 agents", "272,000,000 agents (9 Billion Total Swarm)")
            txt_fixed = txt_fixed.replace("7,000", "272,000,000 (9 Billion Total Swarm)")
            with open(p, "w", encoding="utf-8") as fp:
                fp.write(txt_fixed)
            print(f"Normalized {f}")

# 2. Fix eggroll.swarm.md in 01 - Core Architecture
egg_p = os.path.join(agents_dir, "quillan-swarm-files", "eggroll.swarm.md")
if os.path.exists(egg_p):
    with open(egg_p, "r", encoding="utf-8", errors="ignore") as fp:
        txt = fp.read()
    txt = txt.replace("7,000 agents", "active micro-agents (from the 9-billion virtual population)")
    with open(egg_p, "w", encoding="utf-8") as fp:
        fp.write(txt)
    print("Normalized eggroll.swarm.md")

# 3. Synchronize modernized swarm files into 04- quillan-sovereign-skills-and-swarm
src_swarm_dir = os.path.join(agents_dir, "quillan-swarm-files")
dst_swarm_dir = os.path.join(ROOT, "04- quillan-sovereign-skills-and-swarm", "quillan-swarm-files")
if os.path.exists(src_swarm_dir) and os.path.exists(dst_swarm_dir):
    # Remove old duplicate (2) files
    for f in os.listdir(dst_swarm_dir):
        if "(2)" in f:
            os.remove(os.path.join(dst_swarm_dir, f))
    # Copy modernized 36 files
    for f in os.listdir(src_swarm_dir):
        if f.endswith(".md"):
            shutil.copy2(os.path.join(src_swarm_dir, f), os.path.join(dst_swarm_dir, f))
    print(f"Synced modernized swarm manifests to {dst_swarm_dir}")

# 4. Fix Quillan-Samurai.md and quillan.md
for p in [
    os.path.join(ROOT, "06 - Deployment & Platforms", "system prompts", "Quillan-Samurai.md"),
    os.path.join(ROOT, "06 - Deployment & Platforms", "system prompts", "quillan.md")
]:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as fp:
            txt = fp.read()
        txt = txt.replace('SWARM_COUNCIL["\U0001f7e0 34 Council Nodes<br/>~7,000 agents per expert"]', 'SWARM_COUNCIL["\U0001f7e0 34 Council Nodes<br/>~264M micro-agents per expert (9 Billion total)"]')
        txt = txt.replace('~7,000 agents per expert', '~264M micro-agents per expert (9 Billion total)')
        with open(p, "w", encoding="utf-8") as fp:
            fp.write(txt)
        print(f"Normalized {os.path.basename(p)}")

# 5. Fix ECOSYSTEM.md files
for p in [
    os.path.join(ROOT, "02 - Knowledge Foundation", "ECOSYSTEM.md"),
    os.path.join(ROOT, "quillan-knowledge-base", "03-Lineage-and-Ecosystem", "ECOSYSTEM.md")
]:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8", errors="ignore") as fp:
            txt = fp.read()
        txt = txt.replace("7,000 agents per council persona, Throne, and Eggroll", "9 Billion micro-agents total across 35 swarms (~257M-272M per council persona)")
        txt = txt.replace("7,000 agents per council persona", "9 Billion micro-agents total across 35 swarms (~257M-272M per council persona)")
        with open(p, "w", encoding="utf-8") as fp:
            fp.write(txt)
        print(f"Normalized {p}")

print("Swarm normalization complete.")

