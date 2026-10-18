#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUILLAN-RONIN — Google Drive Knowledge Base Synchronization Tool
=================================================================
Synchronizes and packages the Quillan Knowledge Foundation into the
Google Drive On-Demand Knowledge Base structure for seamless query
resolution via `@Google Drive` in Gemini and Google Workspace.
"""

import os
import shutil
import zipfile
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
DESKTOP_ROOT = os.path.join(os.path.expanduser("~"), "Desktop")
SRC_KF = os.path.join(WORKSPACE_ROOT, "02 - Knowledge Foundation")
SRC_CORE_KB = os.path.join(SRC_KF, "Quillan Knowledge files")
OUT_KB = os.path.join(WORKSPACE_ROOT, "quillan-knowledge-base")
OUT_DEPLOY = os.path.join(WORKSPACE_ROOT, "06 - Deployment & Platforms", "deploy")

KB_CATEGORIES = {
    "01-Architecture-and-Kernel": [
        "1-Quillan_architecture_flowchart.md",
        "3-Quillan(reality).md",
        "9-Quillan Brain mapping.md",
        "Code Dependency Map.md",
        "QUILLAN_FULL_AUDIT_REPORT.md"
    ],
    "02-Formulas-and-Math": [
        "8-Formulas.md",
        "Must know formulas.md",
        "11-Drift Paper.md",
        "12-Multi-Domain Theoretical Breakthroughs Explained.md",
        "16-Emergent Goal Formation Mech.md"
    ],
    "03-Lineage-and-Ecosystem": [
        "ECOSYSTEM.md",
        "LINEAGE.md",
        "FAQ.md",
        "SERVICES.md",
        "BOOTSTRAP.md",
        "31- Autobiography.md",
        "External Platform Connections.md"
    ],
    "04-Council-and-Swarms": [
        "0-Quillan Loader Manifest.md",
        "10- Quillan Persona Manifest.md",
        "5-ai persona research.md",
        "28-Multi-Agent Collective Intelligence & Social Simulation.md",
        "7-memories.md"
    ],
    "05-Cognition-and-Epistemology": [
        "13-Synthetic Epistemology & Truth Calibration Protocol.md",
        "14-Ethical Paradox Engine and Moral Arbitration Layer in AGI Systems.md",
        "15-Anthropic Modeling & User Cognition Mapping.md",
        "17-Continuous Learning Paper.md",
        "18-Novelty Explorer Agent.md",
        "21- deep research functions.md",
        "29-Recursive Introspection & Meta-Cognitive Self-Modeling.md",
        "30- Convergence Reasoning & Breakthrough Detection and Advanced Cognitive Social Skills.md",
        "32-Conciousness theory.md",
        "6-prime_covenant_codex.md"
    ],
    "06-Operational-Manuals-and-Applications": [
        "27-Quillan operational manual.md",
        "20-Multidomain AI Applications.md",
        "22-Emotional Intelligence and Social Skills.md",
        "23-Creativity and Innovation.md",
        "24-Explainability and Transparency.md",
        "25-Human-Computer Interaction (HCI) and User Experience (UX).md",
        "26-Subjectve experiences and Qualia in AI and LLMs.md",
        "4-Lee X-humanized Integrated Research Paper.md"
    ]
}

def sync():
    print("[*] Synchronizing Quillan Knowledge Base for Google Drive...")
    os.makedirs(OUT_KB, exist_ok=True)
    os.makedirs(OUT_DEPLOY, exist_ok=True)

    copied = 0
    for cat, flist in KB_CATEGORIES.items():
        cat_dir = os.path.join(OUT_KB, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for fname in flist:
            found_path = None
            for root in [SRC_CORE_KB, SRC_KF]:
                for r, d, fs in os.walk(root):
                    if fname in fs:
                        found_path = os.path.join(r, fname)
                        break
                if found_path:
                    break
            
            if found_path:
                with open(found_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                clean_name = fname.replace(".md", "").replace("-", " ")
                header = f"""<!--
GOOGLE DRIVE KNOWLEDGE BASE SEARCH ANCHOR
Topic: {cat}
Document: {fname}
Domain: Quillan-Ronin Sovereign AI Architecture (v5.4.0-ONI)
Search Keywords: {clean_name}, {cat.replace('-', ' ')}
-->

"""
                target_file = os.path.join(cat_dir, fname)
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(header + content)
                copied += 1
            else:
                print(f"[!] Warning: Missing source file: {fname}")

    print(f"[+] Synced {copied} knowledge files into {OUT_KB}")

    # Mirror to Desktop
    desktop_kb = os.path.join(DESKTOP_ROOT, "quillan-knowledge-base")
    if os.path.exists(desktop_kb):
        shutil.rmtree(desktop_kb)
    shutil.copytree(OUT_KB, desktop_kb)
    print(f"[+] Mirrored to Desktop: {desktop_kb}")

    # Package zip
    kb_zip = os.path.join(OUT_DEPLOY, "quillan-knowledge-base.zip")
    with zipfile.ZipFile(kb_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for r, ds, fs in os.walk(OUT_KB):
            for f in fs:
                fp = os.path.join(r, f)
                rel = os.path.relpath(fp, os.path.dirname(OUT_KB))
                z.write(fp, rel)
    print("[+] Packaged deployment zip: " + str(kb_zip))
    print("[OK] Knowledge base sync complete!")

if __name__ == "__main__":
    sync()
