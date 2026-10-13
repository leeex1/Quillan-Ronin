#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 GLOBAL CANONICAL SYNCHRONIZER ACROSS ALL PROMPTS & CONFIGS
---------------------------------------------------------------------------------------
Standardizes:
1. Canonical Production Version: v5.3.1
2. Active Training Frontier Version: v5.4
3. 34 Council Expert Personas: C0-ASTRA through C33-PREDATOR
4. Quillan (Core) as sovereign central orchestrator (Brainstem/Thalamus)
"""

import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path("c:/02_QUILLAN")
PROMPT_DIRS = [
    BASE_DIR / "system prompts",
    BASE_DIR / "system prompts" / "System prompts for models",
    BASE_DIR / "configs",
]

def clean_file(path: Path):
    if not path.is_file() or path.suffix not in [".md", ".yaml", ".json", ".py"]:
        return

    content = path.read_text(encoding="utf-8", errors="replace")
    orig = content

    # 1. Versioning cleanup: replace legacy versions with v5.3.1
    content = re.sub(r'Quillan\s+v[0-46-9]\.[0-9]+(?:\.[0-9]+)?', 'Quillan v5.3.1', content, flags=re.IGNORECASE)
    content = re.sub(r'Quillan-Ronin\s+v[0-46-9]\.[0-9]+(?:\.[0-9]+)?', 'Quillan-Ronin v5.3.1', content, flags=re.IGNORECASE)

    # 2. Council count alignment
    content = re.sub(r'(?:18|32|33)\s+Council\s+(?:Personas|Members|Experts)', '34 Council Experts (C0–C33)', content, flags=re.IGNORECASE)
    content = re.sub(r'\(C1[–-]C18\)', '(C0–C33)', content)
    content = re.sub(r'\(C1[–-]C32\)', '(C0–C33)', content)

    # 3. Update evaluation configs candidate checkpoints
    if path.name == "eval_config.yaml":
        content = re.sub(
            r'checkpoint_candidates:.*?(?=output_report:)',
            'checkpoint_candidates:\n    - "C:/02_QUILLAN/checkpoints/quillan_frontier_v2_best.pt"\n    - "C:/02_QUILLAN/checkpoints/quillan_ronin_v531_sovereign_production.pt"\n  ',
            content,
            flags=re.DOTALL
        )

    if content != orig:
        path.write_text(content, encoding="utf-8")
        print(f"✅ Synchronized: {path.relative_to(BASE_DIR)}")
    else:
        print(f"➖ Checked (Clean): {path.relative_to(BASE_DIR)}")


def main():
    total_files = 0
    for pdir in PROMPT_DIRS:
        if not pdir.exists():
            continue
        for f in pdir.glob("*"):
            if f.is_file():
                clean_file(f)
                total_files += 1

    print(f"\n🎉 Finished global canonical synchronization across {total_files} assets!")


if __name__ == "__main__":
    main()
