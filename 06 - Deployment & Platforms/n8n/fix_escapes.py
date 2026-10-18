#!/usr/bin/env python3
"""Double backslashes on flagged docstring lines (warning-free, no behavior change)."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\02_QUILLAN\03 - Training & Model\scripts"
FIX = {
    "compile_unified_sft_dataset.py": [42],
    "run_mini_20q_eval.py": [14, 15],
    "run_mini_sft_20q_eval.py": [14, 15],
    "run_mini_sft_eval_fixed.py": [6],
}
for fn, linenos in FIX.items():
    p = f"{BASE}\\{fn}"
    lines = open(p, encoding="utf-8").readlines()
    for ln in linenos:
        before = lines[ln - 1]
        lines[ln - 1] = before.replace("\\", "\\\\")
        print(f"{fn}:{ln}: {before.strip()[:60]}")
    open(p, "w", encoding="utf-8").writelines(lines)
print("fixed")
