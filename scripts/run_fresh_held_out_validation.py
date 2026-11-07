#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: RUN FRESH HELD-OUT VALIDATION
=============================================================================
Evaluates models on the 60 completely fresh, unseen held-out probes.
=============================================================================
"""

import os
import sys
import json
import time

sys.path.append(r"C:\02_QUILLAN\scripts")
from held_out_benchmark_battery_v2 import FRESH_HELD_OUT_BATTERY
from run_professional_benchmark_suite import evaluate_suite_on_model

def main():
    print("=" * 80)
    print("EVALUATING ON FRESH HELD-OUT BATTERY (60 PROBES - UNSEEN)")
    print("=" * 80)
    
    models_to_test = [
        ("Model C Dragon (1.0B Teacher)", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v3-q4_k_m.gguf"),
        ("Model B Mini (0.5B Student)", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v3-q4_k_m.gguf"),
        ("Model A Flagship (1.2B Student)", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"),
    ]

    results = {}
    for name, path in models_to_test:
        if not os.path.exists(path):
            print(f"[WARN] File not found: {path}")
            continue
        res = evaluate_suite_on_model(name, path, FRESH_HELD_OUT_BATTERY)
        results[name] = res

    out_file = r"C:\02_QUILLAN\exports\fresh_held_out_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Fresh held-out results saved to: {out_file}")

if __name__ == "__main__":
    main()

