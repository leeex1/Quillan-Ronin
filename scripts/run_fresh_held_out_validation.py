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
import glob
import argparse

sys.path.append(r"C:\02_QUILLAN\scripts")
from held_out_benchmark_battery_v2 import FRESH_HELD_OUT_BATTERY
from run_professional_benchmark_suite import evaluate_suite_on_model

EXPORTS_DIR = r"C:\02_QUILLAN\exports"

def find_latest_gguf(pattern):
    files = glob.glob(os.path.join(EXPORTS_DIR, pattern))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]

def main():
    parser = argparse.ArgumentParser(description="Evaluate on Fresh Held-Out Battery (60 Unseen Probes)")
    parser.add_argument("--flagship", type=str, default=None, help="Path to Flagship GGUF")
    parser.add_argument("--dragon", type=str, default=None, help="Path to Dragon GGUF")
    parser.add_argument("--mini", type=str, default=None, help="Path to Mini GGUF")
    parser.add_argument("--out", type=str, default=os.path.join(EXPORTS_DIR, "fresh_held_out_benchmark_results.json"))
    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATING ON FRESH HELD-OUT BATTERY (60 PROBES - UNSEEN)")
    print("=" * 80)

    flagship_path = args.flagship or find_latest_gguf("quillan-1b-sovereign-wsp-*.gguf") or find_latest_gguf("*flagship*.gguf")
    dragon_path = args.dragon or find_latest_gguf("quillan-dragon-1b-wsp-*.gguf") or find_latest_gguf("*dragon*.gguf")
    mini_path = args.mini or find_latest_gguf("quillan-mini-0.5b-wsp-*.gguf") or find_latest_gguf("*mini*.gguf")

    models_to_test = []
    if flagship_path and os.path.exists(flagship_path):
        models_to_test.append(("Model A Flagship (1.2B)", flagship_path))
    if mini_path and os.path.exists(mini_path):
        models_to_test.append(("Model B Mini (0.5B)", mini_path))
    if dragon_path and os.path.exists(dragon_path):
        models_to_test.append(("Model C Dragon (1.0B)", dragon_path))

    results = {}
    for name, path in models_to_test:
        print(f"\n[*] Target: {name} -> {path}")
        res = evaluate_suite_on_model(name, path, FRESH_HELD_OUT_BATTERY)
        results[name] = res

    out_file = args.out
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Fresh held-out results saved to: {out_file}")

if __name__ == "__main__":
    main()

