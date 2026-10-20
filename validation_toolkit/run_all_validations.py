#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_validations.py - Master Validation Suite Runner for Quillan-Ronin v5.4.0-ONI
Runs:
  Suite 1: 23 Mathematical Formulas & Computational Invariants (test_mathematical_wiring.py)
  Suite 2: BitNet 1.58b, Offload & Checkpoint Integrity (test_bitnet_and_offload_packs.py)
  Suite 3: Standalone Tools, Gates & Hardware Governance (test_standalone_tools_pack.py)
"""

import sys
import time
import importlib.util
from pathlib import Path

VAL_DIR = Path(__file__).resolve().parent


def load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_master_validation() -> int:
    t0 = time.time()
    print("\n" + "#" * 75)
    print("  QUILLAN-RONIN v5.4.0-ONI: MASTER SYSTEM INTEGRITY & VALIDATION SUITE")
    print("#" * 75 + "\n")

    mod1 = load_module_from_file("val_suite_1", VAL_DIR / "test_mathematical_wiring.py")
    res1 = mod1.test_suite_mathematical_wiring()
    print("\n")

    mod2 = load_module_from_file("val_suite_2", VAL_DIR / "test_bitnet_and_offload_packs.py")
    res2 = mod2.test_suite_bitnet_and_offload()
    print("\n")

    mod3 = load_module_from_file("val_suite_3", VAL_DIR / "test_standalone_tools_pack.py")
    res3 = mod3.test_suite_standalone_tools()
    print("\n")

    elapsed = time.time() - t0

    print("#" * 75)
    print("  FINAL VALIDATION SCORECARD & CERTIFICATION MATRIX")
    print("#" * 75)
    print(f"  Suite 1: 23 Mathematical Formulas & Invariants : {'PASS' if res1 else 'FAIL'}")
    print(f"  Suite 2: BitNet 1.58b, Offload & Checkpoint     : {'PASS' if res2 else 'FAIL'}")
    print(f"  Suite 3: Standalone Tools & Hardware Governance: {'PASS' if res3 else 'FAIL'}")
    print("-" * 75)
    all_passed = res1 and res2 and res3
    print(f"  OVERALL RESULT : {'100% PASS - ALL PILLARS HARDENED AND OPERATIONAL' if all_passed else 'FAIL'}")
    print(f"  Execution Time : {elapsed:.2f} seconds")
    print("#" * 75 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_master_validation())
