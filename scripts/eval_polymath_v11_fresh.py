import os
import sys
import json

sys.path.append(r"C:\02_QUILLAN\scripts")
from held_out_benchmark_battery_v2 import FRESH_HELD_OUT_BATTERY
from run_professional_benchmark_suite import evaluate_suite_on_model

m11 = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-polymath-v11-q4_k_m.gguf"
res = evaluate_suite_on_model("Model A Flagship (Polymath v11)", m11, FRESH_HELD_OUT_BATTERY)
print("POLYMATH V11 OVERALL ACCURACY:", res["overall_accuracy"], f"({res['total_passed']}/{res['total_probes']})")
print("CATEGORY BREAKDOWN:", res["category_summary"])

