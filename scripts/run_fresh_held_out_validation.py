#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: RUN FRESH HELD-OUT VALIDATION (60 UNSEEN PROBES)
=============================================================================
Supports Track A (raw baseline) vs Track B (Native WSP + Sovereign Cognitive
Harness: Quillan-Samurai.md + 5-Wave Penta-Process + WoT + Nemesis Gate).
=============================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
from held_out_benchmark_battery_v2 import FRESH_HELD_OUT_BATTERY
from run_professional_benchmark_suite import compute_wilson_ci, start_llama_server
from sovereign_cognitive_harness import resolve_fleet_ggufs, run_probe_with_harness

EXPORTS_DIR = r"C:\02_QUILLAN\exports"


def evaluate_suite_on_model(
    model_name: str,
    gguf_path: str,
    battery: List[Dict[str, Any]],
    track: str = "A",
    system_mode: str = "sovereign_core",
    port: int = 8089,
    ctx: int = 4096,
) -> Dict[str, Any]:
    """Evaluate a GGUF checkpoint on FRESH_HELD_OUT_BATTERY under Track A or Track B."""
    t0 = time.time()
    cat_results: Dict[str, List[int]] = {}
    logs: List[Dict[str, Any]] = []
    total_passed = 0

    with start_llama_server(gguf_path, port=port, ctx=ctx) as base_url:
        for idx, probe in enumerate(battery):
            cat = probe["category"]
            pid = probe["id"]
            prompt = probe["prompt"]
            checker = probe["checker"]
            max_toks = probe.get("max_tokens", 384)

            res = run_probe_with_harness(
                base_url=base_url,
                prompt=prompt,
                category=cat,
                max_tokens=max_toks,
                track=track,
                system_mode=system_mode,
                prompt_src=probe.get("prompt_src", ""),
                entry_point=probe.get("entry_point", ""),
            )
            graded_resp = res["graded_response"]
            try:
                passed = int(bool(checker(graded_resp)))
            except Exception:
                passed = 0

            total_passed += passed
            cat_results.setdefault(cat, []).append(passed)
            logs.append({
                "id": pid,
                "category": cat,
                "track": res["track"],
                "passed": passed,
                "latency": round(res["latency_s"], 3),
                "nemesis_retries": res["nemesis_retries"],
                "nemesis_notes": res["nemesis_notes"],
                "response": graded_resp,
                "raw_response": res["raw_response"],
            })
            if (idx + 1) % 10 == 0:
                print(f"  [{model_name} | Track {track}] {idx + 1}/{len(battery)} | passed={total_passed}")

    elapsed = round(time.time() - t0, 2)
    acc, lo, hi = compute_wilson_ci(total_passed, len(battery))
    cat_summary = {}
    for cat, vals in cat_results.items():
        c_acc, c_lo, c_hi = compute_wilson_ci(sum(vals), len(vals))
        cat_summary[cat] = {"passed": sum(vals), "total": len(vals), "accuracy": c_acc, "ci_95": [c_lo, c_hi]}

    return {
        "model_name": model_name,
        "track": f"Track {track}",
        "system_mode": system_mode if track.upper() == "B" else "raw",
        "gguf_path": gguf_path,
        "overall_accuracy": acc,
        "ci_95": [lo, hi],
        "total_passed": total_passed,
        "total_probes": len(battery),
        "elapsed_seconds": elapsed,
        "category_summary": cat_summary,
        "detailed_logs": logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on Fresh Held-Out Battery (60 Unseen Probes)")
    parser.add_argument("--flagship", type=str, default=None, help="Path to Flagship GGUF")
    parser.add_argument("--dragon", type=str, default=None, help="Path to Dragon GGUF")
    parser.add_argument("--mini", type=str, default=None, help="Path to Mini GGUF")
    parser.add_argument("--track", type=str, default="both", choices=["A", "B", "both"], help="Track A (raw), Track B (Cognitive Harness), or both")
    parser.add_argument("--system-mode", type=str, default="sovereign_core", choices=["sovereign_core", "full"], help="Track B system prompt mode")
    parser.add_argument("--ctx", type=int, default=4096, help="llama-server context size")
    parser.add_argument("--port", type=int, default=8089, help="llama-server port")
    parser.add_argument("--out", type=str, default=os.path.join(EXPORTS_DIR, "fresh_held_out_benchmark_v23_results.json"))
    args = parser.parse_args()

    fleet = resolve_fleet_ggufs(flagship=args.flagship, mini=args.mini, dragon=args.dragon)
    models_to_test = []
    if fleet["flagship"] and os.path.exists(fleet["flagship"]):
        models_to_test.append(("Model A Flagship (1.2B)", fleet["flagship"]))
    if fleet["mini"] and os.path.exists(fleet["mini"]):
        models_to_test.append(("Model B Mini (0.5B)", fleet["mini"]))
    if fleet["dragon"] and os.path.exists(fleet["dragon"]):
        models_to_test.append(("Model C Dragon (1.0B)", fleet["dragon"]))

    tracks = ["A", "B"] if args.track == "both" else [args.track]
    results: Dict[str, Any] = {}

    for name, path in models_to_test:
        for trk in tracks:
            key = f"{name} [Track {trk}]"
            print(f"\n[*] Target: {key} -> {os.path.basename(path)}")
            results[key] = evaluate_suite_on_model(
                name, path, FRESH_HELD_OUT_BATTERY, track=trk, system_mode=args.system_mode, port=args.port, ctx=args.ctx
            )
            print(f"    -> {key}: {results[key]['overall_accuracy']}% ({results[key]['total_passed']}/{results[key]['total_probes']})")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Fresh held-out results saved to: {args.out}")


if __name__ == "__main__":
    main()
