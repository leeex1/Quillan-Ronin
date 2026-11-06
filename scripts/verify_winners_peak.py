#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import evaluate_checkpoint

corpus_path = r"C:\02_QUILLAN\exports\held_out_eval_corpus.txt"

models = [
    ("Model B (0.5B Mini Winner)", "MODEL_B", "clean_base", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v3-q4_k_m.gguf"),
    ("Model C (1B Dragon Winner)", "MODEL_C", "clean_base", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v3-q4_k_m.gguf")
]

for label, mid, variant, path in models:
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING {label}", flush=True)
    print(f"Path: {path} ({os.path.getsize(path)/(1024*1024):.2f} MB)", flush=True)
    print(f"=======================================================", flush=True)
    res = evaluate_checkpoint(label, mid, variant, path, corpus_path)
    acc = res.get("overall_accuracy_pct")
    passed = res.get("total_passed")
    total = res.get("total_items")
    ppl = res.get("held_out_ppl")
    tps = res.get("mean_gen_tps")
    print(f"OVERALL RESULT: {acc}% ({passed}/{total} passed) | PPL: {ppl} | Speed: {tps} tok/s", flush=True)
    print("DOMAIN ACCURACIES:", flush=True)
    for d, dacc in res.get("domain_accuracy_pct", {}).items():
        print(f"  {d}: {dacc}%", flush=True)
    print("\nITEM LEVEL BREAKDOWN:", flush=True)
    items_dict = res.get("items", {})
    if isinstance(items_dict, dict):
        for iid, it in items_dict.items():
            st = "PASS" if it.get("passed") else "FAIL"
            dom = it.get("domain")
            resp = it.get("response_preview", "")
            print(f"  [{st}] {iid:6s} ({dom:22s}) | {resp[:100]}", flush=True)
