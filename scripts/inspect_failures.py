import sys
import os

sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import evaluate_checkpoint

corpus_path = r"C:\02_QUILLAN\exports\held_out_eval_corpus.txt"
models = [
    ("Model A Flagship v3", "MODEL_A", "wsp_v3", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"),
    ("Model A Polymath v8", "MODEL_A", "polymath_v8", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-polymath-v8-q4_k_m.gguf"),
    ("Model B Mini v3", "MODEL_B", "wsp_v3", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v3-q4_k_m.gguf"),
    ("Model C Dragon v3", "MODEL_C", "wsp_v3", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v3-q4_k_m.gguf")
]

for label, mid, var, path in models:
    res = evaluate_checkpoint(label, mid, var, path, corpus_path)
    acc = res.get("overall_accuracy_pct", 0)
    passed = res.get("total_passed", 0)
    total = res.get("total_items", 0)
    print(f"\n=======================================================")
    print(f"=== {label} === Acc: {acc}% ({passed}/{total})")
    print(f"=======================================================")
    items = res.get("items", {})
    for item_id, item_data in items.items():
        if not item_data.get("passed"):
            print(f"  FAILED: {item_id} ({item_data.get('domain')})")
            print(f"    Prompt: {item_data.get('prompt')[:90]}...")
            print(f"    Output: {repr(item_data.get('output'))}")

