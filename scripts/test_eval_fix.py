import sys
import os
import re

sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
import phase0_held_out_eval_harness as harness

# Override default max_tokens from 160 to 320
orig_run_single = harness.run_single_prompt

def patched_run_single(gguf_path: str, prompt: str, max_tokens: int = 320):
    return orig_run_single(gguf_path, prompt, max_tokens=max_tokens)

harness.run_single_prompt = patched_run_single

corpus_path = r"C:\02_QUILLAN\exports\held_out_eval_corpus.txt"
model_path = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"

print("Evaluating Model A Flagship with max_tokens=320...")
res = harness.evaluate_checkpoint("Model A Flagship v3", "MODEL_A", "wsp_v3", model_path, corpus_path)

print(f"\n=======================================================")
print(f"OVERALL ACCURACY: {res['overall_accuracy_pct']}% ({res['total_passed']}/{res['total_items']})")
print(f"PPL: {res['held_out_ppl']} | Speed: {res['mean_gen_tps']} tok/s")
print(f"=======================================================")
for d, acc in res['domain_accuracy_pct'].items():
    print(f"  {d}: {acc}%")

print("\nFAILURES:")
for k, v in res['items'].items():
    if not v['passed']:
        print(f"  FAILED: {k} ({v['domain']})")
        print(f"    Output: {repr(v.get('response_preview'))}")

