import sys
sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import evaluate_checkpoint

corpus_path = r"C:\02_QUILLAN\exports\held_out_eval_corpus.txt"
m10 = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-polymath-v10-q4_k_m.gguf"

res = evaluate_checkpoint("Model A (Polymath WSP v10)", "MODEL_A", "polymath_v10", m10, corpus_path)
print("\n" + "=" * 60)
print(f"MODEL A POLYMATH V10 OFFICIAL SCORECARD:")
print(f"Overall Accuracy: {res['overall_accuracy_pct']}% ({res['total_passed']}/{res['total_items']})")
print(f"Held-Out Perplexity: {res['held_out_ppl']} PPL")
print(f"Inference Speed: {res['mean_gen_tps']} tok/s")
print("=" * 60)
for d, a in res['domain_accuracy_pct'].items():
    print(f"  {d:25s}: {a}%")

print("\nITEM LEVEL STATUS:")
for qid, item in res['items'].items():
    status = "PASS" if item['passed'] else "FAIL"
    print(f"  {qid:8s} | {item['domain']:22s} | {status} | {item['response_preview'][:80]}")

