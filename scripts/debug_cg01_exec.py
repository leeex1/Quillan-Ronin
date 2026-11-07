import sys
sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
import phase0_held_out_eval_harness as harness

m = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-polymath-v8-q4_k_m.gguf"
p = "Write a Python function `def compute_gcd(a: int, b: int) -> int:` that returns the greatest common divisor of two non-negative integers. Return only the Python code block."

out, _, _ = harness.run_single_prompt(m, p, max_tokens=320)
print("=== OUT ===")
print(out)
print("=== CLEANED ===")
cand = harness._clean_code_snippet(out)
print(cand)
print("=== EXECUTION ===")
try:
    ns = {}
    exec(cand, ns)
    fn = ns.get("compute_gcd")
    print("fn:", fn)
    if fn:
        r1 = fn(48, 18)
        print("fn(48, 18):", r1, "== 6?", r1 == 6)
        r2 = fn(101, 10)
        print("fn(101, 10):", r2, "== 1?", r2 == 1)
        r3 = fn(0, 7)
        print("fn(0, 7):", r3, "== 7?", r3 == 7)
except Exception as e:
    import traceback
    traceback.print_exc()

