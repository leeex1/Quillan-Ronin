import sys
sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import run_single_prompt

p = 'Write concise Python. No docstring. No comments. Define ONLY the requested function.\ndef monotonic(l: list):\n    """Return True is list elements are monotonically increasing or decreasing."""'
for tag, path in [
    ("polymath-v11", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-polymath-v11-q4_k_m.gguf"),
    ("wsp-v3", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"),
    ("clean-base", r"C:\02_QUILLAN\exports\quillan-1b-clean-base-q4_k_m.gguf"),
]:
    ans, _, _ = run_single_prompt(path, p, max_tokens=200)
    print("===", tag, "===")
    print(ans)

