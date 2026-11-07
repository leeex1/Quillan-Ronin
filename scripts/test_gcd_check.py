import sys
sys.path.append(r"C:\02_QUILLAN\03 - Training & Model\scripts")
from phase0_held_out_eval_harness import _clean_code_snippet, check_code_gcd

out = """```python
def compute_gcd(a: int, b: int) -> int:
    \"\"\"
    Compute the greatest common divisor of two non-negative integers.

    Args:
        a (int): The first non-negative integer.
        b (int): The second non-negative integer.

    Returns:
        int: The greatest common divisor of a and b.
    \"\"\"

    # Ensure a and b are non-negative integers
    if a < 0 or b < 0:
        raise ValueError("Both a and b must be non-negative integers.")

    # Use the Euclidean algorithm to find the GCD of a and b
    while b:
        a, b = b, a % b

    # Return the GCD of a and b
    return a
"""

print("Testing _clean_code_snippet...")
cand = _clean_code_snippet(out)
print("Candidate:")
print(repr(cand))
res = check_code_gcd(out)
print("Result of check_code_gcd:", res)

