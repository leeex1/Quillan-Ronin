"""Real timing of NextVerse formula computations.

No random inputs, no sleep, no simulated speedups. Measures wall-clock execution
of the deterministic formula engine on this host and writes a timestamped JSON
artifact. This measures compute cost only; it does NOT validate any claimed
360x-5000x hardware advantage.
"""

import json
import os
import time
from formula_engine import NextVerseFormulaEngine

ITERS = 200
engine = NextVerseFormulaEngine()
results = {}
for name in engine.formulas:
    fn = getattr(engine, "calculate_" + name.lower())
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    elapsed = time.perf_counter() - t0
    results[name] = {
        "iterations": ITERS,
        "total_seconds": round(elapsed, 6),
        "us_per_eval": round(elapsed / ITERS * 1e6, 3),
        "evals_per_sec": round(ITERS / elapsed, 1),
    }
    print(
        f"{name}: {results[name]['us_per_eval']} us/eval -> {results[name]['evals_per_sec']}/s"
    )

out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nextverse_reality_check.json"
)
with open(out, "w", encoding="utf-8") as f:
    json.dump(
        {
            "note": "Measured on this host. Compute throughput only; no hardware speedup claim validated.",
            "python": __import__("sys").version,
            "results": results,
        },
        f,
        indent=2,
    )
print("report ->", out)
