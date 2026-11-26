#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-process smoke test. No sockets, no Ollama, no llama.cpp, no network.

  & 'C:\\Users\\Admin\\.local\\bin\\python3.11.exe' \\
      'C:\\02_QUILLAN\\09 - Projects\\mcp\\quillan_serve\\smoke_test.py'

Loads the verified 815M checkpoint, prints health, generates from a short prompt
through the repaired KV-cache path, and prints exact output + measured tok/s.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from engine import (  # noqa: E402
    ACTIVE_WINDOW_TOKENS,
    ContextWindowError,
    QuillanServingEngine,
)

PROMPT = "Quillan-Ronin is a sovereign"
NUM_PREDICT = 12


def main() -> int:
    engine = QuillanServingEngine()
    print("HEALTH=" + json.dumps(engine.health(), sort_keys=True))

    result = engine.generate(PROMPT, num_predict=NUM_PREDICT, temperature=0.0)

    print("PROMPT=" + repr(PROMPT))
    print("EXACT_OUTPUT=" + repr(result.response))
    print("TOKENS=" + str(result.eval_count))
    print("DONE_REASON=" + result.done_reason)
    print("EVAL_DURATION_NS=" + str(result.generation_duration_ns))
    print("TOKENS_PER_SEC=" + f"{result.eval_tokens_per_second:.4f}")
    print("RESULT=" + json.dumps(result.to_dict(), sort_keys=True))

    # Guard proof: an over-long request is REFUSED, never silently shortened.
    try:
        engine._guard([0] * ACTIVE_WINDOW_TOKENS, 1)
        guard_ok = False
    except ContextWindowError as exc:
        guard_ok = "NOT truncated" in str(exc)
        print("CONTEXT_GUARD=" + repr(str(exc)[:200]))
    print("CONTEXT_GUARD_OK=" + str(guard_ok))

    ok = (
        result.done
        and result.eval_count > 0
        and result.response.strip() != ""
        and guard_ok
    )
    print("SMOKE_PASS=" + str(ok))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
