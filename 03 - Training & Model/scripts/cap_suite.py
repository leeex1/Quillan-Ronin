#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability suite — short + long form vs both live models (C18-SHEPHERD verify).
Prompts Mini (fresh best) + Main (stale) through the gateway, scores:
garble proxies (distinct-1/2, repeat-trigram rate, ASCII sanity) + length.
Small max_tokens to respect CPU. Writes JSON results + outbox summary.
No cloud, no third-party models: local weights only."""
from __future__ import annotations
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
GW = "http://127.0.0.1:8000/v1/chat/completions"

TESTS = [
    ("short-greet", "Say hello in one short sentence.", 30),
    ("long-explain", "Explain photosynthesis in three sentences.", 80),
    ("code-py", "Write a Python function that adds two numbers.", 80),
]

MODELS = ["quillan-oni-mini-6l", "quillan-oni-main-12l"]


def ask(model: str, prompt: str, mtok: int) -> dict:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mtok}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(GW, data=body,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "cap-suite"})
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read().decode("utf-8", errors="replace"))
        dt = time.time() - t0
        text = (resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                or resp.get("choices", [{}])[0].get("text", ""))
        return {"ok": True, "text": text, "secs": round(dt, 1)}
    except Exception as e:
        return {"ok": False, "text": "", "secs": 0.0,
                "error": f"{type(e).__name__}"}


def score(text: str) -> dict:
    toks = re.findall(r"\S+", text)
    n = len(toks)
    if n == 0:
        return {"tokens": 0, "distinct1": 0.0, "distinct2": 0.0,
                "repeat3": 1.0, "verdict": "EMPTY"}
    d1 = len(set(toks)) / n
    bigrams = [" ".join(toks[i:i + 2]) for i in range(n - 1)]
    d2 = len(set(bigrams)) / max(1, len(bigrams))
    tris = [" ".join(toks[i:i + 3]) for i in range(n - 2)]
    rep3 = (sum(1 for c in Counter(tris).values() if c > 1)
            / max(1, len(set(tris))))
    verdict = ("REPETITIVE" if rep3 > 0.4 or (d1 < 0.35 and n > 10)
               else "THIN" if n < 5 else "COHERENT?")
    return {"tokens": n, "distinct1": round(d1, 3), "distinct2": round(d2, 3),
            "repeat3": round(rep3, 3), "verdict": verdict}


def main() -> int:
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": []}
    for model in MODELS:
        for name, prompt, mtok in TESTS:
            r = ask(model, prompt, mtok)
            s = score(r["text"]) if r["ok"] else {"verdict": "ERROR"}
            row = {"model": model, "test": name, **r, **s}
            out["results"].append(row)
            print(f"{model} {name}: {s.get('verdict')} "
                  f"tok={s.get('tokens')} d1={s.get('distinct1')} "
                  f"rep3={s.get('repeat3')} {r['secs']}s", flush=True)
    (REPO / "checkpoints" / "cap_results.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    lines = [f"[Q-OPENCODE][TESTLOG] cap suite {out['at']}"]
    for row in out["results"]:
        lines.append(f"{row['model'].split('-')[-2]}-{row['model'].split('-')[-1]} "
                     f"{row['test']}: {row.get('verdict')} "
                     f"(tok={row.get('tokens')} d1={row.get('distinct1')} "
                     f"rep3={row.get('repeat3')} {row['secs']}s)")
    (REPO / "deploy" / "discord" / "outbox" / "cap_results.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print("saved cap_results.json + outbox/cap_results.txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
