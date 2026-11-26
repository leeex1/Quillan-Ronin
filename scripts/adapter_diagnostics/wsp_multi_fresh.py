#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the fresh-run validator N times in SEPARATE processes (true fresh reloads)."""

from __future__ import annotations
import subprocess, sys
from pathlib import Path

PY = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"
VAL = r"C:\02_QUILLAN\scripts\adapter_diagnostics\wsp_fresh_run.py"


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--bench", default="20")
    a = ap.parse_args()
    print(f"=== {a.runs} FRESH RUNS | {a.ckpt.name} | bench={a.bench} ===", flush=True)
    results = []
    for i in range(a.runs):
        out = subprocess.run(
            [PY, VAL, "--ckpt", str(a.ckpt), "--bench", a.bench, "--tag", f"#{i + 1}"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        line = [l for l in out.stdout.splitlines() if l.startswith("RUN")]
        print(
            line[0] if line else f"RUN #{i + 1}: ERROR {out.stderr[-200:]}", flush=True
        )
        if line:
            score = int(line[0].split(":")[1].split("/")[0].strip().split()[-1])
            results.append(score)
    if results:
        print(
            f"=== SUMMARY: {results} | all_20={all(r == 20 for r in results)} | min={min(results)} ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
