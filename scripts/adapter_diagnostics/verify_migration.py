#!/usr/bin/env python3
"""Verify C:\02_QUILLAN -> F:\02_QUILLAN copy: file count + bytes + key files."""

from pathlib import Path
import hashlib, sys, json

SRC = Path(r"C:\02_QUILLAN")
DST = Path(r"F:\02_QUILLAN")


def walk(root):
    n = 0
    b = 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                n += 1
                b += p.stat().st_size
        except (OSError, PermissionError):
            pass
    return n, b


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


print("counting source...", flush=True)
nc, bc = walk(SRC)
print(f"  source: {nc} files, {bc / 1e9:.2f} GB", flush=True)
print("counting destination...", flush=True)
nd, bd = walk(DST)
print(f"  dest:   {nd} files, {bd / 1e9:.2f} GB", flush=True)
print(f"  delta files={nd - nc}  delta_bytes={bd - bc}", flush=True)

# key files that must match exactly
keys = [
    r"AGENTS.md",
    r"SOUL.md",
    r"memory.md",
    r"01 - Core Architecture\IDENTITY.md",
    r"02 - Knowledge Foundation\LINEAGE.md",
    r"opencode.json",
    r"scripts\adapter_diagnostics\three_full_merges.py",
    r"exports\quillan-flagship-hifi-g4-exact.safetensors",
]
print("\nkey file hashes:", flush=True)
ok = True
for k in keys:
    a, b = SRC / k, DST / k
    if not a.exists():
        print(f"  MISSING SOURCE {k}", flush=True)
        continue
    if not b.exists():
        print(f"  MISSING DEST   {k}", flush=True)
        ok = False
        continue
    ha, hb = sha(a), sha(b)
    match = ha == hb
    ok &= match
    print(f"  {'OK ' if match else 'BAD'} {k}  {ha} / {hb}", flush=True)

verdict = (nc <= nd) and (bc <= bd) and abs(bd - bc) < 1e8 and ok
print(f"\nVERDICT={'PASS' if verdict else 'FAIL'}", flush=True)
