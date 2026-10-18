#!/usr/bin/env python3
"""Targeted hunt: NVIDIA NIM (nvapi-) + OpenRouter (sk-or-v1-).
Prints provider + file + length ONLY. Never values."""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
SKIP = (".venv", "node_modules", ".git", "__pycache__", "site-packages",
        ".antigravity", ".codeium", "extensions", "Cache", "cache",
        "archive-v0")
PATS = {
    "nvidia-nim": re.compile(r"nvapi-[A-Za-z0-9\-_]{10,}"),
    "openrouter": re.compile(r"sk-or-v1-[A-Za-z0-9\-_]{10,}"),
}
NAMEPATS = re.compile(
    r"(NVIDIA_API_KEY|NVIDIA_NIM_API_KEY|NIM_API_KEY|OPENROUTER_API_KEY|"
    r"OR_API_KEY)\s*[:=]\s*['\"]?([^\s'\";]+)",
    re.I)
ROOTS = [r"C:\02_QUILLAN", os.path.expanduser("~")]


def scan_file(path):
    try:
        if os.path.getsize(path) > 300000:
            return
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return
    for prov, rx in PATS.items():
        for m in rx.finditer(txt):
            print(f"HIT provider={prov} file={path} len={len(m.group(0))}")
    for m in NAMEPATS.finditer(txt):
        print(f"HIT named {m.group(1)} file={path} value_len={len(m.group(2))}")


for root in ROOTS:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP and not d.startswith(".npm")]
        for fn in filenames:
            if (fn in (".env", ".env.local", ".env.production", "mcp.json",
                       "settings.json", "config.json", ".api-keys",
                       "secrets.json", "credentials.json")
                    or fn.endswith(".env") or fn.endswith(".json")
                    or fn.endswith(".txt") or fn.endswith(".md")):
                scan_file(os.path.join(dirpath, fn))
print("scan done")
