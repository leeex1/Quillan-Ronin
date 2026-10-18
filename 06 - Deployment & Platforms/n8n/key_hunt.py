#!/usr/bin/env python3
"""Scan likely spots for LLM API keys. Prints provider + file + length ONLY.
Never prints values."""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
SKIP = (".venv", "node_modules", ".git", "__pycache__", "site-packages",
        ".antigravity", ".codeium", "extensions", "Cache", "cache")
PATS = {
    "anthropic": re.compile(r"sk-ant-[A-Za-z0-9\-_]{10,}"),
    "openai-proj": re.compile(r"sk-proj-[A-Za-z0-9\-_]{10,}"),
    "openai-svc": re.compile(r"sk-svcacct-[A-Za-z0-9\-_]{10,}"),
    "openai-legacy": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "xai": re.compile(r"xai-[A-Za-z0-9]{20,}"),
    "groq": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "gemini": re.compile(r"AIza[A-Za-z0-9\-_]{20,}"),
    "mistral": re.compile(r"[A-Za-z0-9]{32}"),
}
NAMEPATS = re.compile(
    r"(OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|"
    r"XAI_API_KEY|GROQ_API_KEY|MISTRAL_API_KEY|DEEPSEEK_API_KEY|"
    r"OPENROUTER_API_KEY)\s*[:=]\s*['\"]?([^\s'\";]+)",
    re.I)
ROOTS = [r"C:\02_QUILLAN", os.path.expanduser("~")]
NAMES = (".env", ".env.local", ".env.production", "mcp.json", "settings.json",
         "config.json", ".api-keys", "secrets.json", "credentials.json",
         "openai.json", "anthropic.json")


def scan_file(path):
    try:
        if os.path.getsize(path) > 200000:
            return
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return
    for prov, rx in PATS.items():
        m = rx.search(txt)
        if m and prov != "mistral":
            print(f"HIT value-pattern provider={prov} file={path} len={len(m.group(0))}")
    for m in NAMEPATS.finditer(txt):
        print(f"HIT named {m.group(1)} file={path} value_len={len(m.group(2))}")


for root in ROOTS:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP
                       and not d.startswith(".npm")]
        if "node_modules" in dirpath or ".venv" in dirpath:
            continue
        for fn in filenames:
            if fn in NAMES or fn.endswith(".env"):
                scan_file(os.path.join(dirpath, fn))
print("scan done")
