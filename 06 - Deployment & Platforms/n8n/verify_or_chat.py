#!/usr/bin/env python3
"""Direct OpenRouter chat test (bypasses n8n). Key in memory only."""
import json
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
txt = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
           errors="ignore").read()
key = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)", txt).group(1)
body = json.dumps({
    "model": "meta-llama/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Reply with exactly: brain online"}],
    "max_tokens": 20}).encode()
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/chat/completions", data=body,
    headers={"Authorization": f"Bearer {key}",
             "Content-Type": "application/json"})
try:
    out = json.load(urllib.request.urlopen(req, timeout=60))
    print("DIRECT OK:", out["choices"][0]["message"]["content"].strip())
except Exception as e:
    print("DIRECT FAIL:", str(e)[:300])
