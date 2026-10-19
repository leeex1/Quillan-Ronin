#!/usr/bin/env python3
"""Try known-good current models on both providers."""
import json
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
or_txt = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
              errors="ignore").read()
or_key = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)",
                   or_txt).group(1)
nv_txt = open(r"C:\02_QUILLAN\.env", encoding="utf-8",
              errors="ignore").read()
nv_key = re.search(r"nvapi-[A-Za-z0-9\-_]{10,}", nv_txt).group(0)


def chat(url, key, model, tag):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": "Reply with exactly: brain online"}],
        "max_tokens": 20}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=60))
        txt = (out["choices"][0]["message"].get("content") or "").strip()
        print(f"{tag} OK: [{txt[:80]}]")
    except Exception as e:
        print(f"{tag} FAIL:", str(e)[:140])


chat("https://integrate.api.nvidia.com/v1/chat/completions", nv_key,
     "meta/llama-3.3-70b-instruct", "NIM-70B")
chat("https://openrouter.ai/api/v1/chat/completions", or_key,
     "meta-llama/llama-3.3-70b-instruct:free", "OR-70B-FREE")
