#!/usr/bin/env python3
"""Try OpenRouter :free model + NVIDIA NIM key. Keys in memory only."""
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


def post(url, key, payload, tag):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=60))
        print(f"{tag} OK:",
              out["choices"][0]["message"]["content"].strip()[:120])
    except Exception as e:
        print(f"{tag} FAIL:", str(e)[:160])


post("https://openrouter.ai/api/v1/chat/completions", or_key,
     {"model": "meta-llama/llama-3.1-8b-instruct:free",
      "messages": [{"role": "user",
                    "content": "Reply with exactly: brain online"}],
      "max_tokens": 20},
     "OR-FREE")
post("https://integrate.api.nvidia.com/v1/chat/completions", nv_key,
     {"model": "meta/llama-3.1-8b-instruct",
      "messages": [{"role": "user",
                    "content": "Reply with exactly: brain online"}],
      "max_tokens": 20},
     "NIM")
