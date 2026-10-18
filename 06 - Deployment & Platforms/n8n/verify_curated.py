#!/usr/bin/env python3
"""Try curated free models for a REAL answer."""
import json
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
txt = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
           errors="ignore").read()
key = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)", txt).group(1)

for mid in ["nvidia/nemotron-3.5-lightning:free",
            "liquid/lfm-2.5-2.6b:free",
            "cohere/north-mini-code:free",
            "inclusionai/ling-3.0-flash-sante:free"]:
    body = json.dumps({
        "model": mid,
        "messages": [{"role": "user",
                      "content": "What is 12 times 12? Reply with just the number."}],
        "max_tokens": 30}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=90))
        content = (out["choices"][0]["message"].get("content") or "").strip()
        print(f"{mid} -> [{content[:80]}]")
        if content:
            open(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\working_model.txt",
                 "w").write(mid)
            print("SAVED")
            break
    except Exception as e:
        print(f"{mid} FAIL: {str(e)[:100]}")
