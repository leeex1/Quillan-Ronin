#!/usr/bin/env python3
"""Auto-find a working free OpenRouter model."""
import json
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
txt = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
           errors="ignore").read()
key = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)", txt).group(1)


def get(url, key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=30))


models = get("https://openrouter.ai/api/v1/models", key)["data"]
free = [m["id"] for m in models if m["id"].endswith(":free")]
print("free models:", len(free))
print(free[:10])

for mid in free[:6]:
    body = json.dumps({
        "model": mid,
        "messages": [{"role": "user", "content": "Reply with exactly: brain online"}],
        "max_tokens": 20}).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=90))
        content = (out["choices"][0]["message"].get("content") or "").strip()
        print(f"WINNER: {mid} -> [{content[:80]}]")
        open(r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\working_model.txt",
             "w").write(mid)
        break
    except Exception as e:
        print(f"nope {mid}: {str(e)[:100]}")
