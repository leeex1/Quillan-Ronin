#!/usr/bin/env python3
"""Verify OpenRouter key with a free /models call. Prints status only."""
import json
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
txt = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
           errors="ignore").read()
m = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)", txt)
assert m, "key not found in file"
key = m.group(1)
req = urllib.request.Request(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"})
try:
    resp = urllib.request.urlopen(req, timeout=20)
    data = json.load(resp)
    models = data.get("data", [])
    print("OPENROUTER OK, models listed:", len(models))
    print("sample:", [x.get("id") for x in models[:6]])
except Exception as e:
    print("OPENROUTER FAIL:", str(e)[:200])
