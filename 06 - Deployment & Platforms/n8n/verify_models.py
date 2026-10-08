#!/usr/bin/env python3
"""List NIM models, then chat with a valid one + OR free exact id."""
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
        print(f"{tag} OK:",
              out["choices"][0]["message"]["content"].strip()[:120])
        return True
    except Exception as e:
        print(f"{tag} FAIL:", str(e)[:140])
        return False


try:
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/models",
        headers={"Authorization": f"Bearer {nv_key}"})
    models = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    print("NIM models:", len(models))
    llama = [m["id"] for m in models if "llama" in m["id"].lower()][:8]
    print("llama-ish:", llama)
except Exception as e:
    print("NIM LIST FAIL:", str(e)[:140])
    llama = []

if llama and not chat("https://integrate.api.nvidia.com/v1/chat/completions",
                       nv_key, llama[0], "NIM-CHAT"):
    pass
chat("https://openrouter.ai/api/v1/chat/completions", or_key,
     "apodex/apodex-1.1-mini:free", "OR-FREE")
