#!/usr/bin/env python3
import os, sys, requests, json

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("[*] Testing NVIDIA API Connection...", flush=True)

# Test chat completion with a top model
payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [
        {"role": "system", "content": "You are a master teacher in software engineering and AI."},
        {"role": "user", "content": "Explain what a Python generator function is in 2 concise sentences."}
    ],
    "max_tokens": 100,
    "temperature": 0.2
}

try:
    resp = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=15)
    print(f"[+] Status Code: {resp.status_code}", flush=True)
    if resp.status_code == 200:
        res = resp.json()
        print("[+] NVIDIA Teacher Response:\n", res["choices"][0]["message"]["content"], flush=True)
    else:
        print("[-] Error Response:", resp.text, flush=True)
except Exception as e:
    print("[-] Request failed:", e, flush=True)
