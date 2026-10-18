#!/usr/bin/env python3
import requests, json

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [
        {"role": "system", "content": "You are a master teacher."},
        {"role": "user", "content": "Explain what a function is in Python in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.3
}

print("Calling NVIDIA API...")
try:
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=10)
    print(f"Status: {r.status_code}")
    print("Response:", r.text[:300])
except Exception as e:
    print("Error:", e)
