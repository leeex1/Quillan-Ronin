#!/usr/bin/env python3
import requests

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

TEACHER_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-4-340b-instruct"
]

for m in TEACHER_MODELS:
    payload = {
        "model": m,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    try:
        r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=8)
        print(f"{m}: status={r.status_code}")
    except Exception as e:
        print(f"{m}: error={e}")
