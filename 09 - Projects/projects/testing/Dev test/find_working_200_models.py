#!/usr/bin/env python3
import requests

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

resp = requests.get(f"{API_BASE}/models", headers=headers, timeout=10)
all_models = [m["id"] for m in resp.json().get("data", [])]

working = []
print("Testing models for Status 200...")
for m in all_models[:30]:
    payload = {
        "model": m,
        "messages": [{"role": "user", "content": "1+1="}],
        "max_tokens": 5
    }
    try:
        r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=5)
        if r.status_code == 200:
            print(f"  [+] 200 OK: {m}")
            working.append(m)
        else:
            print(f"  [-] {r.status_code}: {m}")
    except Exception as e:
        print(f"  [-] Timeout: {m}")

print(f"\nTotal Working 200 OK Models: {len(working)}")
