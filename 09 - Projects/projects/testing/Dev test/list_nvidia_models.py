#!/usr/bin/env python3
import requests, json

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

resp = requests.get(f"{API_BASE}/models", headers=headers, timeout=10)
if resp.status_code == 200:
    data = resp.json()
    models = [m["id"] for m in data.get("data", [])]
    print(f"Total available NVIDIA models: {len(models)}")
    # Find frontier models
    frontier = [m for m in models if any(k in m.lower() for k in ["405b", "70b", "nemotron", "deepseek", "qwen", "glm", "llama-3.1", "llama-3.3"])]
    print("\nTop Frontier Models Available:")
    for m in sorted(frontier):
        print("  -", m)
else:
    print(f"Failed to fetch models: {resp.status_code} - {resp.text}")
