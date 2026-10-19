#!/usr/bin/env python3
import requests

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_REFINER_PROMPT = (
    "You are a frontier reasoning amplifier. Your task is to take any raw input prompt and "
    "produce a structured reasoning trace.\n"
    "Format your output as:\n"
    "<thought>[Decomposition and proof]</thought>\n"
    "<output>[Final answer]</output>"
)

payload = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [
        {"role": "system", "content": SYSTEM_REFINER_PROMPT},
        {"role": "user", "content": "Explain what a function is in Python."}
    ],
    "max_tokens": 120,
    "temperature": 0.25
}

print("Testing refinery prompt...")
try:
    r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=10)
    print("Status:", r.status_code)
    print("Response:", r.text[:300])
except Exception as e:
    print("Error:", e)
