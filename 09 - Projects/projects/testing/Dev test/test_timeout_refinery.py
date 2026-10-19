#!/usr/bin/env python3
import requests, time

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
API_BASE = "https://integrate.api.nvidia.com/v1"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

SYSTEM_REFINER_PROMPT = (
    "You are a master AI engineer. Format your output as:\n"
    "<thought>[Concise decomposition and verification]</thought>\n"
    "<output>[Direct, brilliant final answer]</output>"
)

for m in ["meta/llama-3.1-8b-instruct", "meta/llama-3.1-70b-instruct"]:
    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": SYSTEM_REFINER_PROMPT},
            {"role": "user", "content": "Explain what a function is in Python in one clear sentence."}
        ],
        "max_tokens": 100,
        "temperature": 0.2
    }
    t0 = time.time()
    try:
        r = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload, timeout=25)
        print(f"[{m}] Status: {r.status_code} ({time.time()-t0:.1f}s)")
        if r.status_code == 200:
            print("Response:", r.json()["choices"][0]["message"]["content"][:200])
    except Exception as e:
        print(f"[{m}] Error ({time.time()-t0:.1f}s):", e)
