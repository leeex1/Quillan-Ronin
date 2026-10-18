#!/usr/bin/env python3
import urllib.request, json, time

url = "http://127.0.0.1:8000/v1/chat/completions"
payload = {
    "model": "quillan-v5.3.1",
    "messages": [
        {"role": "user", "content": "Hello! What is your name?"}
    ],
    "max_tokens": 15,
    "temperature": 0.65
}

t0 = time.time()
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print("\n=== LIVE OPENAI-COMPATIBLE API RESPONSE ===", flush=True)
        print(json.dumps(res, indent=2), flush=True)
        print(f"\n[+] Request completed in {time.time()-t0:.2f}s!\n", flush=True)
except Exception as e:
    print(f"[-] API Error: {e}", flush=True)
