import urllib.request
import json

payload = {
    "model": "crashoverridex/Quillan-Ronin-mini",
    "messages": [
        {"role": "user", "content": "Who created you? Answer in English."}
    ],
    "stream": False,
    "options": {"num_ctx": 2048, "temperature": 0.1}
}
req = urllib.request.Request(
    "http://127.0.0.1:11434/api/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=30) as resp:
    print(json.loads(resp.read().decode())["message"]["content"])
