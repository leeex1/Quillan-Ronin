import urllib.request
import json
import sys

def probe(model_name):
    print(f"\n--- Testing {model_name} ---")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is 15 * 4? Answer with just the number."}],
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.1}
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            print("Content:", repr(msg.get("content", "")))
            print("Thinking:", repr(msg.get("thinking", "")))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    probe("crashoverridex/Quillan-Ronin")
    probe("crashoverridex/Quillan-Ronin-mini")
    probe("crashoverridex/Quillan-Dragon")
