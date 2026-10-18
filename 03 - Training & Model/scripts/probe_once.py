import json
import urllib.request
body = json.dumps({"model": "quillan-oni-mini-6l",
                   "messages": [{"role": "user",
                                 "content": "Say hello in one short sentence."}],
                   "max_tokens": 30}).encode()
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                             data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.loads(r.read().decode("utf-8"))
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print(repr(d["choices"][0]["message"]["content"]), flush=True)
print("usage=" + str(d.get("usage")), flush=True)
