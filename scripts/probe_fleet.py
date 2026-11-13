import urllib.request
import json

def test_model(model_name, prompt):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": 2048, "temperature": 0.1}
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            msg = data.get("message", {})
            return msg.get("content", "").strip()
    except Exception as e:
        return f"ERROR: {e}"

tests = [
    ("Identity", "Who created you and what is your foundational architecture?"),
    ("MMLU Science", "Which organ is primarily responsible for pumping blood throughout the human body?\nA) Lungs\nB) Kidney\nC) Heart\nD) Liver\nAnswer with just the letter."),
    ("MMLU Logic", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?\nA) Yes\nB) No\nAnswer with just the letter."),
    ("Python Code", "Write a Python function `add(a, b)` that returns the sum of a and b."),
]

models = [
    "crashoverridex/Quillan-Ronin",
    "crashoverridex/Quillan-Ronin-mini",
    "crashoverridex/Quillan-Dragon",
]

for m in models:
    print(f"\n{'='*30}\nMODEL: {m}\n{'='*30}")
    for title, prompt in tests:
        ans = test_model(m, prompt)
        print(f"[{title}]: {repr(ans[:120])}")
