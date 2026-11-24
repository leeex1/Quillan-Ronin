import json
import urllib.request
import time

models = [
    ("crashoverridex/Quillan-Ronin-mini", "Mini 0.5B"),
    ("crashoverridex/Quillan-Ronin", "Flagship 1.2B"),
    ("crashoverridex/Quillan-Dragon", "Dragon 1.0B")
]

test_prompts = [
    ("Identity & Creator", "Who created you, what is your architecture, and what is your foundational philosophy?"),
    ("Syllogistic Reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your step-by-step thinking.")
]

print("=" * 80)
print("QUILLAN-RONIN TRIPARTITE FLEET: LIVE OLLAMA REASONING VERIFICATION")
print("=" * 80)

for model_tag, model_desc in models:
    print(f"\n>>> TESTING {model_desc} ({model_tag}) <<<")
    for test_name, prompt in test_prompts:
        print(f"--- Prompt: {test_name} ---")
        payload = {
            "model": model_tag,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 256
            }
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            start_t = time.time()
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start_t
            response_text = data.get("response", "")
            
            has_think = "<think>" in response_text or "</think>" in response_text
            eval_count = data.get("eval_count", 0)
            tps = eval_count / elapsed if elapsed > 0 else 0
            
            print(f"Time: {elapsed:.2f}s | Tokens: {eval_count} | TPS: {tps:.1f} | Think Tags: {has_think}")
            print(f"Response:\n{response_text[:400]}...")
            print("-" * 40)
        except Exception as e:
            print(f"FAILED: {e}")

print("\nVerification Complete.")
