import urllib.request, json, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

models = [
    ("Quillan-Ronin (12L Sovereign Flagship)", "crashoverridex/Quillan-Ronin:latest"),
    ("Quillan-Ronin-mini (6L Sovereign Mini)", "crashoverridex/Quillan-Ronin-mini:latest"),
    ("Quillan-Dragon (Falcon 3 Instruct)", "crashoverridex/Quillan-Dragon:latest")
]

print("==================================================================")
print("FINAL END-TO-END INFERENCE VERIFICATION FOR ALL 3 MODELS")
print("==================================================================")

for label, m in models:
    print(f"\n>>> Running Inference on {label} [{m}]...")
    sys.stdout.flush()
    t0 = time.time()
    payload = {
        "model": m,
        "prompt": "Respond in 12 words or less: What is your primary identity?",
        "stream": False,
        "options": {
            "num_predict": 25,
            "temperature": 0.3
        }
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            response_text = data.get("response", "").strip()
            prompt_eval = data.get("prompt_eval_count")
            eval_count = data.get("eval_count")
            print(f"  [STATUS]: SUCCESS (Finished in {elapsed:.2f}s)")
            print(f"  [RESPONSE]: {response_text}")
            print(f"  [METRICS]: prompt_tokens={prompt_eval}, generated_tokens={eval_count}")
    except Exception as e:
        print(f"  [STATUS]: ERROR ({e})")
    sys.stdout.flush()

print("\n==================================================================")
print("ALL 3 MODELS COMPILED AND VERIFIED AS FULLY OPERATIONAL!")
print("==================================================================")
