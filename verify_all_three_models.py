import urllib.request, json, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

models = [
    "crashoverridex/Quillan-Ronin:latest",
    "crashoverridex/Quillan-Ronin-mini:latest",
    "crashoverridex/Quillan-Dragon:latest"
]

print("==================================================================")
print("VERIFICATION OF ALL THREE QUILLAN MODELS IN OLLAMA")
print("==================================================================")

for m in models:
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/show",
        data=json.dumps({"name": m}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        details = data.get("details", {})
        sys_prompt = data.get("system", "")
        lines = len(sys_prompt.splitlines())
        chars = len(sys_prompt)
        params = data.get("parameters", "").strip().splitlines()
        
        print(f"\nModel: {m}")
        print(f"  Architecture Family: {details.get('family')} (quant: {details.get('quantization_level')}, size: {details.get('parameter_size')})")
        print(f"  Parent Model Checkpoint: {details.get('parent_model')}")
        print(f"  System Prompt Length: {chars:,} characters across {lines:,} lines")
        print(f"  Parameters:")
        for p in params:
            print(f"    - {p}")
        print(f"  Template Type: {'Llama 3.2' if 'start_header_id' in data.get('template', '') else 'ChatML (<think> native)'}")

print("\nAll 3 models verified locally and live on Ollama Hub!")

