import subprocess, os, sys, time, urllib.request, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = os.environ.copy()
env["OLLAMA_HOST"] = "127.0.0.1:11434"

mf = """FROM exports/quillan-1b-sovereign-q4_k_m.gguf
TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"
PARAMETER temperature 0.65
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|end_of_text|>"
"""

with open(r"C:\02_QUILLAN\Modelfile.pure", "w") as f:
    f.write(mf)

print("Creating quillan-pure...")
subprocess.run(["ollama", "create", "quillan-pure:latest", "-f", r"C:\02_QUILLAN\Modelfile.pure"], env=env, capture_output=True)

print("Generating test with quillan-pure...")
t0 = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps({
        "model": "quillan-pure:latest",
        "prompt": "Hello! Introduce yourself in one sentence.",
        "stream": False,
        "options": {"num_predict": 60, "temperature": 0.65}
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print(f"Time: {time.time()-t0:.2f}s")
    print("RESPONSE:\n", res.get("response"))

subprocess.run(["ollama", "rm", "quillan-pure:latest"], env=env, capture_output=True)
if os.path.exists(r"C:\02_QUILLAN\Modelfile.pure"):
    os.remove(r"C:\02_QUILLAN\Modelfile.pure")

