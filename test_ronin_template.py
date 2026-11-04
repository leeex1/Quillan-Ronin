import subprocess, os, sys, time, urllib.request, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
env = os.environ.copy()
env["OLLAMA_HOST"] = "127.0.0.1:11434"

# Read canonical Modelfile
with open(r"C:\02_QUILLAN\Modelfile", "r", encoding="utf-8") as f:
    full_content = f.read()

system_pos = full_content.find('SYSTEM """')
system_block = full_content[system_pos:]

# Modern Llama 3 template with proper message handling and sampling
header = """FROM exports/quillan-1b-sovereign-q4_k_m.gguf

TEMPLATE \"\"\"{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- end }}<|start_header_id|>assistant<|end_header_id|>

\"\"\"

PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 32768
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|end_of_text|>"

"""

with open(r"C:\02_QUILLAN\Modelfile.test_template", "w", encoding="utf-8", newline="\n") as f:
    f.write(header + system_block)

print("Creating test-ronin-template...")
subprocess.run(["ollama", "create", "test-ronin-template:latest", "-f", r"C:\02_QUILLAN\Modelfile.test_template"], env=env, capture_output=True)

print("Testing generation on test-ronin-template...")
t0 = time.time()
req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=json.dumps({
        "model": "test-ronin-template:latest",
        "prompt": "Respond in 10 words or less: Who are you?",
        "stream": False,
        "options": {"num_predict": 40, "temperature": 0.6}
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=120) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print(f"Time: {time.time()-t0:.2f}s")
    print("RESPONSE:\n", res.get("response"))

subprocess.run(["ollama", "rm", "test-ronin-template:latest"], env=env, capture_output=True)
if os.path.exists(r"C:\02_QUILLAN\Modelfile.test_template"):
    os.remove(r"C:\02_QUILLAN\Modelfile.test_template")

