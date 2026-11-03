import urllib.request
import json
import os
import gguf

print("Checking Ollama API...")
try:
    res = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
    tags = json.loads(res.read())
    print("Ollama API online. Models:")
    for m in tags.get("models", []):
        print(f"  {m['name']} -> {m['size']/1e9:.2f} GB")
except Exception as e:
    print("Ollama API not running / error:", e)

print("\nChecking Ollama Manifests on disk:")
m_root = r"C:\Users\Admin\.ollama\models\manifests"
for root, dirs, files in os.walk(m_root):
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, m_root)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                layers = [l.get("digest") for l in data.get("layers", []) if l.get("mediaType") == "application/vnd.ollama.image.model"]
                print(f"  {rel} -> {layers}")
        except Exception:
            pass
