import os
import json

base = r"C:\Users\Admin\.ollama\models\manifests"
for root, dirs, files in os.walk(base):
    for f in files:
        p = os.path.join(root, f)
        try:
            with open(p, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            for l in d.get("layers", []):
                if "model" in l.get("mediaType", ""):
                    size_mb = l.get("size", 0) / (1024 * 1024)
                    print(f"{os.path.relpath(p, base)} --> {l.get('digest')} ({size_mb:.1f} MB)")
        except Exception:
            pass

