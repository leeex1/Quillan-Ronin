import os
from pathlib import Path
DKEYS = ("llama", "bitnet", "ggml", "engine", "inference", "quant", "server",
         "vllm", "onnx", "tensorrt", "trt", "openvino")
SKIP = ("node_modules", "__pycache__", ".venv", "open-webui", "Nextverse",
        "site-packages", ".git", "emojis", "static", "assets", "docs",
        "papers", "Formal Papers", "knowledge", "legacy")
ROOTS = [r"C:\02_QUILLAN", r"C:\03_WORKSPACE", r"C:\04_CONFIG",
         r"C:\Users\Admin\Downloads", r"C:"]


def walk(root, maxdepth):
    out_dirs, out_big, out_qbin = [], [], []
    for dp, dns, fns in os.walk(root, topdown=True):
        depth = dp.count(os.sep) - root.count(os.sep)
        dns[:] = [d for d in dns if d not in SKIP and not d.startswith(".")]
        if depth > maxdepth:
            dns[:] = []
            continue
        for d in dns:
            if any(k in d.lower() for k in DKEYS):
                out_dirs.append(dp + os.sep + d)
        for f in fns:
            fl = f.lower()
            p = os.path.join(dp, f)
            try:
                if fl.endswith(".qbin"):
                    out_qbin.append((os.path.getsize(p) // 1024, p))
                elif fl.endswith((".bin", ".gguf", ".safetensors", ".pt",
                                   ".onnx", ".engine", ".plan")):
                    sz = os.path.getsize(p) // 1024 // 1024
                    if sz >= 100:
                        out_big.append((sz, p))
            except OSError:
                pass
    return out_dirs, out_big, out_qbin


for root in ROOTS:
    if not Path(root).exists():
        continue
    dd, big, qb = walk(root, 4 if root != "C:" else 2)
    print(f"== {root} ==", flush=True)
    for d in dd:
        print(f"DIR {d}", flush=True)
    for sz, p in sorted(big, reverse=True)[: 20]:
        print(f"BIG {sz}MB {p}", flush=True)
    for kb, p in qb:
        print(f"QBIN {kb}KB {p}", flush=True)
print("sweep done", flush=True)
