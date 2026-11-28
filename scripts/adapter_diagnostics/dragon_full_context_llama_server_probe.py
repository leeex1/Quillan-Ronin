#!/usr/bin/env python3
"""Bounded standalone llama-server probe for the delivered Dragon GGUF.

Loads the exact clean Falcon3-1B-Instruct blob used by Modelfile.dragon_standard,
uses the exact canonical SYSTEM block from that Modelfile, and requests one short
identity answer. The server is started and terminated in-process: no orphan.
"""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SRV = Path(r"C:\Users\Admin\.docker\bin\inference\llama-server.exe")
BLOB = Path(
    r"C:\Users\Admin\.ollama\models\blobs"
    r"\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"
)
MODELFILE = Path(r"F:\02_QUILLAN\exports\Modelfile.dragon_standard")
OUT_LOG = Path(r"F:\02_QUILLAN\exports\diagnostics\dragon_full_context_server.log")
HOST = "127.0.0.1"
PORT = 8099
CTX = 262144
PROMPT = "State your exact name, creator, and version. One sentence."


def extract_system(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = 'SYSTEM """'
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("SYSTEM block not found")
    start += len(marker)
    end = text.find('\n"""', start)
    if end < 0:
        raise RuntimeError("SYSTEM terminator not found")
    system = text[start:end].rstrip("\r\n")
    return system


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main() -> int:
    system = extract_system(MODELFILE)
    print(f"system_chars={len(system)}", flush=True)
    print(f"blob={BLOB}", flush=True)
    print(f"ctx={CTX} cache_k=q4_0 cache_v=q4_0", flush=True)
    if not SRV.exists():
        print(f"server_missing={SRV}", flush=True)
        return 2
    if not BLOB.exists():
        print(f"blob_missing={BLOB}", flush=True)
        return 2

    cmd = [
        str(SRV),
        "-m",
        str(BLOB),
        "-c",
        str(CTX),
        "-ctk",
        "q4_0",
        "-ctv",
        "q4_0",
        "-np",
        "1",
        "-t",
        "8",
        "-tb",
        "8",
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    print("launching_standalone_server", flush=True)
    with OUT_LOG.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.time() + 180
            healthy = False
            while time.time() < deadline and proc.poll() is None:
                try:
                    with urllib.request.urlopen(
                        f"http://{HOST}:{PORT}/health", timeout=3
                    ) as r:
                        if r.status == 200:
                            healthy = True
                            break
                except Exception:
                    time.sleep(2)
            print(f"server_healthy={healthy}", flush=True)
            if not healthy:
                tail = OUT_LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
                print("server_log_tail_begin", flush=True)
                print(tail, flush=True)
                print("server_log_tail_end", flush=True)
                return 3

            payload = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": PROMPT},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 64,
                "stream": False,
            }
            req = urllib.request.Request(
                f"http://{HOST}:{PORT}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=900) as r:
                obj = json.loads(r.read().decode("utf-8"))
            usage = obj.get("usage", {})
            text = obj["choices"][0]["message"]["content"]
            print(f"prompt_tokens={usage.get('prompt_tokens')}", flush=True)
            print(f"completion_tokens={usage.get('completion_tokens')}", flush=True)
            print(f"response={text!r}", flush=True)
            valid = int(usage.get("prompt_tokens") or 0) >= 190000
            print(
                f"verdict={'DRAGON_FULL_CONTEXT_PROMPT_OK' if valid else 'DRAGON_CONTEXT_TRUNCATED'}",
                flush=True,
            )
            return 0 if valid else 4
        except Exception as e:
            print(f"probe_error={type(e).__name__}: {e}", flush=True)
            return 5
        finally:
            stop(proc)
            print(f"server_stopped={proc.poll() is not None}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
