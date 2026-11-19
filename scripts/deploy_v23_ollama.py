#!/usr/bin/env python3
"""Headless Ollama Serve + 2M Context Modelfile Registration & Verification Pipeline.

Ensures:
1. User-level and process-level environment variables:
   - OLLAMA_KV_CACHE_TYPE=q4_0
   - OLLAMA_FLASH_ATTENTION=1
2. Headless `ollama serve` is online at http://127.0.0.1:11434 without GUI errors.
3. Re-runs `build_full_sovereign_modelfiles.py` so `FROM` binds to the newest
   promoted `*-wsp-v23*-q8_0.gguf` (or verified fallback if v23 is still building).
4. Registers local Ollama tags (`quillan-ronin-1b:latest`, `quillan-mini:latest`,
   `quillan-dragon:latest`) and remote Hub tags (`crashoverridex/Quillan-Ronin`,
   `crashoverridex/Quillan-Ronin-mini`, `crashoverridex/Quillan-Dragon`).
5. Verifies `/api/show` metadata (`num_ctx 2097152`, un-truncated system prompt).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(r"C:\02_QUILLAN")
OLLAMA_HOST = "http://127.0.0.1:11434"

MODELS = [
    {
        "key": "flagship",
        "modelfile": "Modelfile",
        "local_tag": "quillan-ronin-1b:latest",
        "hub_tag": "crashoverridex/Quillan-Ronin",
    },
    {
        "key": "mini",
        "modelfile": "Modelfile.mini",
        "local_tag": "quillan-mini:latest",
        "hub_tag": "crashoverridex/Quillan-Ronin-mini",
    },
    {
        "key": "dragon",
        "modelfile": "Modelfile.dragon",
        "local_tag": "quillan-dragon:latest",
        "hub_tag": "crashoverridex/Quillan-Dragon",
    },
]


def ollama_env() -> dict[str, str]:
    env = os.environ.copy()
    env["OLLAMA_KV_CACHE_TYPE"] = "q4_0"
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    return env


def ping_ollama(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_headless_ollama_serve() -> None:
    if ping_ollama():
        print("[ollama] Headless server already online at 127.0.0.1:11434")
        return

    print("[ollama] Starting headless `ollama serve` (OLLAMA_KV_CACHE_TYPE=q4_0, OLLAMA_FLASH_ATTENTION=1)...")
    creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    log_path = ROOT_DIR / "exports" / "ollama_headless_serve.log"
    log_file = open(log_path, "ab")
    subprocess.Popen(
        ["ollama", "serve"],
        cwd=str(ROOT_DIR),
        env=ollama_env(),
        stdout=log_file,
        stderr=log_file,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    for _ in range(25):
        time.sleep(0.5)
        if ping_ollama():
            print("[ollama] Headless `ollama serve` is UP at 127.0.0.1:11434")
            return
    raise RuntimeError("Failed to start headless `ollama serve` on 127.0.0.1:11434")


def verify_tag_show(tag: str) -> dict:
    payload = json.dumps({"name": tag}).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    params = data.get("parameters", "")
    system = data.get("system", "")
    has_2m = "2097152" in params
    return {
        "tag": tag,
        "has_2m_num_ctx": has_2m,
        "system_chars": len(system),
        "family": data.get("details", {}).get("family", ""),
        "parameter_size": data.get("details", {}).get("parameter_size", ""),
        "quantization_level": data.get("details", {}).get("quantization_level", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and run headless Ollama v23 deployment.")
    parser.add_argument("--serve-only", action="store_true", help="Ensure headless ollama serve + rebuild Modelfiles only.")
    parser.add_argument("--model", choices=["all", "flagship", "mini", "dragon"], default="all")
    parser.add_argument("--push-hub", action="store_true", help="Also run `ollama push` on hub tags.")
    args = parser.parse_args()

    # 1. Rebuild Modelfiles so FROM binds to latest v23 (or current fallback)
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "build_full_sovereign_modelfiles.py")],
        cwd=str(ROOT_DIR),
        check=True,
    )

    # 2. Ensure headless ollama serve is running with q4_0 KV cache & FlashAttention
    ensure_headless_ollama_serve()

    if args.serve_only:
        for m in MODELS:
            mf = ROOT_DIR / m["modelfile"]
            from_line = next(
                (ln for ln in mf.read_text(encoding="utf-8").splitlines() if ln.startswith("FROM ")),
                "",
            )
            print(f"[ready] {m['key']}: {m['modelfile']} -> {from_line} | tags={m['local_tag']}, {m['hub_tag']}")
        return

    env = ollama_env()
    for m in MODELS:
        if args.model != "all" and m["key"] != args.model:
            continue
        mf_path = ROOT_DIR / m["modelfile"]
        local_tag = m["local_tag"]
        hub_tag = m["hub_tag"]
        print(f"=== Building {local_tag} from {mf_path.name} ===")
        subprocess.run(
            ["ollama", "create", local_tag, "-f", str(mf_path)],
            cwd=str(ROOT_DIR),
            env=env,
            check=True,
        )
        print(f"=== Copying {local_tag} -> {hub_tag} ===")
        subprocess.run(
            ["ollama", "cp", local_tag, hub_tag],
            cwd=str(ROOT_DIR),
            env=env,
            check=True,
        )
        info = verify_tag_show(local_tag)
        print(f"[verified] {json.dumps(info)}")
        if not info["has_2m_num_ctx"] or info["system_chars"] < 600000:
            raise RuntimeError(f"Verification failed for {local_tag}: {info}")

        if args.push_hub:
            print(f"=== Pushing {hub_tag} to Ollama Hub ===")
            subprocess.run(
                ["ollama", "push", hub_tag],
                cwd=str(ROOT_DIR),
                env=env,
                check=True,
            )


if __name__ == "__main__":
    main()
