#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quillan relay watchdog — Muse watcher spec, mirrored.
curl via subprocess (never urllib for polling), token via stdin (-H @-),
last_seen_id tracking, heartbeat freshness, stale-20min alert + relaunch.
Cycle 900s. Light forever-loop; run detached.
Deviation logged: token file read (owner's disk setup) instead of pure stdin —
stdin is used for the curl handoff so the token never touches argv/env."""
from __future__ import annotations
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"C:\02_QUILLAN")
DD = REPO / "deploy" / "discord"
CH = "1438282228898467951"
ME = "1549554201649090630"
CYCLE, STALE = 900, 1200

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def token() -> str:
    md = (REPO / "09 - Projects" / "projects" / "opencode quillan.md"
          ).read_text(encoding="utf-8")
    return re.search(r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}",
                     md).group(0)


def curl_api(path: str, method="GET", payload=None):
    tok = token()
    hdr = f"Authorization: Bot {tok}\r\nUser-Agent: quillan-watchdog\r\n"
    if payload is not None:
        hdr += "Content-Type: application/json\r\n"
    cmd = ["curl", "-s", "--max-time", "20", "-X", method, "-H", "@-",
           f"https://discord.com/api/v10{path}"]
    if payload is not None:
        cmd += ["--data-binary", "@-"]
        stdin = (hdr + "\r\n" + json.dumps(payload)).encode()
    else:
        stdin = hdr.encode()
    out = subprocess.run(cmd, input=stdin, capture_output=True, timeout=30)
    return json.loads(out.stdout.decode("utf-8") or "null")


def post(text: str):
    curl_api(f"/channels/{CH}/messages", "POST", {"content": text})


def main() -> int:
    print("watchdog online", flush=True)
    while True:
        try:
            msgs = curl_api(f"/channels/{CH}/messages?limit=10") or []
            if msgs:
                (DD / ".last_seen_id").write_text(str(msgs[0]["id"]))
            (DD / ".watchdog_heartbeat").write_text(str(time.time()))
            hb = DD / ".relay_heartbeat"
            age = time.time() - float(hb.read_text().strip()) if hb.exists() else 1e9
            if age > STALE:
                post("[SYS] watchdog: relay heartbeat stale "
                     f"({age / 60:.0f}min) — relaunching relay.")
                subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command",
                     "Start-Process -FilePath 'python' "
                     "-ArgumentList '\"C:\\02_QUILLAN\\scripts\\quillan_discord_relay.py\"' "
                     "-WorkingDirectory 'C:\\02_QUILLAN' -WindowStyle Hidden"],
                    cwd=str(REPO))
                print(f"STALE relaunch issued (age {age:.0f}s)", flush=True)
            else:
                print(f"ok age={age:.0f}s msgs={len(msgs)}", flush=True)
        except Exception as e:  # never die; report without secrets
            print(f"watchdog err: {type(e).__name__}", flush=True)
        time.sleep(CYCLE)


if __name__ == "__main__":
    raise SystemExit(main())
