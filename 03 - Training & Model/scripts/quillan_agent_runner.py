#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QUILLAN COUNCIL AGENT RUNNER - makes the 35 .agent.md definitions real agents.
Loads persona prompt -> queries live Oni via gateway -> returns specialist contribution."""
import json
import re
import urllib.request
from pathlib import Path

REPO = Path(r"C:\02_QUILLAN")
AGENTS_DIR = REPO / ".github" / "agents"
GATEWAY_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "quillan-oni-mini-6l"
# Substrate switch: "oni" (our weights via gateway, default + ship target)
# or "external" (any OpenAI-compatible endpoint for dev help; set env QUILLAN_DEV_API).
import os as _os
SUBSTRATE = _os.environ.get("QUILLAN_SUBSTRATE", "oni")
DEV_API_URL = _os.environ.get("QUILLAN_DEV_API", "")
DEV_API_KEY = _os.environ.get("QUILLAN_DEV_KEY", "")
DEV_MODEL = _os.environ.get("QUILLAN_DEV_MODEL", "")

# blackboard C0-C33 (0-indexed) -> agent.md C1-C34 (1-indexed)
BB_TO_FILE = {}
for i in range(34):
    BB_TO_FILE[f"C{i}"] = None
_NAMES = ["astra", "vir", "solace", "praxis", "echo", "omnis", "logos", "metasynth",
          "aether", "codeweaver", "harmonia", "sophiae", "warden", "kaido", "luminaris",
          "voxum", "nullion", "shepherd", "vigil", "artifex", "archon", "aurelion",
          "cadence", "schema", "prometheus", "techne", "chronicle", "calculus",
          "navigator", "tesseract", "nexus", "aeon", "typist", "predator"]
for i, name in enumerate(_NAMES):
    BB_TO_FILE[f"C{i}"] = f"c{i + 1}-{name}.agent.md"


def load_persona(bb_id: str) -> str:
    """Load the persona prompt for a blackboard council ID (e.g. 'C6')."""
    fname = BB_TO_FILE.get(bb_id)
    if not fname:
        return f"You are {bb_id}, a Quillan-Ronin council specialist."
    text = (AGENTS_DIR / fname).read_text(encoding="utf-8")
    body = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    return body.strip()


def deliberate_as(bb_id: str, specialty: str, query: str, max_tokens: int = 120) -> str:
    """Run one council agent: persona + query -> Oni -> contribution."""
    persona = load_persona(bb_id)
    prompt = (f"{persona}\n\nYour specialty: {specialty}.\n"
              f"Council query: {query}\n"
              f"Give your specialist assessment in 2-3 tight sentences. No preamble.")
    if SUBSTRATE == "external" and DEV_API_URL and DEV_MODEL:
        url, model, key = DEV_API_URL, DEV_MODEL, DEV_API_KEY
    else:
        url, model, key = GATEWAY_URL, MODEL, ""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read().decode("utf-8"))
        return d["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[{bb_id} unreachable: {e}]"


if __name__ == "__main__":
    print(deliberate_as("C6", "Logical Consistency", "Is the sky blue?"))
