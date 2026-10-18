#!/usr/bin/env python3
"""Build n8n credential import file from on-disk keys. Run once, then DELETE output."""
import json
import re
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8")

or_txt = open(r"C:\Users\Admin\AppData\Local\hermes\.env", encoding="utf-8",
              errors="ignore").read()
or_key = re.search(r"OPENROUTER_API_KEY\s*[:=]\s*['\"]?([^\s'\";]+)", or_txt).group(1)

nv_txt = open(r"C:\02_QUILLAN\.env", encoding="utf-8", errors="ignore").read()
nv_key = re.search(r"nvapi-[A-Za-z0-9\-_]{10,}", nv_txt).group(0)

creds = [
    {"id": str(uuid.uuid4()), "name": "OpenRouter", "type": "openRouterApi",
     "data": {"apiKey": or_key, "url": "https://openrouter.ai/api/v1"}},
    {"id": str(uuid.uuid4()), "name": "NVIDIA NIM", "type": "nvidiaApi",
     "data": {"apiKey": nv_key,
              "url": "https://integrate.api.nvidia.com/v1"}},
]
out = r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\creds-import.json"
open(out, "w", encoding="utf-8").write(json.dumps(creds))
print("wrote", out, "with", len(creds), "credentials (DELETE AFTER IMPORT)")
