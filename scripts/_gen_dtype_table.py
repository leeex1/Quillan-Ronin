#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

full = json.load(
    open(r"C:\02_QUILLAN\scripts\_socket_header_full.json", encoding="utf-8")
)
lines = ["EXPECTED_DTYPE_CODES: Dict[str, str] = {"]
for k in sorted(full):
    if k.endswith(".bitnet_scale"):
        continue
    dtype = full[k]["dtype"]
    lines.append("    {!r}: {!r},".format(k, dtype))
lines.append("}")
Path(r"C:\02_QUILLAN\scripts\_dtype_table.txt").write_text(
    "\n".join(lines), encoding="utf-8"
)
print("dtype lines", len(lines))
print(Counter(full[k]["dtype"] for k in full if not k.endswith(".bitnet_scale")))

p = Path(r"C:\02_QUILLAN\03 - Training & Model\quillan_fused_forward.py")
s = p.read_text(encoding="utf-8")
start = s.index("EXPECTED_DTYPE_CODES: Dict[str, str] = {")
end = s.index("\n}\n", start) + 3
new = Path(r"C:\02_QUILLAN\scripts\_dtype_table.txt").read_text(encoding="utf-8") + "\n"
s = s[:start] + new + s[end:]
p.write_text(s, encoding="utf-8")
print("dtype table injected into module")
