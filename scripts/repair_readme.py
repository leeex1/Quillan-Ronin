#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full README repair:
1. Restore from git HEAD (bypasses PowerShell encoding mangling)
2. Reverse the cp1252 mojibake (encode cp1252 -> decode utf-8)
3. Fence-aware reflow: restore line structure for GitHub/HF rendering
4. Verify content integrity
"""
import subprocess
import sys
from pathlib import Path

REPO = r"C:\02_QUILLAN"
p = Path(REPO) / "README.md"

# 1. Restore from git HEAD as raw bytes
res = subprocess.run(["git", "-C", REPO, "show", "HEAD:README.md"],
                     capture_output=True)
s = res.stdout.decode("utf-8", errors="replace")
print(f"[1] restored from HEAD: {len(s)} chars")

# 2. Reverse mojibake (cp1252 round-trip). ASCII unaffected; genuine
#    non-cp1252 chars are kept as-is via the per-char fallback.
fixed_chars = 0
if "\ufffd" in s or "\u2261" in s or "\u2018" in s or True:  # attempt always; verify after
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        try:
            b = ch.encode("cp1252")
            # collect a maximal run of cp1252-encodable chars
            j = i + 1
            while j < n:
                try:
                    b += s[j].encode("cp1252")
                    j += 1
                except UnicodeEncodeError:
                    break
            try:
                decoded = b.decode("utf-8")
                out.append(decoded)
                if decoded != s[i:j]:
                    fixed_chars += 1
                i = j
                continue
            except UnicodeDecodeError:
                pass
        except UnicodeEncodeError:
            pass
        out.append(ch)
        i += 1
    s2 = "".join(out)
else:
    s2 = s

# sanity: mojibake markers gone?
markers_before = s.count("\u2261\u0192\u00c3") + s.count("\ufffd")
print(f"[2] mojibake fix: {fixed_chars} runs rewritten; residual bad markers: {s2.count(chr(0xfffd))}")
s = s2

# 3. Fence-aware reflow
def reflow(text: str) -> str:
    parts = []
    in_fence = False
    for seg in text.split("```"):
        if not in_fence:
            # headings
            for h in ("## ", "# ", "### ", "#### "):
                seg = seg.replace("\n" + h, "\n\n" + h)
                seg = seg.replace(h, "\n\n" + h, 1) if seg.startswith(h) else seg
            seg = seg.replace("## ", "\n\n## ").replace("# ", "\n\n# ", 1) if seg.startswith("#") else seg
            # generic: newline before any heading token not already at line start
            import re as _re
            seg = _re.sub(r"(?<!\n)(#{1,4} )", r"\n\n\1", seg)
            # horizontal rules
            seg = _re.sub(r"(?<!\n)---", "\n\n---", seg)
            seg = seg.replace("---\n\n\n", "---\n\n")
            # html block tags
            seg = seg.replace("</div>", "</div>\n")
            seg = seg.replace("<div ", "\n<div ")
            seg = seg.replace("</p>", "</p>\n\n")
            seg = seg.replace("<h1 ", "\n<h1 ")
            seg = seg.replace("<table>", "\n<table>")
            seg = seg.replace("</table>", "</table>\n")
            seg = seg.replace("<br/>", "<br/>\n")
            # markdown table rows: '| --- |' style separators and pipe-rows
            seg = _re.sub(r"(?<!\n)(\|[:\- ]+\|)", r"\n\1", seg)
            seg = _re.sub(r"(?<!\n)(\| `)", r"\n| `", seg)
            seg = _re.sub(r"(?<!\n)(\|\s*#)", r"\n| #", seg)
            seg = _re.sub(r"(?<!\n)(\|\s*\d+)", r"\n| \1", seg)
            seg = _re.sub(r"(?<!\n)(\|\s*[A-Z][a-z])", r"\n| \1", seg)
            seg = _re.sub(r"(?<!\n)(\*\*Quillan)", r"\n\1", seg)
            seg = _re.sub(r"(?<!\n)(---\s*\*?\*?\*?)", r"\n\1", seg)
            seg = _re.sub(r"\n{3,}", "\n\n", seg)
        parts.append(seg)
        in_fence = not in_fence
    return "```".join(parts)

s = reflow(s)
lines = s.count("\n") + 1
print(f"[3] reflow done: now {lines} lines, {len(s)} chars")

# 4. Integrity checks
for marker in ("Success Stories", "Comprehensive Benchmark Table", "Model Specs",
               "ONI", "LINEAGE", "Getting Help", "Version History"):
    print(f"    contains {marker!r}: {marker in s}")

p.write_text(s, encoding="utf-8", newline="\n")
print("[4] written:", p)
bad = 0
