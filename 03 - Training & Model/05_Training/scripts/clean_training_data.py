"""
Quillan-Ronin Training Data Cleaner v2.0
==========================================
INTENTIONALLY KEEPS:
  - media_metadata source entirely (trains the MoE expert router)
  - tool_use_excellence, coding_excellence (high-quality signal)
  - Quillan system prompts, templates, persona files
  - Python/JS/code files with actual logic
  - Downloads, misc sources

REMOVES only genuine noise leaked from full_pc_crawl / quillan_ronin:
  - npm/yarn lockfile JSON entries (sha512 integrity + requiresBuild)
  - Raw timestamped log lines (INFO/DEBUG/ERROR spam)
  - Pure copyright/license header-only files (MS boilerplate)
  - TextMate/VS Code grammar token pattern files

Source-aware logic:
  - media_metadata  → ALWAYS KEEP (intentional router training signal)
  - tool/coding     → ALWAYS KEEP
  - quillan_ronin   → filter out npm locks + log spam, keep rest
  - full_pc_crawl   → filter out copyright-only + grammar files, keep code
  - unknown         → apply noise filter

Output: domain_general_routing_clean.jsonl
"""
import json, re
from pathlib import Path

SRC = Path("C:/Users/Admin/Quillan-Ronin/training_data/domain_splits/domain_general_routing.jsonl")
DST = Path("C:/Users/Admin/Quillan-Ronin/training_data/domain_splits/domain_general_routing_clean.jsonl")

# ── Sources to always keep in full ──────────────────────────────────────────
ALWAYS_KEEP_SOURCES = {
    "media_metadata",           # intentional MoE router training
    "tool_use_excellence",
    "coding_excellence",
    "quillan_knowledge",
    "downloads",
}

# ── Noise patterns — only applied to non-protected sources ──────────────────
NOISE_RE = [
    # npm lockfile: has both requiresBuild AND sha512 integrity hash
    re.compile(r'"requiresBuild".*"integrity"\s*:\s*"sha512-', re.DOTALL),
    # Bare sha512 hash lines (no useful text around them)
    re.compile(r'^"[A-Za-z0-9_\-]+"\s*:\s*\{\s*"checkedAt"\s*:\s*\d+.*"integrity"\s*:\s*"sha512-'),
    # Timestamped log spam: 2026-05-14 12:34:56,789 [INFO] Something
    re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\s+\[(?:INFO|DEBUG|WARNING|ERROR|CRITICAL)\]'),
    # Pure MS copyright header files (no real code below)
    re.compile(r'Copyright \(c\) Microsoft Corporation\. All rights reserved\.\s*$', re.MULTILINE),
    # TextMate/VS Code grammar scope names only
    re.compile(r'"name"\s*:\s*"(?:comment|punctuation|keyword|entity|storage|string|variable|support|meta)\.[a-z.]+\.[a-z.]+"'),
]

MIN_LEN = 30  # anything shorter is noise


def is_noise(text: str, source: str) -> bool:
    """Return True if this entry should be removed."""
    if source in ALWAYS_KEEP_SOURCES:
        return False  # never filter protected sources

    if len(text.strip()) < MIN_LEN:
        return True

    # Apply each noise pattern — only needs one match to flag
    for pat in NOISE_RE:
        if pat.search(text[:500]):
            return True

    return False


def main():
    total = kept = removed = 0
    removed_by_source: dict[str, int] = {}
    kept_by_source: dict[str, int] = {}

    with open(SRC, "r", encoding="utf-8", errors="replace") as fin, \
         open(DST, "w", encoding="utf-8") as fout:

        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total += 1

            try:
                d = json.loads(raw_line)
            except json.JSONDecodeError:
                removed += 1
                continue

            text   = d.get("text", "")
            source = d.get("meta", {}).get("source", "unknown")

            if is_noise(text, source):
                removed += 1
                removed_by_source[source] = removed_by_source.get(source, 0) + 1
            else:
                kept += 1
                kept_by_source[source] = kept_by_source.get(source, 0) + 1
                fout.write(raw_line + "\n")

            if total % 20000 == 0:
                print(f"  {total:,} processed — kept {kept:,}, removed {removed:,}", flush=True)

    print(f"\n=== Clean Complete ===")
    print(f"Total:   {total:,}")
    print(f"Kept:    {kept:,} ({100*kept/max(total,1):.1f}%)")
    print(f"Removed: {removed:,} ({100*removed/max(total,1):.1f}%)")

    print(f"\nKept by source:")
    for src, n in sorted(kept_by_source.items(), key=lambda x: -x[1]):
        print(f"  {n:>7,}  {src}")

    print(f"\nRemoved by source:")
    for src, n in sorted(removed_by_source.items(), key=lambda x: -x[1]):
        print(f"  {n:>7,}  {src}")

    size_mb = DST.stat().st_size / 1e6
    print(f"\nOutput: {DST.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
