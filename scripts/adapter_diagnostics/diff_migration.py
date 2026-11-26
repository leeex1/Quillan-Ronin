#!/usr/bin/env python3
"""List files present on C: but missing on F: (post-copy creations expected)."""

from pathlib import Path

SRC = Path(r"C:\02_QUILLAN")
DST = Path(r"F:\02_QUILLAN")


def rel_files(root):
    s = set()
    for p in root.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                s.add(str(p.relative_to(root)))
        except (OSError, PermissionError):
            pass
    return s


a = rel_files(SRC)
b = rel_files(DST)
missing = sorted(a - b)
extra = sorted(b - a)
print(
    f"source={len(a)} dest={len(b)} missing_on_F={len(missing)} extra_on_F={len(extra)}"
)
print("\n=== missing on F: (present on C:) ===")
for f in missing[:60]:
    p = SRC / f
    try:
        sz = p.stat().st_size
    except:
        sz = -1
    print(f"  {sz:>12}  {f}")
print("\n=== extra on F: ===")
for f in extra[:30]:
    print("  " + f)
