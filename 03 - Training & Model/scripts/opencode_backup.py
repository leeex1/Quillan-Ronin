"""Safe online backup + audit for opencode sessions DB."""
from __future__ import annotations
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ALLOWED_ROOTS: tuple[Path, ...] = (
    Path(r"C:\05_ARCHIVE\Backups"),
    Path(r"C:\Users\Admin\.local\share\opencode"),
    Path(r"C:\02_QUILLAN"),
)

def _canonical(p: str | Path) -> Path:
    pp = Path(p).resolve()
    # reject symlinks for backup target to avoid aliasing
    if pp.is_symlink():
        raise ValueError(f"symlink not allowed: {pp}")
    return pp

def _ensure_allowed(p: Path, roots: Sequence[Path]) -> Path:
    cp = _canonical(p)
    for r in roots:
        try:
            cp.relative_to(r.resolve())
            return cp
        except ValueError:
            continue
    # allow exact root itself
    for r in roots:
        if cp == r.resolve():
            return cp
    raise ValueError(f"path escapes allowed roots: {cp}")

def make_logger(trace_id: str = "backup-001") -> logging.Logger:
    lg = logging.getLogger("opencode.backup")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('{"level":"%(levelname)s","msg":"%(message)s","trace_id":"'+trace_id+'","component":"opencode.backup"}'))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg

@dataclass(frozen=True)
class AuditRow:
    directory: str
    count: int
    last_updated: int

def audit(db_path: Path, logger: logging.Logger, trace_id: str) -> list[AuditRow]:
    # Time O(n) over session table index on directory; Space O(d) distinct dirs.
    src = _ensure_allowed(db_path, ALLOWED_ROOTS)
    uri = f"file:{src}?mode=ro"
    rows: list[AuditRow] = []
    with sqlite3.connect(uri, uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute("SELECT directory, COUNT(*) c, MAX(time_updated) last_up FROM session GROUP BY directory ORDER BY last_up DESC")
        for r in cur.fetchall():
            rows.append(AuditRow(str(r["directory"]), int(r["c"]), int(r["last_up"])))
    logger.info(f"audit ok dirs={len(rows)}", extra={})
    return rows

def online_backup(src_db: Path, dst_db: Path, logger: logging.Logger) -> Path:
    # Uses SQLite online backup API: safe against live WAL. Time O(pages), Space O(db size) ~2GB.
    src = _ensure_allowed(src_db, ALLOWED_ROOTS)
    dst = _ensure_allowed(dst_db, ALLOWED_ROOTS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(f"refusing overwrite, use O_EXCL semantics: {dst}")
    # CREATE_NEW equivalent: open with 'x' to fail if exists (race-safe)
    with open(dst, "xb") as _:
        pass
    dst.unlink()  # remove placeholder, let sqlite create it
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=30) as s:
        with sqlite3.connect(str(dst), timeout=30) as d:
            s.backup(d, pages=100)  # bounded pages per step avoids long lock
    logger.info(f"backup ok src={src.name} dst={dst.name} size={dst.stat().st_size}")
    return dst

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Online backup + audit opencode.db")
    ap.add_argument("--src", default=r"C:\Users\Admin\.local\share\opencode\opencode.db")
    ap.add_argument("--dst", default="")
    ap.add_argument("--trace-id", default="backup-001")
    a = ap.parse_args()
    lg = make_logger(a.trace_id)
    try:
        src = Path(a.src)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dst = Path(a.dst) if a.dst else Path(rf"C:\05_ARCHIVE\Backups\opencode_{stamp}.db")
        out = online_backup(src, dst, lg)
        for r in audit(out, lg, a.trace_id):
            lg.info(f"dir={r.directory} count={r.count} last={r.last_updated}")
        return 0
    except Exception as e:
        lg.error(f"backup failed err={type(e).__name__}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
