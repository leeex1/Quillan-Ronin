"""Move opencode sessions between directories safely (transactional UPDATE)."""
from __future__ import annotations
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ALLOWED_DB = Path(r"C:\Users\Admin\.local\share\opencode\opencode.db").resolve()
ALLOWED_DIRS = ("C:/Users/Admin", "C:/02_QUILLAN", "C:/Windows/System32", "C:/Windows")

def make_logger(trace_id: str) -> logging.Logger:
    lg = logging.getLogger("opencode.migrate")
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('{"level":"%(levelname)s","msg":"%(message)s","trace_id":"'+trace_id+'","component":"opencode.migrate"}'))
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    return lg

def _norm_dir(d: str) -> str:
    if not isinstance(d, str) or not d:
        raise ValueError("empty directory")
    if ".." in d or "~" in d:
        raise ValueError(f"unsafe directory: {d}")
    nd = d.replace("\\", "/").rstrip("/")
    if nd not in ALLOWED_DIRS:
        raise ValueError(f"directory not allow-listed: {nd}")
    return nd

@dataclass(frozen=True)
class Plan:
    from_dir: str
    to_dir: str
    ids: tuple[str, ...]

def ensure_project_dir(con: sqlite3.Connection, project_id: str, directory: str, now_ms: int) -> bool:
    cur = con.execute("SELECT 1 FROM project_directory WHERE project_id=? AND directory=?", (project_id, directory))
    if cur.fetchone():
        return False
    con.execute("INSERT INTO project_directory(project_id,directory,time_created) VALUES(?,?,?)", (project_id, directory, now_ms))
    return True

def migrate(db_path: str, from_dir: str, to_dir: str, limit: int, apply: bool, logger: logging.Logger, trace_id: str) -> dict:
    """Move sessions preserving original time_updated order. Time O(k), Space O(k)."""
    if Path(db_path).resolve() != ALLOWED_DB:
        raise ValueError("db path not allowed (must be canonical opencode.db)")
    src = _norm_dir(from_dir)
    dst = _norm_dir(to_dir)
    if src == dst:
        raise ValueError("from_dir == to_dir")
    if limit <= 0 or limit > 5000:
        raise ValueError("limit must be 1..5000")
    with sqlite3.connect(str(ALLOWED_DB), timeout=30, isolation_level=None) as con:
        con.execute("PRAGMA busy_timeout=30000")
        total_from = con.execute("SELECT COUNT(*) FROM session WHERE directory=?", (src,)).fetchone()[0]
        total_to_before = con.execute("SELECT COUNT(*) FROM session WHERE directory=?", (dst,)).fetchone()[0]
        rows = con.execute("SELECT id,title,time_updated FROM session WHERE directory=? ORDER BY time_updated DESC LIMIT ?", (src, limit)).fetchall()
        logger.info(f"plan from={src} total={total_from} pilot={len(rows)} to_before={total_to_before}")
        if not apply:
            return {"dry_run": True, "from_total": total_from, "to_before": total_to_before, "pilot": len(rows), "sample": [r[0] for r in rows[:3]]}
        if not rows:
            return {"dry_run": False, "moved": 0}
        ids = [r[0] for r in rows]
        con.execute("BEGIN IMMEDIATE")
        try:
            proj = con.execute("SELECT project_id FROM session WHERE id=?", (ids[0],)).fetchone()
            pid = proj[0] if proj else "global"
            now_ms = int(time.time() * 1000)
            created = ensure_project_dir(con, pid, dst, now_ms)
            qmarks = ",".join("?" for _ in ids)
            # Preserve time_updated to keep history order (only directory changes)
            cur2 = con.execute(f"UPDATE session SET directory=? WHERE id IN ({qmarks})", (dst, *ids))
            moved = cur2.rowcount
            to_after = con.execute("SELECT COUNT(*) FROM session WHERE directory=?", (dst,)).fetchone()[0]
            if to_after != total_to_before + moved:
                raise RuntimeError("count mismatch, rolling back")
            con.execute("COMMIT")
            logger.info(f"moved={moved} to_after={to_after} proj_link_created={created}")
            return {"dry_run": False, "moved": moved, "to_before": total_to_before, "to_after": to_after}
        except Exception:
            try:
                con.execute("ROLLBACK")
            except Exception:
                pass
            raise

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Move opencode sessions (dry-run default)")
    ap.add_argument("--from-dir", default="C:/Users/Admin")
    ap.add_argument("--to-dir", default="C:/02_QUILLAN")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--db", default=str(ALLOWED_DB))
    ap.add_argument("--trace-id", default="migrate-001")
    a = ap.parse_args()
    lg = make_logger(a.trace_id)
    try:
        res = migrate(a.db, a.from_dir, a.to_dir, a.limit, a.apply, lg, a.trace_id)
        lg.info(f"done res={res}")
        return 0
    except Exception as e:
        lg.error(f"migrate failed err={type(e).__name__}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
