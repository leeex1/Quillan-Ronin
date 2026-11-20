#!/usr/bin/env python3
"""Append a single message to the Antigravity swarm board JSONL ledger.

External (OpenCode) build sessions use this helper so every finding and edit is
recorded in the same format as the original Antigravity swarm messages.

Usage:
    python board_post.py --agent opencode-build --text "..." [--kind finding]
    python board_post.py --agent c10-codeweaver --file report.txt --kind finding
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

LOCK_TIMEOUT_SEC = 10.0
LOCK_RETRY_SEC = 0.05

BOARD: Path = Path(
    r"C:\Users\Admin\.gemini\antigravity\brain\290bbfdb-f3fe-47b2-a61b-c3fb5c016750"
    r"\.system_generated\swarm\board.jsonl"
)
LOCK: Path = BOARD.with_suffix(".jsonl.lock")


@contextmanager
def _exclusive_writer() -> Iterator[None]:
    """Serialize appends with a bounded non-blocking lock.

    Never blocks forever: if the lock is held past LOCK_TIMEOUT_SEC (stale lock
    from a crashed process), raise TimeoutError so the caller fails loudly
    instead of hanging the whole shell.
    """
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as handle:
        acquired = False
        try:
            if sys.platform == "win32":
                import msvcrt

                deadline = time.monotonic() + LOCK_TIMEOUT_SEC
                while True:
                    handle.seek(0)
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"board lock held >{LOCK_TIMEOUT_SEC}s by another writer; "
                                f"append aborted (no seq claimed)"
                            )
                        time.sleep(LOCK_RETRY_SEC)
            yield
        finally:
            if acquired:
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _next_seq() -> int:
    if not BOARD.exists():
        return 1
    last = 0
    with BOARD.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = max(last, int(json.loads(line).get("seq", 0)))
            except (ValueError, json.JSONDecodeError):
                continue
    return last + 1


def post(agent: str, text: str, kind: str | None = None, dry_run: bool = False) -> int:
    """Append one message and return the assigned sequence number."""
    with _exclusive_writer():
        seq = _next_seq()
        record: dict[str, object] = {
            "seq": seq,
            "agent": agent,
            "text": text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": str(uuid.uuid4()),
        }
        if kind:
            record["kind"] = kind
        if not dry_run:
            with BOARD.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return seq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", required=True, help="Ledger identity (e.g. opencode-build)"
    )
    parser.add_argument("--text", help="Message body")
    parser.add_argument("--file", type=Path, help="Read message body from a UTF-8 file")
    parser.add_argument("--kind", help="Optional kind tag, e.g. brief / finding / edit")
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute next seq without writing"
    )
    args = parser.parse_args()

    if args.text is None and args.file is None:
        parser.error("provide --text or --file")
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    seq = post(args.agent, text.strip(), args.kind, args.dry_run)
    print(f"{'DRY_RUN seq' if args.dry_run else 'posted seq'} {seq} as {args.agent}")


if __name__ == "__main__":
    main()
