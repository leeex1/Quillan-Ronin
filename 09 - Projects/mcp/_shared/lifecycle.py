"""Parent-death reaper for Quillan MCP servers.

Clients (opencode, the Antigravity language server) spawn servers through
cmd/npx wrappers. Windows has no process-group semantics, so when the client
dies the wrapper dies and the real server (py/node grandchild) is orphaned --
which is why every client restart left a permanent new generation behind.

This installs one daemon thread that polls the original parent PID and calls
os._exit() once the parent is gone. The recorded PID is the *wrapper*, so the
grandchild case is covered too: wrapper dies -> PID stops resolving -> exit.

It never reads stdin: the MCP transport owns that stream and a second reader
would swallow JSON-RPC frames.

Opt out with QUILLAN_NO_PARENT_WATCH=1.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

__all__ = ["install", "is_installed"]

_WAIT_TIMEOUT = 0x00000102
_SYNCHRONIZE = 0x00100000

_INSTALLED = False
_LOCK = threading.Lock()


def _parent_alive(ppid: int) -> bool:
    """True while the process lives. Windows-safe (os.kill(pid, 0) is not)."""
    if ppid <= 0:
        return True

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(_SYNCHRONIZE, False, ppid)
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True  # never kill a server because the probe broke

    try:
        os.kill(ppid, 0)
        return True
    except OSError:
        return False


def _daemonize_threads() -> None:
    """Stop stray non-daemon workers holding the process open at shutdown."""
    for thread in threading.enumerate():
        if thread is threading.current_thread():
            continue
        try:
            thread.daemon = True
        except Exception:
            pass


def is_installed() -> bool:
    return _INSTALLED


def install(
    server_name: str = "MCP", interval: float = 3.0, exit_code: int = 0
) -> bool:
    """Start the parent watchdog. Idempotent; True if this call installed it.

    Call as early as possible -- the parent must still be resolvable when the
    original PID is captured.
    """
    global _INSTALLED

    if os.environ.get("QUILLAN_NO_PARENT_WATCH") == "1":
        return False

    with _LOCK:
        if _INSTALLED:
            return False
        _INSTALLED = True

    parent_pid = os.getppid()

    def _watch() -> None:
        while True:
            time.sleep(interval)
            if not _parent_alive(parent_pid):
                try:
                    sys.stderr.write(
                        f"[{server_name}] parent {parent_pid} exited; "
                        f"reaping {os.getpid()}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                os._exit(exit_code)

    thread = threading.Thread(
        target=_watch,
        name=f"{server_name}-parent-watch",
        daemon=True,
    )
    thread.start()

    atexit.register(_daemonize_threads)
    return True
