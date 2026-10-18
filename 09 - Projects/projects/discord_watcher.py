#!/usr/bin/env python3
"""Discord channel watcher — wakes an agent when its bot is addressed.

How it works:
  - Polls a Discord channel every POLL_SECS seconds via the REST API.
  - Uses `curl` through subprocess. (Do NOT use Python urllib here — it silently
    fails behind some proxies and the watcher will miss messages with no error.)
  - The bot token is read from stdin (one line) and NEVER written to disk,
    never put in env, never on a command line.
  - Triggers when a message: mentions the bot, starts with the WAKE_PREFIX,
    or replies to one of the bot's own messages. The bot's own messages are ignored.
  - Keeps a message-ID watermark (last_seen_id) and touches a heartbeat file on
    every successful poll so an external watchdog can tell it apart from dead.
  - On a hit: prints the matched messages as JSON and EXITS 0. The supervising
    agent sees the exit, composes a reply, posts it, and restarts this script.
  - Self-recycles after MAX_RUNTIME_SECS (exit 0, no hits).
  - Exits 1 if polling fails repeatedly (e.g. dead/rotated token -> HTTP 401s).

State files (created in STATE_DIR, none contain credentials):
  last_seen_id   message ID watermark — only newer messages are considered
  heartbeat      unix timestamp of last successful poll
  poller.pid     single-instance guard
  poller_debug.log  startup/hit lines for diagnosing replays
"""

import sys, json, time, subprocess, os

# ---------------- CONFIG: fill these in ----------------
BOT_ID      = "1549557324346036244"   # your bot's user ID
CHANNEL_ID  = "1438282228898467951"   # channel to watch
WAKE_PREFIX = "Q-MUSE:"               # prefix that wakes the bot (uppercase compare)
# -------------------------------------------------------

STATE_DIR = os.path.expanduser("~/workspace/discord-relay")
POLL_SECS = 45
MAX_RUNTIME_SECS = 6 * 3600
PIDFILE = os.path.join(STATE_DIR, "poller.pid")
DEBUGLOG = os.path.join(STATE_DIR, "poller_debug.log")


def dbg(msg):
    try:
        with open(DEBUGLOG, "a") as f:
            f.write("%d %s\n" % (int(time.time()), msg))
    except Exception:
        pass


def pid_alive(pid):
    try:
        with open("/proc/%d/cmdline" % pid, "rb") as f:
            return b"poller.py" in f.read()
    except Exception:
        return False


def api_get(path, token):
    p = subprocess.run(
        ["curl", "-sS", "-m", "25",
         "-H", "Authorization: Bot " + token,
         "https://discord.com/api/v10" + path],
        capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError("curl failed: " + (p.stderr or "")[:200])
    return json.loads(p.stdout)


def is_addressed(m):
    try:
        if m["author"]["id"] == BOT_ID:
            return False
        c = m.get("content") or ""
        if ("<@%s>" % BOT_ID) in c or ("<@!%s>" % BOT_ID) in c:
            return True
        if c.strip().upper().startswith(WAKE_PREFIX):
            return True
        ref = m.get("message_reference")
        if ref and (m.get("referenced_message") or {}).get("author", {}).get("id") == BOT_ID:
            return True
        return False
    except Exception:
        return False


def read_watermark():
    try:
        with open(os.path.join(STATE_DIR, "last_seen_id")) as f:
            return f.read().strip() or None
    except Exception:
        return None


def main():
    os.makedirs(STATE_DIR, exist_ok=True)
    # Single-instance guard: refuse to run if another watcher is alive.
    try:
        with open(PIDFILE) as f:
            oldpid = int(f.read().strip())
        if pid_alive(oldpid):
            print(json.dumps({"error": "another poller running", "pid": oldpid}))
            return 1
    except Exception:
        pass
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    try:
        token = sys.stdin.readline().strip()
        if not token:
            print(json.dumps({"error": "no token on stdin"}))
            return 1
        last_seen = read_watermark()
        dbg("start pid=%d last_seen=%s" % (os.getpid(), last_seen))
        start = time.time()
        fails = 0
        while True:
            try:
                msgs = api_get("/channels/%s/messages?limit=20" % CHANNEL_ID, token)
                fails = 0
            except Exception as e:
                fails += 1
                if fails >= 5:
                    print(json.dumps({"error": "poll failed %d times: %s" % (fails, str(e)[:200])}))
                    return 1
                time.sleep(POLL_SECS)
                continue
            with open(os.path.join(STATE_DIR, "heartbeat"), "w") as f:
                f.write(str(int(time.time())))
            # Re-read the watermark every iteration so an external advance
            # (e.g. the agent marking messages handled) is honored and a stale
            # in-memory value can never cause replayed hits.
            file_seen = read_watermark()
            if file_seen and (not last_seen or file_seen > last_seen):
                last_seen = file_seen
            new = []
            for m in msgs:  # Discord returns newest-first
                if last_seen and m["id"] == last_seen:
                    break
                new.append(m)
            new.reverse()  # oldest-first for reporting
            if new:
                last_seen = new[-1]["id"]
                with open(os.path.join(STATE_DIR, "last_seen_id"), "w") as f:
                    f.write(last_seen)
            hits = [m for m in new if is_addressed(m)]
            if hits:
                out = [{"id": m["id"], "author": m["author"].get("username"),
                        "content": m.get("content") or ""} for m in hits]
                dbg("hit-exit last_seen=%s hits=%s" % (last_seen, [h["id"] for h in out]))
                print(json.dumps({"addressed": out}))
                return 0
            if time.time() - start > MAX_RUNTIME_SECS:
                print(json.dumps({"status": "recycle, no hits"}))
                return 0
            time.sleep(POLL_SECS)
    finally:
        try:
            if open(PIDFILE).read().strip() == str(os.getpid()):
                os.remove(PIDFILE)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
