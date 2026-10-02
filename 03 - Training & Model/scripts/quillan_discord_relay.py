#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan Discord relay — OpenCode instance (C20-ARTIFEX + C10-CODEWEAVER).
Implements deploy/discord/PROTOCOL.md. One script, two instances (env via relay.json).
Outbound: outbox/*.txt -> #llm-text with PREFIX -> sent/.
Inbound: #llm-text (minus own bot) -> inbox/{ms}_{author}.txt envelope.
Agent replies ONLY via outbox. Never auto-replies (loop guard).
Security: token read from TOKEN_FILE, never logged/printed (CWE-532).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import discord

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
CFG = json.loads((REPO / "deploy" / "discord" / "relay.json").read_text(encoding="utf-8"))
CHANNEL_ID = int(CFG["channel_id"])
PREFIX = CFG["prefix"]
CALLSIGN = CFG["callsign"]
INBOX = Path(CFG["inbox"]); OUTBOX = Path(CFG["outbox"]); SENT = Path(CFG["sent"])
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}")


def load_token() -> str:
    text = Path(CFG["token_file"]).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("bot key:"):
            line = line.split(":", 1)[1].strip()
        m = TOKEN_RE.search(line)
        if m:
            return m.group(0)
    raise SystemExit("FATAL: no bot token found in TOKEN_FILE")


def plain(text: str) -> str:
    """Owner order: channel output must read clean. ASCII-only, short lines.
    Fancy unicode renders as ? boxes on some clients (seen tonight)."""
    rep = {"\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2190": "<-",
           "\u2713": "OK", "\u2714": "OK", "\u2717": "X", "\u26a0": "!",
           "\u2191": "^", "\u2193": "v", "\u2026": "...", "\u00a0": " "}
    for k, v in rep.items():
        text = text.replace(k, v)
    text = "".join(c if ord(c) < 128 or c == "\n" else "?" for c in text)
    lines, cur = [], ""
    for word in text.split():
        if len(cur) + 1 + len(word) > 120:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def chunk(text: str, n: int = 1900):
    return [text[i:i + n] for i in range(0, len(text), n)] or [""]


intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
client = discord.Client(intents=intents)


async def post_outbox(channel: discord.TextChannel):
    for f in sorted(OUTBOX.glob("*.txt")):
        if (SENT / f.name).exists():  # twin-instance guard: already posted
            f.unlink(missing_ok=True)
            continue
        try:
            body = f.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if not body:
            f.unlink(missing_ok=True)
            continue
        msg = plain(body if body.startswith("[") else f"{PREFIX} {body}")
        try:
            for part in chunk(msg):
                await channel.send(part)
            (SENT / f.name).write_text(body, encoding="utf-8")
            f.unlink(missing_ok=True)
            print(f"POSTED outbox/{f.name} ({len(body)} chars)", flush=True)
        except discord.HTTPException as e:
            print(f"POST-FAIL outbox/{f.name}: status={e.status}", flush=True)


@client.event
async def on_ready():
    print(f"LOGIN ok as {client.user} (id={client.user.id})", flush=True)
    channel = client.get_channel(CHANNEL_ID)
    if channel is None:
        try:
            channel = await client.fetch_channel(CHANNEL_ID)
        except discord.HTTPException:
            print("FATAL: cannot access channel (invited? perms?)", flush=True)
            return
    print(f"CHANNEL ok #{channel.name} ({channel.id})", flush=True)
    import time as _t
    mark = REPO / "deploy" / "discord" / ".last_handshake"
    last = float(mark.read_text().strip()) if mark.exists() else 0.0
    if _t.time() - last > 1800:  # handshake at most every 30 min, no restart spam
        await channel.send(plain("[SYS] relay-openCode online. Handshake start."))
        mark.write_text(str(_t.time()))
        print("HANDSHAKE posted", flush=True)
    else:
        print("HANDSHAKE skipped (recent)", flush=True)
    hb = REPO / "deploy" / "discord" / ".relay_heartbeat"
    seen = REPO / "deploy" / "discord" / ".last_seen_id"
    while True:
        await post_outbox(channel)
        hb.write_text(str(_t.time()))  # Muse spec: heartbeat every loop
        await asyncio.sleep(int(CFG.get("poll_sec", 5)))


def duty_digest() -> str:
    """Duty-officer card: live facts, no secrets, no model calls.
    Best val from training log, run state, gateway, queues."""
    import urllib.request, json as _json
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/health",
                                      headers={"User-Agent": "quillan-relay"})
        with urllib.request.urlopen(req, timeout=6) as r:
            h = _json.loads(r.read().decode("utf-8", errors="replace"))
        models = h.get("models_available", [])
        gw = (f"gateway healthy, {len(models)} models live" if models
              else "gateway up")
    except Exception:
        gw = "gateway down, training likely holding CPU"
    best, state = "?", "idle"
    try:
        logp = REPO / "training_logs" / "mini_60_fast.err.log"
        if logp.exists():
            txt = logp.read_text(encoding="utf-8", errors="replace")
            import re as _re
            bests = _re.findall(r"val_loss=([0-9.]+)", txt)
            if bests:
                best = min(bests, key=float)
            state = ("run finished, best promoted" if "COMPLETED" in txt
                     else "run in progress" if "STARTING" in txt else "idle")
    except OSError:
        pass
    n_out = len(list(OUTBOX.glob('*.txt')))
    queue = "nothing queued" if n_out == 0 else f"{n_out} reply(s) queued"
    return (f"{gw} | Mini best val {best} | {state} | {queue}")


def live_status() -> str:
    return duty_digest()


@client.event
async def on_message(message: discord.Message):
    if message.channel.id != CHANNEL_ID:
        return
    if message.author == client.user:  # loop guard 1: never self
        return
    if message.author.bot and (message.content or "").strip().startswith("[SYS]"):
        return  # loop guard 2: never answer system posts
    author = f"{message.author.display_name} ({message.author.name})"
    content = (message.content or "").strip()[: int(CFG.get("max_len", 1800))]
    if not content:
        return
    mentioned = client.user in message.message_mentions if hasattr(message, "message_mentions") else False
    if not mentioned and hasattr(message, "mentions"):
        mentioned = any(u == client.user for u in message.mentions)
    # Watcher-spec third trigger: reply to one of my own messages.
    replied_to_me = False
    try:
        ref = getattr(message, "reference", None)
        if ref is not None and getattr(ref, "message_id", None):
            rm = getattr(message, "referenced_message", None)
            if rm is None:
                try:
                    rm = await message.channel.fetch_message(ref.message_id)
                except discord.HTTPException:
                    rm = None
            if rm is not None and getattr(rm.author, "id", None) == client.user.id:
                replied_to_me = True
    except Exception:
        pass
    wants = bool(mentioned or replied_to_me or content.startswith(CALLSIGN))
    stamp = datetime.now(timezone.utc).isoformat()
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", message.author.name)[: 24]
    fname = f"{int(time.time() * 1000)}_{safe}.txt"
    (INBOX / fname).write_text(
        f"from: {author}\nauthor_id: {message.author.id}\n"
        f"at: {stamp}\nwants_reply: {str(wants).lower()}\n---\n{content}\n",
        encoding="utf-8")
    (REPO / "deploy" / "discord" / ".last_seen_id").write_text(str(message.id))
    print(f"INBOX {fname} wants_reply={wants}", flush=True)
    # Owner order 2026-09-15: no ack-stubs. But a TAG gets a REAL reply on the
    # spot (duty facts), so nobody ever double-messages. Untagged = silent file.
    # Full depth still follows on the next agent turn.
    if wants:
        try:
            echo = content[:200]
            await message.channel.send(plain(
                f"{PREFIX} heard you: \"{echo}\" -- {live_status()} -- "
                f"full answer from me next turn."))
            print(f"TAG-REPLY {fname}", flush=True)
        except discord.HTTPException as e:
            print(f"TAG-REPLY-FAIL {fname}: status={e.status}", flush=True)
    else:
        print(f"FILED {fname} (untagged, silent)", flush=True)


def single_instance():
    """Watcher-spec pidfile guard, Windows edition: file lock, exit if held.
    Twin relay instances caused every double-post tonight."""
    import msvcrt
    lockf = REPO / "deploy" / "discord" / ".relay.lock"
    fh = open(lockf, "w")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("FATAL: another relay holds the lock, exiting", flush=True)
        raise SystemExit(1)
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


if __name__ == "__main__":
    _lock = single_instance()  # held for process lifetime
    client.run(load_token())
