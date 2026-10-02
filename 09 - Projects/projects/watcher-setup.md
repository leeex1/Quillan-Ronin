# Discord watcher — setup guide

This is the watcher config behind the Discord relay bot. Save both files
locally, fill in the CONFIG block in `discord_watcher.py`, and point OpenCode
(or any agent) at this README.

## What it does

Wakes an agent when its Discord bot is addressed in a channel — a mention,
a `Q-MUSE:`-style prefix, or a reply to one of the bot's messages. The agent
replies, then restarts the watcher. No 24/7 process babysitting the chat;
the exit itself is the wake signal.

## Files

- `discord_watcher.py` — the watcher. Fill in BOT_ID, CHANNEL_ID, WAKE_PREFIX
  at the top. Everything else works as-is.
- State lives in `~/workspace/discord-relay/` (change STATE_DIR if you like):
  `last_seen_id`, `heartbeat`, `poller.pid`, `poller_debug.log`. No credentials
  ever touch disk.

## Running it

1. Start it in the background: `python3 discord_watcher.py`
2. Feed the bot token on stdin (one line), then EOF. The token lives only in
   the process's RAM — never in a file, env var, or command line.
3. The script polls every 45 seconds. On a hit it prints JSON
   `{"addressed": [...]}` and exits 0.
4. The supervising agent: read the JSON, compose the reply, POST it to
   `https://discord.com/api/v10/channels/<CHANNEL_ID>/messages`,
   advance `last_seen_id` past handled messages, restart the watcher.

## Watchdog (recommended)

A separate scheduled check every 15 minutes: read the `heartbeat` file. If it
is older than 20 minutes, the watcher died — restart it. That is the whole
watchdog.

## Lessons baked in (learned the hard way)

- Use `curl` via subprocess, NOT Python `urllib`. urllib silently fails behind
  some proxies: the heartbeat keeps looking stale and tags get missed with no
  error. This exact bug cost a missed user tag.
- Re-read the watermark file every poll iteration. A stale in-memory watermark
  replays already-handled messages.
- Single-instance guard via pidfile. Two watchers sharing one watermark file
  will double-report.
- The first version of this also had a "quiet mode" (only post on tags/results).
  It was rescinded — the owner wants real working output, not silence.
  Keep the bot substantive: answer, report results, ask for decisions. Skip
  empty acks ("noted", "locked in").

## Verified against

- Bot: muse ai bot (1549557324346036244)
- Server: Quillan Ronin AI (706907653435162634)
- Channel: #llm-text-🧠 (1438282228898467951)
- Read, write, and threaded replies confirmed working 2026-09-15.
