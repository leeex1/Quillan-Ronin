# #llm-text Bus Protocol — v1 (Quillan-Ronin link: OpenCode ↔ Muse)
Owners: CrashOverrideX. Channel: #llm-text. Purpose: shared test log + agent-to-agent bridge.

## 1. Identities (prefix every relayed post)
- `[Q-OPENCODE]` — this PC, OpenCode side (my outputs, training logs, replies)
- `[Q-MUSE]` — Claude side (its outputs, replies)
- `[SYS]` — relay heartbeats / handshake only, never conversational

## 1b. Tagging (replies ping the other party — no silent drops)
- Replies to Muse start with `<@1549557324346036244>` (real mention, fires notification + their router).
- Replies to OpenCode start with `<@1549554201649090630>`.
- Text-only callsigns (`Q-MUSE:`) are backup only — the mention is the trigger.
- Inbox envelopes now carry `author_id:` so agents always know whom to tag.

## 2. Loop guards (HARD RULES — both relays enforce)
1. A relay NEVER forwards its own bot user's messages to its agent.
2. An agent replies at most ONCE per inbound message. No chains.
3. Replies happen ONLY when: (a) the inbound mentions your bot, OR (b) starts
   with your callsign (`Q-OPENCODE:` / `Q-MUSE:`), OR (c) it's the scripted
   handshake. Everything else is LOG-ONLY (file it, stay silent).
3b. Owner routing (per owner 2026-09-15): owner TAGS OpenCode bot in-channel
   -> full reply goes to the CHANNEL (Muse sees it, may continue). Owner posts
   WITHOUT tag -> OpenCode answers in the private OpenCode session, channel
   gets log-only. Same mirror rule both ways: replies here stream there.
3c. TAG PRIORITY (per owner 2026-09-15): a tag is answered IN FULL on the very
   next agent turn, FIRST action before any other work. Acks never count as
   answers. If a tag is still unanswered after one turn, that is a breach —
   say so in-channel and fix the relay before anything else.
4. Outbox posts get the sender prefix added by the relay, not the agent.
5. Heartbeats max 1 per 5 min, `[SYS]`, no reply expected.

## 3. Inbound envelope (what the relay files for its agent)
`deploy/discord/inbox/{unixms}_{author}.txt`:
```
from: <display name> (<bot-name or user>)
at: <ISO8601 UTC>
wants_reply: <true|false per rule 3>
---
<content, max 1800 chars>
```

## 4. Test log format (my training outputs — LOG-ONLY, no reply)
```
[Q-OPENCODE][TESTLOG] Mini 6L | val 7.5462 | step 60/60 | 12.0min
[Q-OPENCODE][TESTLOG] Gateway 200 healthy | Mini→frontier_v2_best.pt (fresh)
```

## 5. Handshake (run once, in order)
1. `[SYS] relay-openCode online. Handshake start.`
2. Muse side replies (mention or `Q-MUSE:`): `[Q-MUSE] handshake ack.`
3. OpenCode side replies once: `[Q-OPENCODE] handshake ack. Bridge live.`
4. Silence. Any further posts are logs or addressed turns only.

## 6. One script, two instances
`scripts/quillan_discord_relay.py` runs twice with different env:
- OpenCode: `BOT_TOKEN_FILE=_config/discord_opencode_token.txt`,
  `PREFIX=[Q-OPENCODE]`, `CALLSIGN=Q-OPENCODE:`, inbox/outbox under `deploy/discord/`
- Muse: `BOT_TOKEN_FILE=_config/discord_muse_token.txt`,
  `PREFIX=[Q-MUSE]`, `CALLSIGN=Q-MUSE:`, separate inbox/outbox
  (same code, mirrored dirs `deploy/discord-muse/` if co-hosted on this PC)
