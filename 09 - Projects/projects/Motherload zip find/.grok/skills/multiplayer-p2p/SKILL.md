---
name: multiplayer-p2p
description: >
  Peer-to-peer realtime multiplayer over WebRTC data channels: every user of
  the deployed app connects directly to every other user (full mesh), the
  server only brokers the handshake at /api/rtc. Lowest possible latency, zero
  per-message server cost. Use for 2-8 player co-op/casual realtime: shared
  cursors, drawing, party games, casual action. Triggers: p2p, peer to peer,
  webrtc, low latency multiplayer, direct connection.
metadata:
  short-description: "WebRTC P2P mesh, signaled at /api/rtc"
user-invocable: false
---

# Multiplayer (WebRTC peer-to-peer)

All visitors on the same deployed domain join one default room, opening a
native WebRTC data channel directly to every other visitor — game traffic
itself never touches a server. A tiny relay at `/api/rtc` handles only the
routing of the connection handshake (SDP/ICE) while peers connect. What you
use from the kit is client-side only; the relay is yours.

Latency is browser↔browser (often 5–40ms) with zero per-tick server cost.

| Piece | Path |
|---|---|
| Mesh primitive (start here) | `P2PRoom` from `@/lib/multiplayer` |
| React room binding (optional, you create) | `src/lib/multiplayer/use-p2p-room.ts` |
| Signaling relay (you create) | `src/lib/multiplayer/signaling.server.ts` |
| HTTP mount (you create) | `src/routes/api/rtc.ts` |

**Trust model — read before choosing P2P.** There is no server authority:
every peer runs its own copy of the rules and can lie (position, score,
anything). Peers also learn each other's IP addresses during ICE. P2P is for
**co-op and casual play among people who choose to play together** — never for
competitive ranking, cheat-sensitive, or anonymous-stranger matchmaking.
Competitive or cheat-sensitive play is not supported in this template: push
back in product terms rather than shipping it on P2P.

Practical limits: a full mesh is O(N²) connections — cap rooms at ~8 peers.
Roughly 10–20% of peer pairs sit behind strict NATs and cannot connect; the
kit surfaces this per peer as `connectionState: "failed"` — show it in the
UI rather than hanging.

## Setup (once)

Create the two server files — nothing works without them:

1. `src/lib/multiplayer/signaling.server.ts` — the DB-backed signaling relay
   (Neon deployed, PGLite in preview).
2. `src/routes/api/rtc.ts` — mounts it at `/api/rtc` (GET poll, POST
   signal/leave).

**Copy both from `references/signaling-relay.md`**, which also carries the
schema note: the relay creates its own two tables on first use
(`CREATE TABLE IF NOT EXISTS`), so **nothing goes in `migrations/`** unless you
deliberately want to own the schema.

## Using the primitive

`P2PRoom` is framework-free, and a "room" is just a rendezvous key — a lobby
code, a 1:1 call id, a shared-document id, any string (≤64 chars). Any
architecture sits on top of the same three calls:

```ts
import { P2PRoom } from "@/lib/multiplayer";

const p2p = new P2PRoom({
  room: "doc-42",
  selfId: myId,
  name: "ani",
  onPeersChanged: (peers) => render(peers),
  onMessage: (from, data, channel) => apply(from, data, channel),
});
await p2p.join();
p2p.broadcast(state); // unreliable "state" channel — game-rate, stale drops
p2p.send(event, to); // reliable channel — exactly-once events (to optional)
p2p.close();
```

For the common "everyone on this app plays together" shape in React, copy the
`useP2PRoom` hook (plus a worked component: game-rate broadcast loop at ~20
sends/s, reliable one-shot events) from `references/react-binding.md`.

Patterns:

1. `broadcast()` = unreliable/unordered, for continuously-refreshed state
   (positions, cursors). `send()` = reliable/ordered, for events that must
   arrive exactly once. Never stream game-rate state on `send()`; interpolate
   between broadcasts for smooth motion.
2. Late joiners know nothing: on a new peer appearing in `p2p.peers`, an
   existing peer should `send()` it the current shared state. Exactly one
   peer must answer: compare ids among the peers that were ALREADY in the
   room (your `selfId` plus `p2p.peers` minus the newcomer) and answer only
   if your `selfId` is the smallest — so two simultaneous joiners neither
   double-answer nor go unanswered.
3. Room ids: omit for "everyone on this app plays together"; pass
   `room: code` for private lobbies (generate a short code, put it in the URL).
4. Peers disappear without goodbye (tab close, sleep): treat a peer missing
   from `p2p.peers` as gone and drop its entities.
5. A React binding that captures `room`/`name` on first render (the one in
   `references/react-binding.md` does) needs a remount to change them — key the
   component on the room code.

## Diagnostics

Each entry in `p2p.peers` carries `connectionState`, `rttMs` (data-channel
ping), and `candidateType` (`host`/`srflx` = direct). To override STUN, add
`VITE_STUN_URLS` (comma-separated) to `.grok/app-env.json` and **restart the dev
server** (Vite reads env at startup; HMR will not pick it up) — never write a
`.env` in this sandbox.
