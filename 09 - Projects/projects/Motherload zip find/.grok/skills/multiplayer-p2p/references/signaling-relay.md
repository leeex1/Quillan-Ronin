# Signaling relay and API route (create once)

The relay is **yours**, not part of the kit: copy it as-is or serve the same
`RtcPollResponse` shape from any store. Only rendezvous traffic (roster +
SDP/ICE) passes through it; game data flows peer-to-peer.

## Schema — nothing to do by default

The relay below creates its two tables on first use
(`CREATE TABLE IF NOT EXISTS`, once per process) — nothing ships in
`migrations/` and the template itself never touches your database. If you'd
rather own or extend the schema (extra columns, your own migration ordering),
copy this into one of your app migrations; `IF NOT EXISTS` makes the runtime
ensure and your migration coexist safely:

```sql
-- CREATE TABLE IF NOT EXISTS webrtc_peers (
--   room TEXT NOT NULL,
--   peer_id TEXT NOT NULL,
--   name TEXT NOT NULL DEFAULT '',
--   last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
--   PRIMARY KEY (room, peer_id)
-- );
-- CREATE TABLE IF NOT EXISTS webrtc_signals (
--   id BIGSERIAL PRIMARY KEY,
--   room TEXT NOT NULL,
--   to_peer TEXT NOT NULL,
--   from_peer TEXT NOT NULL,
--   kind TEXT NOT NULL,
--   payload JSONB NOT NULL,
--   created_at TIMESTAMPTZ NOT NULL DEFAULT now()
-- );
-- CREATE INDEX IF NOT EXISTS webrtc_signals_inbox
--   ON webrtc_signals (room, to_peer, id);
```

## `src/lib/multiplayer/signaling.server.ts`

Copy this file as-is (or adapt it — it is yours, not part of the kit):

```ts
// src/lib/multiplayer/signaling.server.ts
/**
 * WebRTC signaling over the app database (Neon deployed, PGLite in preview).
 * Only rendezvous traffic passes through here — roster + SDP/ICE relay while a
 * mesh forms; game data then flows peer-to-peer. DB-backed so any serverless
 * instance can serve any poll. Mount at /api/rtc (see the multiplayer-p2p
 * skill); the client side lives in `@/lib/multiplayer`.
 *
 * The GET poll is the whole peer lifecycle: the first poll (since=0) IS the
 * join — it registers the peer, returns the roster, and prunes stale rows.
 * Peer ids are random per mount, so a fresh inbox never has old signals to
 * skip and no join/cursor handshake is needed.
 */
import { z } from "zod";
import { getSql, type Sql } from "@/lib/db";
import type { PeerRow, RtcPollResponse, SignalRow } from "./p2p";

const ID = z.string().regex(/^[a-zA-Z0-9_-]{1,64}$/);
const signalSchema = z.object({
  op: z.literal("signal"),
  room: ID,
  from: ID,
  to: ID,
  kind: z.enum(["offer", "answer", "ice"]),
  // SDP offers are typically 3–10KB; the cap only blocks abuse (payload is
  // re-serialized at insert — cheap at this size). An absent
  // payload is rejected here (JSON.stringify(undefined) has no .length).
  payload: z.unknown().refine((v) => v !== undefined && JSON.stringify(v).length <= 32_768, {
    message: "payload too large",
  }),
});
const leaveSchema = z.object({ op: z.literal("leave"), room: ID, peer: ID });
const postSchema = z.discriminatedUnion("op", [signalSchema, leaveSchema]);


const PEER_TTL_SECONDS = 30;
const SIGNAL_TTL_SECONDS = 60;

/**
 * The kit ships no migration: tables are created on first use (IF NOT EXISTS)
 * so the app's migrations/ namespace stays fully in the agent's hands. Agents
 * who want to own/extend the schema can copy the DDL from the multiplayer-p2p
 * skill into their own migration — both coexist safely. Memoized on globalThis
 * (the db.ts pattern) so dev HMR never runs two ensures concurrently; a failed
 * ensure clears the slot so the next request retries.
 */
const globalRef = globalThis as typeof globalThis & {
  __rtcSchemaPromise__?: Promise<void>;
};

function ensureSchema(sql: Sql): Promise<void> {
  globalRef.__rtcSchemaPromise__ ??= (async () => {
    await sql.query(
      `CREATE TABLE IF NOT EXISTS webrtc_peers (
         room TEXT NOT NULL,
         peer_id TEXT NOT NULL,
         name TEXT NOT NULL DEFAULT '',
         last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
         PRIMARY KEY (room, peer_id)
       )`,
    );
    await sql.query(
      `CREATE TABLE IF NOT EXISTS webrtc_signals (
         id BIGSERIAL PRIMARY KEY,
         room TEXT NOT NULL,
         to_peer TEXT NOT NULL,
         from_peer TEXT NOT NULL,
         kind TEXT NOT NULL,
         payload JSONB NOT NULL,
         created_at TIMESTAMPTZ NOT NULL DEFAULT now()
       )`,
    );
    await sql.query(
      `CREATE INDEX IF NOT EXISTS webrtc_signals_inbox
         ON webrtc_signals (room, to_peer, id)`,
    );
  })().catch((err) => {
    globalRef.__rtcSchemaPromise__ = undefined;
    throw err;
  });
  return globalRef.__rtcSchemaPromise__;
}

async function roster(sql: Sql, room: string): Promise<PeerRow[]> {
  // LIMIT bounds the blast radius of room-stuffing; the mesh caps out ~8.
  const rows = await sql.query<{ peer_id: string; name: string }>(
    `SELECT peer_id, name FROM webrtc_peers
     WHERE room = $1 AND last_seen > now() - make_interval(secs => $2)
     ORDER BY peer_id LIMIT 32`,
    [room, PEER_TTL_SECONDS],
  );
  return rows.map((r) => ({ id: r.peer_id, name: r.name }));
}

async function touchPeer(sql: Sql, room: string, peer: string, name: string) {
  await sql.query(
    `INSERT INTO webrtc_peers (room, peer_id, name, last_seen)
     VALUES ($1, $2, $3, now())
     ON CONFLICT (room, peer_id)
     DO UPDATE SET last_seen = now(), name = EXCLUDED.name`,
    [room, peer, name],
  );
}

/**
 * Rows are ephemeral; GC rides the polls instead of a cron: joins (since=0)
 * always prune, and ~2% of all other polls do too — so a busy room whose
 * cursors always advance still gets swept, without every heartbeat paying
 * the two DELETEs.
 */
async function prune(sql: Sql) {
  await Promise.all([
    sql.query(`DELETE FROM webrtc_signals WHERE created_at < now() - make_interval(secs => $1)`, [
      SIGNAL_TTL_SECONDS,
    ]),
    sql.query(`DELETE FROM webrtc_peers WHERE last_seen < now() - make_interval(secs => $1)`, [
      PEER_TTL_SECONDS,
    ]),
  ]);
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });
}

/** GET /api/rtc?room&peer&name&since — join (since=0), heartbeat, and inbox. */
async function handleGet(url: URL): Promise<Response> {
  const parsed = z
    .object({
      room: ID,
      peer: ID,
      name: z.string().max(64).default(""),
      since: z.coerce.number().int().min(0).default(0),
    })
    .safeParse({
      room: url.searchParams.get("room"),
      peer: url.searchParams.get("peer"),
      name: url.searchParams.get("name") ?? "",
      since: url.searchParams.get("since") ?? 0,
    });
  if (!parsed.success) return json({ error: "invalid query" }, 400);
  const { room, peer, name, since } = parsed.data;

  const sql = await getSql();
  await ensureSchema(sql);
  if (since === 0 || Math.random() < 0.02) await prune(sql);
  await touchPeer(sql, room, peer, name);
  const rows = await sql.query<{
    id: number;
    from_peer: string;
    kind: SignalRow["kind"];
    payload: unknown;
  }>(
    `SELECT id, from_peer, kind, payload FROM webrtc_signals
     WHERE room = $1 AND to_peer = $2 AND id > $3
     ORDER BY id LIMIT 200`,
    [room, peer, since],
  );
  const body: RtcPollResponse = {
    peers: await roster(sql, room),
    signals: rows.map((r) => ({
      id: r.id,
      from: r.from_peer,
      kind: r.kind,
      payload: r.payload,
    })),
  };
  return json(body);
}

async function handlePost(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return json({ error: "invalid JSON" }, 400);
  }
  const parsed = postSchema.safeParse(body);
  if (!parsed.success) return json({ error: "invalid request" }, 400);
  const msg = parsed.data;
  const sql = await getSql();
  await ensureSchema(sql);

  if (msg.op === "signal") {
    await sql.query(
      `INSERT INTO webrtc_signals (room, to_peer, from_peer, kind, payload)
       VALUES ($1, $2, $3, $4, $5)`,
      [msg.room, msg.to, msg.from, msg.kind, JSON.stringify(msg.payload)],
    );
  } else {
    await sql.query(`DELETE FROM webrtc_peers WHERE room = $1 AND peer_id = $2`, [
      msg.room,
      msg.peer,
    ]);
  }
  return json({ ok: true });
}

/** Request entrypoint for the /api/rtc route (GET poll, POST signal/leave). */
export async function handleSignaling(request: Request): Promise<Response> {
  try {
    if (request.method === "GET") return await handleGet(new URL(request.url));
    if (request.method === "POST") return await handlePost(request);
    return json({ error: "method not allowed" }, 405);
  } catch (error) {
    console.error("[rtc] signaling error:", error);
    return json({ error: "signaling failed" }, 500);
  }
}
```

## Mount the API route


```ts
// src/routes/api/rtc.ts
import { createFileRoute } from "@tanstack/react-router";
import { handleSignaling } from "@/lib/multiplayer/signaling.server";

const handle = ({ request }: { request: Request }) => handleSignaling(request);

export const Route = createFileRoute("/api/rtc")({
  server: { handlers: { GET: handle, POST: handle } },
});
```
