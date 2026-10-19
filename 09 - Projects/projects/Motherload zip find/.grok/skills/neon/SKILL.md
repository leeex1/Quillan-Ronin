---
name: neon
description: >
  Use Neon Postgres (the database) in this TanStack Start app. Use when the app
  needs to store or query data, persist state, or keep per-user data. Triggers on
  "database", "Postgres", "Neon", "save data", "store data", "persist", "tables",
  "SQL", "query", "migrations".
metadata:
  short-description: "Neon Postgres (with a local PGLite fallback) for this template"
user-invocable: false
---

# Neon Postgres

**The database is opt-in** (AGENTS.md §0.5): use it only when the app needs data
that outlives a browser session or is shared across devices. Otherwise ship no
migrations, don't import `@/lib/db`, and keep state in `localStorage` / zustand.

This template ships a ready-made, **dual-mode** database integration:

- **Configured** (env var set, e.g. deployed): real **Neon Postgres**.
- **Not configured** (sandbox live preview): the DB falls back to a local
  **PGLite** (embedded WASM Postgres), so the preview always renders. Build
  against the `@/lib/db` helper; both modes work with the same API.

Packages are **preinstalled** — do not `npm install` them: `pg` (node-postgres,
the regular Postgres driver) and `@electric-sql/pglite` (local DB fallback).

For **user accounts, sign-in, and reading the current user**, see the separate
**`auth` skill** — this skill is just the database.

## Turning the database on

Set `deploy.database` to `true` in `.grok/app-env.json`:

```json
{ "VITE_AUTH_ENABLED": "false", "deploy": { "database": true } }
```

That is what tells the platform to provision Neon for the deployed app; leave it
`false` and the deploy gets no `DATABASE_URL`, so the app silently runs on a
throwaway PGLite that loses its rows. Shipping `migrations/*.sql`, or sign-in,
provisions one regardless — this flag is for an app that queries `@/lib/db`
without either. It is not a `VITE_` key and never reaches the browser.

## Env vars — do **not** create a `.env` file

**Never write a `.env` / `.env.local` / `.env.example` for the database.** In
the sandbox live preview, leave `DATABASE_URL` unset — `@/lib/db` automatically
uses embedded PGLite. When the app is deployed, the platform injects
`DATABASE_URL` (Neon); you do not provision or write it yourself.

| Var | Where | Purpose |
|---|---|---|
| `DATABASE_URL` | server | Neon connection string when deployed (optional — PGLite fallback if unset) |

Never hardcode it; never expose non-`VITE_` vars to the client.

## Database (server-only)

`@/lib/db` exports `getSql()` and `dbSource`: a **regular Postgres driver**
(node-postgres, `pg`) against `DATABASE_URL`, or a local **PGLite** fallback when
unset. Same API either way — a tagged template (and `.query()`) resolving to
`rows[]`. Call ONLY from a `createServerFn` handler / server loader, never a
client component. Define schema in `migrations/`, not inline.

```ts
import { createServerFn } from "@tanstack/react-start";
import { getSql } from "@/lib/db";

export const listPosts = createServerFn({ method: "GET" }).handler(async () => {
  const sql = await getSql();
  // Type the row shape — a server fn's return must be provably serializable.
  return sql<{ id: number; title: string }>`select id, title from posts order by id desc`;
  // or: return sql.query<{ id: number; title: string }>("select id, title from posts where id = $1", [id]);
});
```

**Per-user data (only once the app has sign-in).** A regular driver has full DB
access, so scope **every** query to the authenticated user server-side — never
trust a client-sent id. Use the prewired **`authMiddleware`** to get a verified
`context.userId`, then filter by it. Full pattern (middleware, calling from
client code, fail-closed semantics) is in the **`auth` skill**:

```ts
import { authMiddleware } from "@/lib/auth/middleware";

export const listTodos = createServerFn({ method: "GET" })
  .middleware([authMiddleware])
  .handler(async ({ context }) => {
    const sql = await getSql();
    return sql<{ id: number; title: string }>`select id, title from todos where user_id = ${context.userId} order by id desc`;
  });
// mutations must scope writes too: `... where id = ${id} and user_id = ${context.userId}`
```

**Without sign-in (the default), do NOT use `authMiddleware` / `requireUserId`.**
The dev user they fall back to is a preview-only convenience. A deployed app's
`VITE_AUTH_ENABLED` is set by the platform, not by this workspace (today the
deployer always sets it to `"true"`), so deployed, both reject every visitor —
and an auth-off app ships no sign-in route for them to recover with. A
database-only app keeps its rows unowned: no `user_id` column, or one literal
constant. Add the middleware as part of turning sign-in on (the `auth` skill's
upgrade steps), and re-scope or drop those rows then.

Unowned rows are world-readable and world-writable through your public server
functions: never persist personal or sensitive data (names, emails, free text
about a person) in this mode, and leave out destructive bulk mutations
(delete-all, overwrite-all) — if the app needs them, propose sign-in instead.

## Migrations

`migrations/*.sql` are the single schema source. They apply to **Neon on deploy**
(`npm run build` runs `db:migrate` against `DATABASE_URL`, so Vercel ships with
the schema ready) and to the **PGLite** preview **automatically on startup**, so
dev matches prod.

Neither applier descends into subdirectories, so the Better Auth schema at
`migrations/auth/0001_auth.sql` is **not** applied unless the app turns sign-in
on and copies it up (**do not edit** it — see the `auth` skill). Put your app's
schema in NEW ordered files starting at `0002`:

```sql
-- migrations/0002_schema.sql — example for a todos app; use YOUR app's tables
create table if not exists todos (
  id         serial primary key,
  user_id    text not null,
  title      text not null,
  done       boolean not null default false,
  created_at timestamptz not null default now()
);
create index if not exists todos_user_id_idx on todos (user_id);
```

Never edit an applied file — it is tracked by name in `_migrations` and will not
re-run (add a new file instead; new files apply to the running preview on the
next request). Prefer idempotent statements (`… if not exists`). Tables with
per-user data should carry a `user_id text not null` column (TEXT, not UUID — the
preview dev user id is the string `'dev-user'`).

## Preview ↔ production parity

`getSql()` normalizes result types so both backends return identical, JSON-safe
shapes: `bigint`/`count(*)` → `number`, `date` → `'YYYY-MM-DD'` string,
`interval` → text, `numeric` → string. Remaining differences to respect:

- **`bigint` past 2^53 loses precision** as a number — cast `::text` if you
  ever need huge integers (row counts are fine).
- **Preview DB is in-memory**: wiped on dev-server restart, single-connection
  (no lock contention or concurrent-write conflicts), and loads **no
  extensions** — do not `create extension`; stick to core Postgres.
- **Neon's pooled endpoint keeps no session state** — don't rely on `SET`,
  `LISTEN/NOTIFY`, or session advisory locks.
- **Keep `user_id` columns `text`** — preview uses `'dev-user'`, production uses
  Better Auth's text ids; a `uuid` column breaks preview inserts.
- Deployed Neon queries traverse the network (and may cold-resume) — avoid
  N+1 query patterns that feel free against the in-process preview DB.
