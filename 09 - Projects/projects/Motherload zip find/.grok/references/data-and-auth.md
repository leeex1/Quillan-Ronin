# Data & auth — implementation

Read this **after** `AGENTS.md` §0.5 has already said the app needs a database
and/or sign-in. The decision (auth OFF by default, the closed trigger list, "no
migrations / no `@/lib/db` unless triggered") lives in `AGENTS.md`, not here.

Full guides + snippets: the **`neon` skill** (database) and the **`auth` skill**
(sign-in), under `.grok/skills/`.

## Database (`@/lib/db`, server-only)

- `const sql = await getSql()` from `@/lib/db`. Use it **only** inside
  `createServerFn` handlers / server loaders.
- Dual-mode: a regular Postgres driver (node-postgres, `pg`) when `DATABASE_URL`
  is set, else a local **PGLite** fallback — so the preview always renders.
- In preview, PGLite **bootstraps at server start** (`ensureDbReady`) once the
  app has migrations. Do not remove that.
- A deployed app is provisioned a real database when it ships `migrations/*.sql`
  or sign-in. An app that needs one without either says so with
  `"deploy": {"database": true}` in `.grok/app-env.json` — see the `neon` skill.

## Migrations

- `migrations/*.sql` is the single schema source: applied to **Neon on deploy**
  (`npm run build` runs `npm run db:migrate`, so Vercel ships with the schema
  ready) and to the **PGLite** preview automatically on startup.
- Add the app's tables as ordered files (`migrations/0002_*.sql`), not inline.
- An app that needs no database adds no `.sql` file, and then no migration runs
  anywhere.
- The Better Auth schema sits outside that scope in `migrations/auth/` (neither
  applier descends into it); the `auth` skill's "Turning sign-in on" copies it
  up. Do not edit it. Applied files are keyed by **basename**, so a database
  that already has `0001_auth.sql` will not re-run it.

## Server functions

`createServerFn` with input via `.validator()` — the current API on the
installed version (`.inputValidator()` is deprecated). Examples in the `neon`
and `auth` skills.

## Auth wiring (only once §0.5 says accounts)

- The app runs its **own** Better Auth at `/api/auth/*` and federates to the
  shared Grok auth broker for **Google** and **X**. The only other supported
  method is this app's own **email/password** (local Better Auth, off by
  default — enable only via `src/lib/auth/email-password.ts`; **never rewrite**
  `src/lib/auth/server.ts`). No other social providers, magic links, passkeys,
  or OTP/phone.
- Two routes: `src/routes/api/auth/$.ts` (mounts Better Auth at `/api/auth/*`)
  and `src/routes/login.tsx` (provider buttons via `signIn(providerId)`). Copy
  the snippets from the `auth` skill.
- The live-preview popup at `/auth/popup` is served by the template Vite plugin
  (`vite.config.ts` → `popup.server.ts`); `AGENTS.md` § "First scaffold" states
  the rule about not adding a React route there.
- Read the user with `useCurrentUser()` (`@/lib/auth/use-current-user`) and gate
  UI with `SignedIn` / `SignedOut` / `UserButton` (`@/lib/auth/gates`).
- Sign-in is **real even in the live preview** — it federates via a baked shared
  preview client — so a visitor is signed out until they sign in. Build real
  sign-in; do **NOT** scaffold demo/mock/hardcoded users.
- **Authorize every server function** with `authMiddleware`
  (`@/lib/auth/middleware`): `createServerFn().middleware([authMiddleware])`
  hands the handler a **verified** `context.userId` (resolved from the
  same-origin session; throws when signed out). Scope **every** query by that
  `user_id`. Never trust a client-sent id.

## Turning sign-in on (at scaffold or later)

`.grok/app-env.json` (`{"VITE_AUTH_ENABLED": "false"}`) is the switch, read by
`npm run dev` / `build` / `preview` alike through `scripts/with-app-env.mjs` —
which is why Vite is never started outside those scripts. Follow the `auth`
skill's **"Turning sign-in on"** (flag, schema, then routes) in that order: the
routes alone render the disabled branch. `npm run check:auth`, run against a
live dev server, fails when that server and the next build disagree about the
flag (exit 0 agree, 1 diverged, 2 could not observe).

## Env

On deploy the platform injects `DATABASE_URL` + per-app auth creds; live preview
needs neither (baked preview client, PGLite fallback). Deployed behind the gate,
signed-in Grok viewers get the app session automatically from `x-grok-identity`
(see the `auth` skill — `references/grok-identity.md`); the broker federation
covers anonymous viewers and no-gate contexts.

## HARD RULE — connector / AppData API (backend only)

**Never call connector or AppData APIs from frontend code.**

| Allowed | Forbidden |
| --- | --- |
| `createServerFn({ method: "POST" }).handler` that dynamic-imports `@/lib/app-data/client.server` and calls `callTool` | Importing `@/lib/app-data/client.server` from a route component, `useEffect`, event handler, or any client module |
| UI calling that **server function** only | Browser `fetch("/__gate/app-data/…")`, `fetch` to the connectors host, or any direct CallTool from the client |
| Types/constants from `@/lib/app-data` (no network) | Putting `x-connector-access-token`, connector JWTs, or gate secrets in client state, props, or `VITE_*` env |

**Flow (required):** browser → **this app's** `createServerFn` → **app backend**
SDK → public connectors host (`connectors.grok.me`, **auth required**) → gate.
Unauthenticated hits on the connectors host redirect to gate OIDC sign-in.

If you need Drive / Gmail / calendar / connector data, load the **`app-data`
skill** (`.grok/skills/app-data/SKILL.md`) and follow it exactly. Do not invent
a client-side connector client.
