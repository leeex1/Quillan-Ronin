---
name: auth
description: >
  Add user accounts and sign-in to this TanStack Start app. Use when the app
  needs authentication, sign-in, user accounts, protected routes, or per-user
  data. Triggers on "auth", "login", "log in", "sign in", "sign up", "account",
  "users", "authentication", "protected", "who is logged in", "current user",
  "per-user".
metadata:
  short-description: "Auth via the Grok broker (Google, X) or local email/password — no other methods supported"
user-invocable: false
---

# Auth

This app runs its **own** [Better Auth](https://better-auth.com) at
`/api/auth/*`, federating to the shared **Grok auth broker** (`auth.grok.me`)
via the `genericOAuth` plugin. This template wires **Google** and **X**.

**Supported sign-in methods — use ONLY these three; nothing else is supported:
Google, X, and email/password.** No other social/OAuth provider (GitHub, Apple,
Discord, …), no magic links, passkeys, OTP, phone/SMS, or anonymous sign-in. Do
not add entries to `GROK_PROVIDERS`. Method detail and the email/password switch
(edit **only** `src/lib/auth/email-password.ts`): `references/sign-in-methods.md`.
**Exception — gate viewers are signed in already; NEVER render login/re-auth
buttons for them. Connector / app-data apps: ONLY gate "Continue with Grok",
no Google/X buttons**: `references/grok-identity.md`.

**Sign-in is OFF by default** — the template ships `.grok/app-env.json` with
`{"VITE_AUTH_ENABLED": "false"}`, so only add accounts when the ask calls for
them (AGENTS.md §0.5). Switching it on is "Turning sign-in on" below.

**Once on, sign-in is REAL — including in the sandbox live preview.** Do **NOT**
scaffold demo/mock/hardcoded users. Preview: popup + baked preview client;
deployed: per-app client + `DATABASE_URL` + zero-click gate sign-in
(`references/prewired-and-env.md`).

**While OFF** (`VITE_AUTH_ENABLED=false`) a **dev user** is returned so a
non-auth app renders without a signed-in visitor — dev and preview only.
Deployed, the flag is the platform's (always `"true"`), so `requireUserId`
rejects every visitor — auth-off apps use neither it nor `authMiddleware`.

Everything is **preinstalled and pre-wired in `src/lib/auth/`** — do not
`npm install` anything; `better-auth` is the only auth package (never
`@neondatabase/*`, `@stackframe/*`, or `@clerk/*`). **Do not edit or rewrite any
file under `src/lib/auth/`** — `server.ts` least of all — except
`email-password.ts` for its one flag. Per-file map:
`references/prewired-and-env.md`.

**`/auth/popup` is already handled by the template Vite plugin**
(`vite.config.ts` → `popup.server.ts`): it never paints the React app. **Do NOT
create `src/routes/auth/popup.tsx`** (or any React page / client OAuth at that
path) — that shows the full app inside the popup, the common failure mode.

**Never write a `.env` / `.env.local` / `.env.example`** in this sandbox: live
preview needs **zero** env configuration and a deployed app gets its vars
injected by the platform. The knobs that exist are in
`references/prewired-and-env.md` — never expose a non-`VITE_` var to the client.

`migrations/auth/0001_auth.sql` is the Better Auth schema — **do not edit**. It
sits outside the globbed `migrations/` directory (neither applier descends), so
it is not applied to apps without sign-in; "Turning sign-in on" copies it up.

## Turning sign-in on

Do all of this — the routes alone render the disabled branch:

1. **Flag:** delete the `VITE_AUTH_ENABLED` key from `.grok/app-env.json` and
   **restart the dev server**. Vite reads env at startup, so HMR will not pick
   it up. `npm run dev`, `npm run build` and `npm run preview` all read that
   file through `scripts/with-app-env.mjs`, so preview and the built output flip
   together — never start Vite directly.
2. **Schema:** `cp migrations/auth/0001_auth.sql migrations/0001_auth.sql`, then
   restart so it applies. It is tracked by basename in `_migrations`, so a
   database that already has it will not re-run it.
3. **Routes:** add `src/routes/api/auth/$.ts` + `src/routes/login.tsx` — copy
   both from `references/wiring.md` (the catch-all API route is what makes
   `/api/auth/*` and the broker callback work).
4. **Sign out:** a login with no way out is not done — render `<UserButton />`
   from `@/lib/auth/gates` (wires `signOut()`; hides sign-out for gate sessions).
5. **Existing data:** wrap the app's server functions in `authMiddleware` (an
   auth-off app must not have been using it — see the `neon` skill). Rows from
   before sign-in existed are **development data**: drop and recreate them
   unless the user says otherwise — don't hand them to whoever signs in first.

## Building on it once it's on

- **Sign in / out:** `signIn(providerId)` and `signOut()` from
  `@/lib/auth/client`; `GROK_PROVIDERS` renders the buttons. The popup,
  bearer-token hand-off, and request attachment are internal — leave them alone.
  Prefer `<UserButton />` (it handles the pending and failure states); `signOut()`
  rejects when deployed if the server never confirms — catch it. Never
  `authClient.signOut()`: it leaves the preview bearer token attached to every
  later request, so the visitor stays signed in.
- **Reading the user:** `useCurrentUser()` is display-only (`null` means
  *loading OR signed out*, so never redirect on it alone); guard on
  `useCurrentUserState()`'s `isPending` instead. Gates (`SignedIn`, `SignedOut`,
  `SignInGate`, `RedirectToSignIn`, `UserButton`) live in `@/lib/auth/gates`.
  CTA hard rules, skeleton, and cookie-SSR zero-flash: `references/session-ui.md`.
- **Per-user data (mandatory):** every server function that touches per-user data
  must use the prewired `authMiddleware` and scope every read **and** write to
  `context.userId` — a Postgres driver has full DB access, so nothing else limits
  the query. Keep `user_id` columns `TEXT`; never trust a client-supplied user id;
  signed out, the middleware throws `UnauthorizedError` (401). Code and
  disabled-mode semantics: `references/per-user-data.md`.
- **Security model:** headless broker, `__Host-` cookies + `trustedOrigins`, and
  Fetch-Metadata sibling isolation are already wired — never weaken them to make
  an error go away (`references/sign-in-methods.md` covers the model and the
  "Invalid origin" fix).

