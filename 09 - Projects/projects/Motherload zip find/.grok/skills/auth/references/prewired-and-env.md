# What's pre-wired, and the env vars

## `src/lib/auth/`

| File | Use it for |
|---|---|
| `client.ts` | Browser client. `signIn(providerId)`, `signOut()`, `authEnabled`, `GROK_PROVIDERS`. |
| `server.ts` | The Better Auth instance (server-only). **Do not edit or rewrite.** Import only from `/api/auth/$`. |
| `email-password.ts` | **Only** place to enable local email/password (`emailAndPasswordEnabled = true`). |
| `popup.server.ts` | Live-preview popup handler (server-only). Already wired by the Vite plugin — do not create a route for it. |
| `providers.ts` | `GROK_PROVIDERS` — the fixed broker upstream list (Google and X only; don't add others). |
| `use-current-user.ts` | `useCurrentUser()` / `useCurrentUserState()` React hooks. |
| `gates.tsx` | `SignedIn`, `SignedOut`, `RedirectToSignIn`, `UserButton`. |
| `middleware.ts` | `authMiddleware` for server functions → verified `context.userId`. |
| `verify.server.ts` | `requireUserId()` / `getSessionUser()` (server-only) for manual wiring. |

## How each mode gets its credentials

- **Live preview** (`*.grok-sandbox.com`): the app is an embedded iframe, so
  sign-in opens a **popup** (a top-level redirect to the broker can't work inside
  the iframe) and federates via a baked shared **preview client**
  (`src/lib/auth/preview.ts`). The handler 302s straight to the broker/upstream
  login; on return the popup posts the session bearer back in a tiny HTML page.
  Sessions (and email/password users) persist in the app's embedded PGLite DB —
  the SAME DB as app data — and, since the iframe's cookies are partitioned,
  ride that bearer token; all of it lives in `src/lib/auth`.
  Restarting the preview resets the DB.
- **Deployed**: the deployer injects a per-app client + `DATABASE_URL`, so
  sign-in persists identities in Postgres.

## Env vars — do **not** create a `.env` file

**Never write a `.env` / `.env.local` / `.env.example` for auth (or anything
else) in this sandbox.** Live preview sign-in works out of the box with **zero**
env configuration: the server falls back to the baked preview client in
`src/lib/auth/preview.ts`, derives the `*.grok-sandbox.com` origin per-request,
mints a process-stable session secret, and persists sessions in embedded
PGLite. Deployed apps get `GROK_AUTH_*` / `BETTER_AUTH_*` / `DATABASE_URL`
injected by the platform — still not something you write into a file.

Optional process-env knobs (platform / rare overrides only — **do not** put
these in a file you create):

| Var | Where | Purpose |
|---|---|---|
| `VITE_AUTH_ENABLED` | client | `"false"` in the shipped `.grok/app-env.json` (dev user); drop the key to turn sign-in ON. Only client-visible auth flag |
| `BETTER_AUTH_URL` | server | app's own public origin; unset in preview (origin is derived per-request) |
| `BETTER_AUTH_SECRET` | server | signs this app's own sessions (process-stable fallback in preview; survives HMR) |
| `GROK_AUTH_ISSUER` | server | the shared broker (defaults to `https://auth.grok.me`) |
| `GROK_AUTH_CLIENT_ID` / `GROK_AUTH_CLIENT_SECRET` | server | per-app client (falls back to the preview client) |
| `DATABASE_URL` | server | when deployed, Better Auth persists here (preview persists to the embedded PGLite — same DB as app data) |

Never expose a non-`VITE_` var to the client. The preview client id/secret live
server-only in `src/lib/auth/preview.ts`.
