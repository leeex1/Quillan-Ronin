# Supported sign-in methods (Google, X, email/password — and nothing else)

Use **only** these three — no other method is supported:

- **Google** and **X** — federated through the Grok broker (pre-wired here). The
  broker federates these two upstreams and nothing else, so do **not** add entries
  to `GROK_PROVIDERS` beyond them (the broker rejects an unknown `idp`).
- **Email + password** — this app's OWN Better Auth, persisted in your database
  (never the broker, never mocked). Better Auth is DB-backed in BOTH modes — real
  Postgres when deployed and the embedded PGLite in the sandbox preview — so
  email/password accounts are stored and survive across requests, **in preview
  too**. It's off by default. Enable it by editing **only**
  `src/lib/auth/email-password.ts`:

  ```ts
  // src/lib/auth/email-password.ts
  export const emailAndPasswordEnabled = true; // was false
  ```

  **Do not edit or rewrite `src/lib/auth/server.ts`** (or any other file under
  `src/lib/auth/` except `email-password.ts` for this flag). That file is
  pre-wired; "fixing" it by regenerating Better Auth config breaks live-preview
  sign-in.

  The pre-applied schema already has the `account.password` column — no migration
  needed. Then build sign-up / sign-in forms with `authClient.signUp.email(...)`
  and `authClient.signIn.email(...)` from `@/lib/auth/client`.

  **Do not** add `emailAndPassword` as a plugin entry (that is a syntax/type
  error). **Do not** invent a new Better Auth config.

  If sign-up/sign-in returns **"Invalid origin"**, do **not** disable CSRF and
  do **not** edit `server.ts`. The template's `trustedOrigins` already covers
  `*.grok-sandbox.com` and local loopback on port 8080 (`localhost` /
  `127.0.0.1` / `[::1]`). Open the app at one of those origins (not a random
  host/port).

Do **NOT** add or use anything else: no other social / OAuth providers (GitHub,
Apple, Discord, Microsoft, Facebook, …), and no magic links, passkeys, one-time
codes / OTP, phone / SMS, or anonymous sign-in.

## Security model (already handled — don't undo it)

- **Headless broker**: the broker offers the upstream sign-in methods
  and holds their shared secrets; this app only holds its own per-app client
  id/secret and names the upstream it wants via each provider's `idp` hint. The
  broker forwards straight to Google/X. Users never see the broker.
- **`__Host-` cookies + `trustedOrigins`**: a sibling `*.grok.me` app can't toss a
  `Domain=.grok.me` cookie, and Better Auth rejects cross-origin `/api/auth`
  calls.
- **Sibling isolation**: `authMiddleware` rejects scripted cross-site/same-site
  requests (Fetch-Metadata), so a sibling can't ride this app's session cookie
  into its server functions.
- The upstream Google/X tokens live only on the broker; this app only ever gets a
  broker-issued identity and mints its own local session.
