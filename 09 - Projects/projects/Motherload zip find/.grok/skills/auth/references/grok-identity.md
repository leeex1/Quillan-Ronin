# Sign in with Grok (deployed apps — zero clicks)

Behind the edge gate, every proxied request from a signed-in Grok viewer
carries an unforgeable `x-grok-identity` JWT (EdDSA, minted per request; the
gate strips any client-supplied copy). The pre-wired `gateIdentitySessions`
plugin (`src/lib/auth/gate-session.server.ts`) verifies it against the gate's
JWKS (`/__gate/identity-key`, via `src/lib/auth/gate-identity.server.ts`) and,
when the app has no session yet, materializes the Better Auth session for that
viewer automatically — no sign-in button, no redirect, no broker round-trip.
`useSession` / `useCurrentUser` simply return the Grok user.

The broker OAuth flow is the **fallback** for anonymous/public viewers and for
contexts without the gate. The live preview gets the same zero-click identity
from the in-VM preview proxy at `http://127.0.0.1:6014` (preview tokens carry
audience `preview`); the popup mechanism remains the fallback there.

## Never render sign-in or re-auth UI to a gate viewer

A gate-authenticated viewer is **already signed in**, and the gate refreshes
connector tokens on every proxied request — **the platform has no re-auth
concept**. Never render "Re-auth with Grok", "Sign in again", "Refresh
session", or a standing "Continue with Grok" button: sign-in UI may appear
only in the `app-data` skill's `login` error state, after a connector call
actually returned `loginRequired: true`.

The live preview gets the same zero-click session, so a sign-in button visible
to the owner in the preview indicates a bug (a CTA rendered while the session
check was still pending, or rendered in reaction to a data error) — fix it,
don't restyle it. For any fallback sign-in surface use `<SignInGate>` from
`@/lib/auth/gates`: it renders sign-in UI only after the session check resolved
to no user, never during loading and never from a data error.

Sign-out is also a no-op for a gate session — the next request re-materializes
it from `x-grok-identity`, an instant sign-back-in loop. `<UserButton />`
already hides its sign-out control for gate sessions; never build a custom
sign-out (or any sign-out route/handler) for gate viewers.

## Files (pre-wired — do not edit)

| File | Role |
|---|---|
| `gate-identity.server.ts` | Verifies the gate's `x-grok-identity` viewer JWT (EdDSA vs the gate JWKS; fail-closed). Server-only. |
| `gate-session.server.ts` | Better Auth plugin that turns a verified gate identity into the app session with zero clicks. Already registered in `server.ts`. |

## Env (deployer-injected)

| Var | Scope | Meaning |
|---|---|---|
| `GROK_PROJECT_ID` | server | deployed apps: enables "Sign in with Grok" (`x-grok-identity` audience check `app:<project_id>`) |
| `GROK_GATE_ORIGIN` | server | gate public origin override (JWKS + issuer pin); unset → preview mode (no `GROK_PROJECT_ID`) defaults to the in-VM proxy `http://127.0.0.1:6014` (audience `preview`), deployed mode derives it from the inbound host |

Deployed behavior: gate-authenticated viewers are signed in automatically from
`x-grok-identity`; the deployer also injects a per-app broker client +
`DATABASE_URL`, so the fallback sign-in persists identities in Postgres.

## Connector / app-data apps: gate sign-in only

When the `app-data` skill applies, the login page offers ONLY "Continue with
Grok" via the gate sign-in — the zero-click `x-grok-identity` session above, or
the gate-built `loginUrl` returned by a connector call. Do not wire Google/X
buttons for these apps: a broker login can mint an identity that is not the
gate viewer the connector data belongs to. The three-method rule applies to
apps without connector data. Still no new `GROK_PROVIDERS` entries, still never
edit `src/lib/auth/`.
