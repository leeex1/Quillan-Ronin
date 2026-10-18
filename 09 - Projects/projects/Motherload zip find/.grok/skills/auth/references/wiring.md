# Wiring the routes (do this once)

**Live-preview popup is PRE-WIRED — do not create it.**
`signIn` opens `/auth/popup`; the template Vite plugin
(`authPopupPlugin` in `vite.config.ts`) serves it via `popup.server.ts`.
**Never** add `src/routes/auth/popup.tsx` (or any React page / client OAuth at
that path). Doing so loads the full app shell in the popup ("the app opened
instead of Google") — that is always wrong.

**1. Mount Better Auth** — create the catch-all API route (this is what makes
`/api/auth/*` work; the broker's OAuth callback lands here):

```ts
// src/routes/api/auth/$.ts
import { createFileRoute } from "@tanstack/react-router";
import { auth } from "@/lib/auth/server";

export const Route = createFileRoute("/api/auth/$")({
  server: {
    handlers: {
      GET: ({ request }) => auth.handler(request),
      POST: ({ request }) => auth.handler(request),
    },
  },
});
```

**2. Add a sign-in page** — buttons that kick off the broker flow. Import from
`@/lib/auth/client`. With the flag on, `authEnabled` is true in preview and
deployed, so the buttons show and work in the live preview; the `else` branch
shows while auth is still disabled (`VITE_AUTH_ENABLED=false`):

```tsx
// src/routes/login.tsx
import { createFileRoute } from "@tanstack/react-router";
import { GROK_PROVIDERS, authEnabled, signIn } from "@/lib/auth/client";

export const Route = createFileRoute("/login")({ component: Login });

function Login() {
  return (
    <main className="grid min-h-screen place-items-center p-6">
      <div className="w-full max-w-sm space-y-3">
        <h1 className="text-xl font-semibold">Sign in</h1>
        {authEnabled ? (
          GROK_PROVIDERS.map((p) => (
            <button
              key={p.providerId}
              type="button"
              onClick={() => signIn(p.providerId, { callbackURL: "/" })}
              className="w-full cursor-pointer rounded-md border border-neutral-300 px-4 py-2 hover:bg-neutral-100 dark:border-neutral-700 dark:hover:bg-neutral-900"
            >
              Continue with {p.label}
            </button>
          ))
        ) : (
          <p className="text-sm text-neutral-500">Sign-in is disabled.</p>
        )}
      </div>
    </main>
  );
}
```

`RedirectToSignIn` sends signed-out users to `/login` by default (override with
`<RedirectToSignIn to="/somewhere" />`). Style the page however you like — see
the `design-ui` skill.

That's it — call `signIn(providerId)` from your sign-in buttons. The popup,
bearer-token hand-off, and request attachment are all inside `src/lib/auth` +
the Vite plugin; leave them alone.
