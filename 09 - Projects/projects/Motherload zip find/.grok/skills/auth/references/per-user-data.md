# Per-user data (server-side — mandatory)

Pair auth with the DB (see the `neon` skill). A regular Postgres driver has full
DB access, so **every** server function that touches per-user data must verify
the caller and scope rows to them. Use the prewired **`authMiddleware`**: it
resolves the same-origin session to a verified `context.userId` (and rejects
scripted cross-site/sibling requests) — no token threading:

```ts
import { createServerFn } from "@tanstack/react-start";
import { getSql } from "@/lib/db";
import { authMiddleware } from "@/lib/auth/middleware";

export const listTodos = createServerFn({ method: "GET" })
  .middleware([authMiddleware])
  .handler(async ({ context }) => {
    const sql = await getSql();
    // Type the row shape — a server fn's return must be provably serializable.
    return sql<{ id: number; title: string; done: boolean }>`select id, title, done from todos where user_id = ${context.userId} order by id desc`;
  });

// Inputs go through `.validator()` (the current API); the client passes `{ data }`:
export const addTodo = createServerFn({ method: "POST" })
  .validator((title: string) => title.trim())
  .middleware([authMiddleware])
  .handler(async ({ context, data: title }) => {
    if (!title) return;
    const sql = await getSql();
    await sql`insert into todos (user_id, title) values (${context.userId}, ${title})`;
  });
// mutations must scope writes too: `... where id = ${id} and user_id = ${context.userId}`
```

Call these from **client code** (effects, event handlers, React Query) — that's
where `Sec-Fetch-Site: same-origin` holds:

```ts
useEffect(() => { listTodos().then(setTodos).catch(() => setTodos([])); }, []);
```

Semantics: signed out → the middleware throws `UnauthorizedError` (message
`"Unauthorized"`, `status` 401 — match it to send the visitor to sign-in), in the
live preview too (real auth). With auth disabled (`VITE_AUTH_ENABLED=false`) it
resolves the dev user (`"dev-user"`) in dev and preview only — the deployed flag
comes from the deployer (today always `"true"`), so deployed it rejects every
visitor — which is why an app without sign-in must not use the middleware at all
(see the `neon` skill). Keep `user_id` columns
`TEXT` (Better Auth uses text ids; the disabled dev user is `'dev-user'`). Never
trust a client-supplied user id — only the middleware / `requireUserId()` result.
