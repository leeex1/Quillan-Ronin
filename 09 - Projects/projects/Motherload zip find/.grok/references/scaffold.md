# First-scaffold snippets

Copy-paste bodies for the four entry files `AGENTS.md` § "First scaffold"
requires. They match the **installed** TanStack Start: `src/router.tsx` is
resolved by a **named `getRouter` export**, and older
`createRouter`-default-export / `app/`-directory conventions are rejected by the
plugin.

```tsx
// src/router.tsx
import { createRouter } from "@tanstack/react-router";
import { AppErrorComponent } from "@/lib/error-component";
import { routeTree } from "./routeTree.gen"; // generated on first dev/build

export function getRouter() {
  return createRouter({ routeTree, defaultErrorComponent: AppErrorComponent });
}
```

```tsx
// src/routes/__root.tsx — the document shell
import { createRootRoute, HeadContent, Outlet, Scripts } from "@tanstack/react-router";
import { AuthProvider } from "@/lib/auth/provider";
import { PreviewHostBridge } from "@/components/preview-host-bridge";
import appCss from "../styles.css?url";

const APP_NAME = "My App";

export const Route = createRootRoute({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: APP_NAME },
      { name: "theme-color", content: "#000000" },
    ],
    links: [
      { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" },
      { rel: "stylesheet", href: appCss },
      { rel: "manifest", href: "/__grok/manifest.webmanifest" },
      { rel: "apple-touch-icon", href: "/__grok/icon-180.png" },
    ],
  }),
  component: () => (
    <html lang="en" suppressHydrationWarning>
      <head>
        <HeadContent />
      </head>
      <body>
        {/* Keep this bridge — lets the Grok preview chrome drive the app; noops when not embedded. */}
        <PreviewHostBridge />
        <AuthProvider>
          <Outlet />
        </AuthProvider>
        <Scripts />
      </body>
    </html>
  ),
});
```

```tsx
// src/routes/index.tsx
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  return <main className="p-8">Hello</main>;
}
```

```css
/* src/styles.css */
@import "tailwindcss";

@layer base {
  button:not(:disabled), [role="button"]:not(:disabled) { cursor: pointer; }
}
```
