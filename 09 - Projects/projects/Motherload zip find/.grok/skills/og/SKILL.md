---
name: og
description: >
  Share-link previews and app identity for apps on *.grok.me: the injector-owned
  og:image card, the SVG favicon, and PWA icons for installable apps.
  Use when scaffolding, renaming, or restyling the app — and for share /
  unfurl / OG / Twitter card questions. A custom 1200×630 card from the app's
  own art is the default — games of every kind (DOM board/word games
  included), whimsical apps, creative tools, and brand-forward pages; only
  plain utilities keep the placeholder. Always run the brand-asset pass as a
  `task` subagent and never wait for it.
  Triggers on "share", "rename", "app name", "OG", "Open Graph",
  "twitter card", "unfurl", "og:image", "og:type", "x:game:image",
  "x-banner", "link preview", "social card", "thumbnail", "preview image",
  "favicon", "app icon", "PWA", "manifest", "installable", "home screen",
  "SEO", "meta description".
metadata:
  short-description: "Brand assets: og.jpg card, X feed banner, SVG favicon, PWA icons — always a non-blocking `task` subagent"
user-invocable: false
---

# Share cards, favicon, and app icons

A deployed app (`https://{name}.grok.me`) unfurls with a 1200×630 card; every app (preview included) shows a
favicon in the tab. **Share-card `<meta>` tags are not authored in `__root.tsx`** — the injector
(`scripts/grok-pwa-shared.mjs`) overwrites `og:*` and `twitter:card` on every HTML response. Identity data is
the only thing anyone writes, and the pass writes all of it — dispatching, you seed none of it:

- `src/lib/og/site.json` — not pre-seeded, created only if needed: `{ "title", "type"?: "x:game", "card"?: "custom", "color"?: "RRGGBB" }`; title defaults to the host slug.
- `public/og.jpg` — custom 1200×630 card (optional; placeholder otherwise)
- `public/x-banner.jpg` — games only: 50:11 (1200×264) X feed card
- `public/favicon.svg` — linked from root `head()`; until the pass lands the tab just shows the browser default, which fails nothing

**Extend `__root.tsx`; never replace it wholesale** (auth SSR, redesigns, skill excerpts): dropping the
favicon link ships a blank tab icon no local check catches.

## Decide: which card this app gets

**Default: a custom card** from the app's own art — games of every kind and rendering tech (Canvas/WebGL *and*
DOM board, card, word, puzzle, quiz: a tic-tac-toe grid of divs is still a game), whimsical apps, creative
tools, content- and brand-forward pages. **When in doubt, make the custom card.**

**Plain utility apps only** (converters, CRUD trackers, dashboards, notes/admin — apps whose face is the data)
keep the default `og.grok.me` placeholder: no `public/og.jpg`. Its URL, the `"color"` knob and the rename
rule: `references/placeholder-card.md`.

## `og:type` for games

**A game of any kind** carries `"type": "x:game"` in `src/lib/og/site.json` — the pass writes it, owning that
file. No hostname, never gated on a custom card, never "corrected" to `website`, bare `game`, `twitter:card`
or an invented `x:type`: X's pipeline keys off that exact value. Non-games omit it. Your check, not your
edit — missing it, or missing `public/x-banner.jpg` once a custom card exists, is a **BRAND WARNING**.

## Brand-asset pass: always a subagent, never waited for

**Have the `task` tool? Dispatch this pass as a subagent and never generate card art yourself** — inline it
puts minutes of generation latency in front of the user. As soon as name and palette settle (AGENTS.md §
"Parallel work"), dispatch this prompt verbatim. It is complete — do not open the references below to
enrich it; the pass reads them itself:

> You are the brand-asset pass. Follow the `og` skill, which tells you where to start. App `<NAME>`,
> `og:type` `<TYPE>`, palette `<PALETTE>`. You solely own `public/` brand assets and `src/lib/og/site.json`.

Keep building. Stay sequential only if the user is art-directing or the art to reuse doesn't exist yet.

**Never wait for it.** No `wait_tasks`, and **never `get_task_output` on the brand task**: reading a task's
output consumes it, and a consumed task sends no completion notification, so the card's result — a failure
included — would reach nobody. Answer as soon as the app renders; the pass wakes you later, and that turn is
**one short sentence at most** — one that asks for a republish, since `public/og.jpg` ships in the build
and a card that lands after a publish never reaches the live app on its own:
"Added the share card — publish again if you already did." / "The card failed; the default one stands."

While the pass keeps `/workspace/.grok/og-pending` fresh, brand checks say nothing about the card: in flight
is not a finding. The marker goes stale after 10 minutes, so a very long pass lets the warning through — but
a brand warning while it runs is never a cue to redo its work.

**No `task` tool? Then you are the pass** — build the assets now; nothing else will. Whoever runs it claims
that marker, stages files under `/workspace/.grok/` — never inside `public/`, which `vite build` copies
verbatim into the deployed app — hands it over with `scripts/write-atomic.mjs` so no build reads half a JPEG,
and self-checks with `node scripts/brand-check.mjs --game`.

## Build the assets — the pass reads these, the parent does not

**Dispatching? Do not open them** — the prompt above is complete. One read carries this procedure in your
context every later turn while the subagent does the work anyway.

**You are the pass?** Start at `references/brand-pass.md`, then read
per asset you owe: `references/custom-card.md` for `public/og.jpg`, `references/x-banner.md` for the
games-only `public/x-banner.jpg`, `references/favicon-and-icons.md` for `public/favicon.svg` plus the PWA
raster icons — those only when the user asked for installable/PWA, never invent a manifest. **Hand-author
that SVG, never `imagine_text_to_image`**: it must stay crisp at 16px. Writing `site.json` for a game?
`references/og-type-contract.md` argues the spellings X rejects.

Regenerate on rename or a material identity change — `APP_NAME`, the `site.json` `title` and the baked-in card
title move together. Without `imagine_text_to_image` or the xAI Images API, keep the `og.grok.me` card; never
ship a broken `og:image` URL.

## Not supported

No `/api/og` route, no runtime image renderer, no per-route cards — the card is one static site-wide image
(`public/og.jpg` or the placeholder service). If you add `robots.txt`, never blanket `Disallow: /`: crawlers
must fetch `/` to read the tags.
