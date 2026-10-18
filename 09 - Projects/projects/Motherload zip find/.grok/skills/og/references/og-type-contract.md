# `og:type` — why `x:game`, and the rejected alternatives

The game signal is written as `"type": "x:game"` in `src/lib/og/site.json`; the
platform injector turns it into `<meta property="og:type" content="x:game">`.
This will live forever in every shipped game, so the channel is fixed:

| Option | Verdict |
| --- | --- |
| **`site.json` `"type": "x:game"` → injector emits `<meta property="og:type" content="x:game">`** | **Chosen.** You write the field in `src/lib/og/site.json`; the PWA injector emits the standard OG content-type property. Value is a **namespaced type** (`x:game`) so it cannot be confused with a future global OGP `game` type or bare `website`. X's card pipeline keys off this exact emitted value. Do not put the meta tag in `__root.tsx`. Do not shorten to bare `game`. |
| **`og:type="game"` (no `x:`)** | **Rejected.** Looks like a global OGP type that does not exist on [ogp.me](https://ogp.me/#types); the product contract is the namespaced `x:game` string. |
| **`twitter:card`** | **Wrong layer for the game signal.** Layout only (`summary_large_image`). The PWA injector always emits it — do not treat the `og` skill as the place that adds it. Never use it as the game type. |
| **Separate `x:type` / `x:card` meta properties** | **Do not invent.** Extra properties double surface area (agents forget one of two tags). Namespacing inside `og:type`'s content (`x:game`) is enough. |

X (Twitter) uses `og:type="x:game"` when unfurling `*.grok.me` links to present
the card as a **game** rather than a generic website. This is a product contract
with X's card pipeline — keep the tag, and do not "correct" it to `website`
during refactors. Non-games should omit `og:type` or use `website` (the scraper
default).
