# Running the brand-asset pass (the pass's own contract)

Read this when you *are* the pass — dispatched as the brand subagent, or
building the assets inline because no `task` tool exists. The parent keeps
building and never waits, so everything below is yours to get right unobserved.

## 1. Claim the marker, keep it fresh, always release it

`touch /workspace/.grok/og-pending` before generating and again before each
`imagine_*` call — while that marker is fresh `brand-check.mjs` suppresses the
missing-card warnings the parent would otherwise act on, and it goes stale after
10 minutes so a killed pass cannot silence the check forever.
`rm -f /workspace/.grok/og-pending` on every exit path, success or not.

## 2. Hand every file over atomically

`public/og.jpg`, `public/x-banner.jpg` and `src/lib/og/site.json` alike. The
parent may be mid-`npm run build` and would then read a half-written JPEG. Write
to a staged path under `/workspace/.grok/` — never inside `public/`, which
`vite build` copies verbatim into the deployed app, and never on another
filesystem such as `/tmp`, where the hand-over cannot be a rename — then:

```sh
node scripts/write-atomic.mjs /workspace/.grok/og.jpg.tmp public/og.jpg
```

`src/lib/og/site.json` is the only file this pass writes under `src/`. Hand it
over **once**, with the finished card — never a field at a time: its
`"card": "custom"` flag has to land *with* the card and not before, because the
bake trusts the flag on its own and would emit an `og:image` URL for a file that
does not exist.

## 3. Verify your own work, because nobody waits for it

Run the card checks in `custom-card.md`, then `node scripts/brand-check.mjs --game`
(drop `--game` for non-games), which prints a JSON verdict and exits non-zero
on any `BRAND WARNING`. That run judges the files on disk and reports
`"pending": true` — the marker only demotes the parent's gates — and it treats a
missing `public/og.jpg` as a failure whatever kind of app this is, because
producing one is what this pass is normally for.

**The one exception**: a pass launched for a plain utility that keeps the
`og.grok.me` card (favicon, PWA icons, and title only — SKILL.md § "Decide:
which card this app gets") is not there to produce a card, so it adds
`--placeholder-ok` and the missing card is then the expected verdict rather than
a failure:

```sh
node scripts/brand-check.mjs --placeholder-ok
```

Use it only for that launch — passing it on a custom-card app hides the one
thing that pass owes. Report pass or fail in your answer text.
