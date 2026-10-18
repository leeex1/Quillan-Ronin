# Favicon and PWA icons

## Favicon: hand-author `public/favicon.svg`

Every app gets one, and it works in live preview immediately (no host needed).

- **Write the SVG by hand — never `imagine_text_to_image`.** It must stay crisp at 16px:
  one bold glyph or shape, flat fills from the app's design tokens, a square
  `viewBox`, a handful of elements at most. For whimsical apps an emoji-text
  SVG is a fine quick win:

  ```svg
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <rect width="100" height="100" rx="20" fill="#1a1b26"/>
    <text x="50" y="50" font-size="62" text-anchor="middle"
      dominant-baseline="central">🥕</text>
  </svg>
  ```

- Wire it in the root `head()`:

  ```tsx
  links: [
    { rel: "icon", type: "image/svg+xml", href: "/favicon.svg" },
    { rel: "stylesheet", href: appCss },
  ],
  ```

- Verify it renders non-blank and legible small (browser tab in a preview
  screenshot, or read back a rasterized copy). Update it when the app's theme
  or name changes meaning — it is part of the app's identity, not a set-and-forget.

## PWA icons: only for installable apps

When the app ships a web manifest (the user asked for installable / PWA /
home-screen behavior — do **not** invent a manifest just to have icons), add
raster icons derived from the favicon artwork so the identity stays
consistent:

- `public/icon-192.png` and `public/icon-512.png` — the favicon's glyph on
  its tile, rasterized at size. Playwright (baked into the sandbox) can
  screenshot the served `/favicon.svg` at a 192/512 viewport; or redraw the
  same mark as a flat PNG. Keep it bold and flat — no photographic detail.
- A maskable variant (`"purpose": "maskable"`) needs the glyph inside the
  center ~80% safe zone so launcher shapes don't clip it.
- Wire the manifest `icons` array plus `theme_color` / `background_color`
  from the app's design tokens, and read the 192 back to confirm it stays
  legible.
