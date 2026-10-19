---
name: design-ui
description: >
  Design and build polished, non-generic UI for this TanStack Start + React +
  Tailwind v4 + shadcn/Radix app. Use whenever you create or restyle any
  interface surface — pages, landing pages, dashboards, forms, modals, nav, and
  game overlays (start screens, HUD, menus). Covers design tokens, layout,
  typography, color, spacing, motion, and the anti-"AI-slop" rules that keep
  output from looking generic. Triggers on "design", "UI", "make it look good",
  "polish", "landing page", "theme", "style", "redesign", "ugly", "clean up".
metadata:
  short-description: "Polished, non-generic UI: tokens, layout, type, color, motion, anti-slop"
user-invocable: false
---

# Design & UI

Make interfaces that look intentional and premium, not template-generic. This is
the single biggest quality lever in the app builder. Apply it to **DOM / overlay
UI** — pages, chrome, HUD, menus, forms. (For a 3D game's gameplay canvas, see
the `building-games` skill; this skill governs the DOM UI layered over it.)

**Read `references/` for depth** (loaded on demand — don't inline it all):
- `references/refined-ui.md` — the full product-chrome/overlay design system.
- `references/typography.md` — type scale, pairing, rhythm.
- `references/surfaces.md` — elevation, borders, shadows, layering.
- `references/animations.md` — motion, easing, transitions.
- `references/performance.md` — keep UI smooth (60fps, no jank).

---

## 1. Design-system-first (do this before styling anything)

Define the system once, then compose from it. **Never** sprinkle ad-hoc values.

- **Tokens in CSS (Tailwind v4 is CSS-first).** Put the palette, radii, and fonts
  in `src/styles.css` under `@theme` as CSS variables; consume them as Tailwind
  utilities. One source of truth.
  ```css
  @import "tailwindcss";
  @theme {
    --color-bg: #0b0b0f;      --color-surface: #16161d;
    --color-fg: #e7e7ea;      --color-muted: #a0a0ab;
    --color-primary: #14b8a6; --color-border: #26262f;
    --radius: 0.75rem;        --font-sans: "Inter", system-ui, sans-serif;
  }
  ```
- **Use shadcn/ui components** (Radix primitives + `cva` variants + `tailwind-merge`)
  for buttons, dialogs, dropdowns, inputs, etc. They're accessible and consistent.
  Generate them into `src/components/ui`; style via tokens, not inline hex.
- **Tailwind v4 base fix — buttons need a pointer cursor.** v4's Preflight makes
  `<button>` use `cursor: default`, which feels broken. Add this once in
  `src/styles.css` so buttons/clickable roles show a pointer:
  ```css
  @layer base {
    button:not(:disabled),
    [role="button"]:not(:disabled) { cursor: pointer; }
  }
  ```
- **Ban ad-hoc styling:** no raw hex in JSX, no `text-white`/`bg-black` literals,
  no arbitrary values like `p-[16px]` or `text-[13px]`. If you need a value,
  it becomes a token or a scale step.

## 2. The quantified rubric (cheap rules that prevent "ugly")

- **≤ 3–5 colors total** (one primary + neutrals + at most one accent). No random
  extra hues. Don't default to purple unless asked.
- **≤ 2 font families** (often one). Pair a display/heading with a body, or use one.
- **Line-height 1.4–1.6** for body; tighter for large headings.
- **When you override a background color, override the foreground/text color too**
  (contrast must hold — check both light and dark).
- **Mobile-first**: design the ~390px layout first, then scale up. No horizontal
  overflow; tap targets ≥ 44px.
- **Consistent spacing scale** (4/8-based). Generous whitespace beats cramming.
- **One accent, used sparingly** for primary actions — not everywhere.

## 3. Anti-AI-slop (the tells that make output look generic — avoid)

- **No gradient-blob filler**, no giant hero gradients as a substitute for content.
- **No emoji as icons** — use a real icon set (`lucide-react`).
- **No hand-drawn SVG** illustrations/maps/charts — use real libraries (`recharts`
  for charts) or real generated images.
- **No placeholder images / lorem-gray boxes** in the final product — generate
  real images or use real content; set `crossOrigin="anonymous"` on canvas images.
- **Avoid the overused-font look** (default system-only, or Comic Sans-tier picks).
- **Every element earns its place.** Cut decorative noise. Establish a system,
  then vary with intent — not randomness.
- **Match the existing UI when editing** an app in place; don't introduce a second
  visual language.

## 4. Layout & hierarchy

- Clear visual hierarchy: one primary action per view; size/weight/color express
  importance. See `references/typography.md` and `references/surfaces.md`.
- Use real layout structure (grid/flex, container max-widths), not absolute-position
  hacks. Align to a consistent grid.
- Empty states, loading states, and error states are part of the design — don't
  ship blank/janky intermediate states.

## 5. Motion (subtle, purposeful)

- Short, eased transitions (150–250ms) on hover/press/enter; respect
  `prefers-reduced-motion`. Details in `references/animations.md`.
- Never animate layout in a way that causes jank; prefer transform/opacity.

## 6. Game overlays (when this pairs with `building-games`)

The gameplay canvas is owned by `building-games`. This skill styles the **DOM
overlay**: start/"click to play" screen, HUD, score, menus, pause, mobile
controls. Keep overlay readable over the canvas (backdrop, contrast), and keep it
out of the pointer-lock/gameplay input path.

---

## Finish checklist (before you call UI done)
- Tokens defined in `@theme`; no ad-hoc hex / arbitrary values in JSX.
- ≤ 5 colors, ≤ 2 fonts, consistent spacing scale.
- Contrast holds; foreground overridden wherever background is.
- Mobile (~390px) has no overflow; targets ≥ 44px.
- Real icons/images/charts — none of the anti-slop tells.
- Loading/empty/error states handled; motion subtle and reduced-motion-safe.
- Rendered and eyeballed in a browser (see AGENTS.md verification), not just curl.
