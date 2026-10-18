# Refined UI (product chrome and overlays)

Build interfaces that feel **intentional, calm, and expensive** — tight typography, restrained color, concentric radii, and fluid spacing. Prefer systems over one-off styles.

## Scope

**In scope**

- Start / title / menu surfaces layered over a game or app
- Overlay HUD and status chrome (score, health, timers, objectives)
- Pause, settings, win/lose, modals, sheets, toasts, on-screen controls
- Marketing-adjacent panels, cards, nav, footers, and form chrome in Build outputs
- Design tokens, type scales, radius scales, spacing, borders, shadows, motion on those surfaces

**Out of scope**

- Gameplay systems, entities, physics, combat, progression logic
- WebGL / canvas scenes, 3D art, materials, VFX, level design
- Decorative illustration that fights the product chrome

When a task spans gameplay and UI, apply this skill **only** to the overlay / DOM chrome.

## Non-negotiable anti-slop rules

These are hard fails. Fix before polish.

| Ban | Instead |
| --- | --- |
| Emoji in UI chrome, buttons, empty states, headings, or labels | Plain typography, sparse SVG icons (monochrome or single accent), or nothing |
| Purple, violet, magenta, or yellow / gold as brand or accent fills | Near-neutral surfaces; one restrained accent (cool blue-gray, ink, or soft white on dark) |
| Loud multi-stop gradients on backgrounds, buttons, or cards | Flat or near-flat surfaces; at most a **barely** perceptible linear wash (≤8% lightness delta) on large hero fields |
| Rainbow borders, neon glows, glassmorphism soup | Thin neutral borders (`1px`, low-contrast), soft single-layer shadows or hairline dividers |
| Default Inter-everything with no hierarchy | Deliberate display + body pairing and weight steps (see Typography) |
| Identical radius on parent and child | **Concentric** radii (see Border radius) |
| Random spacing and magic numbers | Tokenized spacing scale |
| Random bounce on every control | Short, tokenized motion (see Motion); reserve overshoot for rare micro moments (badge pop only) |

## Visual language

Aim for a **frontier / editorial** product feel:

- Dark or light **near-black / near-white** fields with quiet gray steps — not pastel candy, not neon cyberpunk
- Large, confident headlines with generous tracking control; body copy that breathes
- Abundant negative space; fewer elements with stronger hierarchy
- Surfaces read as **planes and panels**, not stickers
- Motion is subtle and physical (fade + slight translate / scale), never carnival

## Design tokens (encode once, reuse)

Define CSS variables (or equivalent) at the root of the UI layer. Prefer semantic names over raw hues.

```css
:root {
  /* Surfaces — near-neutral, low chroma */
  --bg: #0a0a0b;
  --bg-elevated: #121214;
  --bg-subtle: #1a1a1e;
  --fg: #f4f4f5;
  --fg-muted: #a1a1aa;
  --fg-subtle: #71717a;
  --border: color-mix(in oklab, var(--fg) 12%, transparent);
  --border-strong: color-mix(in oklab, var(--fg) 22%, transparent);

  /* Single restrained accent — cool, not purple/yellow */
  --accent: #c8ccd4;
  --accent-fg: #0a0a0b;

  /* Radius scale (px) — use concentrically */
  --radius-xs: 4px;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-xl: 24px;

  /* Spacing scale */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --space-8: 64px;

  /* Type */
  --font-display: "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-body: "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;

  --text-xs: clamp(0.75rem, 0.7rem + 0.2vw, 0.8125rem);
  --text-sm: clamp(0.875rem, 0.82rem + 0.25vw, 0.9375rem);
  --text-base: clamp(1rem, 0.95rem + 0.25vw, 1.0625rem);
  --text-lg: clamp(1.125rem, 1rem + 0.5vw, 1.25rem);
  --text-xl: clamp(1.5rem, 1.2rem + 1vw, 2rem);
  --text-2xl: clamp(2rem, 1.4rem + 2vw, 3rem);
  --text-3xl: clamp(2.5rem, 1.6rem + 3vw, 4rem);

  --leading-tight: 1.1;
  --leading-snug: 1.25;
  --leading-normal: 1.5;

  --tracking-tight: -0.02em;
  --tracking-display: -0.03em;

  /* Motion — durations */
  --motion-stagger: 40ms;
  --motion-micro: 80ms;
  --motion-quick: 150ms;
  --motion-fast: 250ms;
  --motion-medium: 350ms;
  --motion-slow: 400ms;
  --motion-emphasis: 500ms;

  /* Motion — easings */
  --ease-smooth-out: cubic-bezier(0.22, 1, 0.36, 1);
  --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
  --ease-in-out: ease-in-out;
  --ease-linear: linear;
  --ease-pop: cubic-bezier(0.34, 1.36, 0.64, 1);

  /* Motion — distances / scales / blur */
  --motion-y-micro: 4px;
  --motion-y-small: 6px;
  --motion-y-base: 8px;
  --motion-y-medium: 12px;
  --motion-scale-modal: 0.96;
  --motion-scale-menu: 0.97;
  --motion-scale-tooltip: 0.98;
  --motion-blur-sm: 2px;
  --motion-blur-md: 3px;
}
```

Light theme mirrors the same structure with inverted neutrals (warm-neutral off-whites, ink text). Keep chroma low.

**Rules**

- Never invent one-off hex in components if a token exists
- Borders use translucent mixes against `--fg` so they adapt on dark/light
- Accent is for **primary actions and focus**, not decorative fills on every card

## Color discipline

1. **Neutrals first.** 90%+ of UI area is background / elevated / text / border neutrals.
2. **One accent family.** Cool gray-blue or pure light-on-dark — never purple, violet, magenta, yellow, gold, or orange for brand chrome unless the user explicitly demands a brand palette.
3. **No emoji color.** Icons are stroke/fill monochrome or currentColor.
4. **Gradients only if nearly invisible** — e.g. vertical wash from `#0a0a0b` to `#101014`. No aurora, mesh, or “AI purple” backgrounds.
5. **Contrast.** Body text meets WCAG AA against its surface. Muted text is for secondary labels only.
6. **Semantic status** (success/warn/danger) may use restrained green / amber / red **only** on small badges/icons — not full panels.

## Border radius (concentric, mandatory)

Outer radius must equal **inner radius + padding** on that axis so curves nest optically.

```text
outerRadius = innerRadius + padding
```

Example: card with `padding: 16px` and inner control radius `8px` → card radius `24px` (`--radius-xl` if tokens map that way).

| Situation | Approach |
| --- | --- |
| Card containing buttons / inputs | Card uses larger radius; children use smaller from the scale |
| Nested panels | Each nesting level steps **down** one radius token |
| Pill controls | Full pill (`9999px`) only for small chips/buttons — not for large cards |
| Modals / sheets | `--radius-lg` or `--radius-xl` on the shell; inner sections `--radius-sm` / `--radius-md` |

Never set the same radius on parent and padded child.

## Typography pairing and weight

**Pairing**

- Prefer a **single family** with strong weight contrast for cohesion (system UI stacks are fine), **or** a restrained display + body pairing where display is used only for hero / title treatments
- Mono only for code, stats, and tabular IDs — not body paragraphs

**Weights (typical scale)**

| Role | Weight | Notes |
| --- | --- | --- |
| Hero / display | 500–600 | Slightly tight tracking (`--tracking-display`) |
| Section titles | 500–600 | Snug leading |
| Body | 400 | Normal leading |
| Labels / meta | 500 | Smaller size, muted color |
| Buttons | 500–600 | Never ultra-black 900 on large type |

**Avoid**

- All-caps long sentences
- Pure black (`#000`) on pure white for large body blocks on marketing surfaces — prefer soft ink / soft paper
- More than **three** effective sizes on one screen chrome (excluding micro legal)

**Fluid type**

- Use `clamp()` tokens above so titles scale with viewport without jumping breakpoints
- Pair fluid type with fluid spacing (`clamp` or viewport-linked gaps) so rhythm stays consistent

## Layout and fluidness

- Prefer **fluid grids** (`minmax`, `auto-fit` / `auto-fill`) over rigid 12-column everything
- Max content width for readable text (~60–75ch); hero type can break wider
- Consistent vertical rhythm from the spacing scale (stack with `--space-4` / `--space-5`, not 13px / 27px)
- Align to edges and columns; avoid “almost aligned” optical noise
- Safe areas on overlays: respect notches with `env(safe-area-inset-*)`
- HUD floats in corners/edges with padding from `--space-4`+; don’t crowd the center except for intentional modals

## Surfaces, borders, depth

- Prefer **hairline borders** and slight elevation over heavy drop shadows
- One shadow recipe max, low opacity, large blur, minimal y-offset
- Dividers: `1px` using `--border`
- Panels on dark: elevated surface one step lighter than page bg, not a bright gray slab
- Avoid frosted blur stacks that obscure gameplay under overlays — if blurred, keep backdrop subtle and content readable

## Components (product chrome)

**Default to [shadcn/ui](https://ui.shadcn.com/).** Prefer official components from the registry over hand-rolled buttons, inputs, dialogs, menus, sheets, cards, tabs, badges, tooltips, selects, and form chrome. Docs and catalog: https://ui.shadcn.com/ and https://ui.shadcn.com/docs/components.

### Setup (when the project can use React + Tailwind)

```bash
npx shadcn@latest init -d --base radix
npx shadcn@latest add button card dialog input label select sheet tabs badge tooltip dropdown-menu separator skeleton alert-dialog
```

- Use non-interactive flags (`-d` / `--defaults`) so agents never block on prompts
- Prefer **new-york** style and theme tokens (`bg-background`, `text-foreground`, `border-border`, `text-muted-foreground`, `bg-card`, `ring-ring`) over ad-hoc hex
- Theme variables may be mapped to the token block above; do not invent a second parallel component kit
- Own the copied source under `components/ui` and restyle with tokens/anti-slop rules — still use the shadcn primitives and composition patterns

### Reach for shadcn first

| Need | Prefer (shadcn) | Notes |
| --- | --- | --- |
| Primary / secondary actions | `Button` | Variants for hierarchy; optional subtle `scale(0.98)` on press |
| Text fields | `Input` / `Textarea` + `Label` | Consistent height (~40–44px), focus ring via theme |
| Confirm / pause / win-lose | `Dialog` or `AlertDialog` | Destructive confirms use `AlertDialog` |
| Settings / side panels | `Sheet` | Edge panels over custom drawers when possible |
| Overflow menus | `DropdownMenu` / `Popover` | Origin-aware from trigger |
| Grouped content | `Card` + `Separator` | Concentric radii with inner controls |
| Status / filters | `Badge` / `Tabs` | Quiet variants; no emoji |
| Hover hints | `Tooltip` | Delayed first show; instant siblings when appropriate |
| Loading placeholders | `Skeleton` | Then cross-fade to content |

Do **not** ship raw `<button>` / `<input>` / ad-hoc `div rounded-xl border p-6` shells when a shadcn component covers the need.

### When shadcn is not available

Plain HTML/CSS overlays (or non-React stacks) may implement the same patterns with tokens below. Match shadcn structure (button variants, dialog/sheet roles, card hierarchy) so a later migration is trivial.

### Buttons (token overlay / fallback)

- Primary: solid near-white or accent-on-dark, dark label; clear hover/active (opacity or slight brightness), `scale(0.98)` on active optional
- Secondary: transparent or elevated with `--border`
- No gradient fills, no emoji, no purple glow

### Inputs (token overlay / fallback)

- Height consistent (e.g. 40–44px touch-friendly)
- Focus ring: 2px subtle accent or light ring, not rainbow
- Placeholder uses `--fg-subtle`

### Cards / menus (token overlay / fallback)

- Concentric radii; padding from spacing scale
- Title + muted description hierarchy
- Actions right-aligned or full-width stacked on small screens

### HUD chips

- Compact, high contrast, tabular numbers (`font-variant-numeric: tabular-nums`)
- Prefer `Badge` when on a shadcn stack; otherwise minimal chrome — no cartoon badges

## Motion and fluidity (production transitions)

Chrome motion should feel like a **small catalog of tuned recipes**, not ad-hoc timings. Prefer **CSS transitions** (interruptible) over keyframes unless the effect is a one-shot staged sequence (success check, error shake). Animate **inner pieces**, not a single wrapping box, when content swaps.

### Universal rules

- Enumerate properties — never `transition: all`
- Prefer `opacity` + `transform` (+ optional light `filter: blur` on crossfades)
- **Enter slower / exit quicker** when asymmetric (modal/menu close uses `--motion-quick`, open uses `--motion-fast`)
- Never enter from `scale(0)` — use `--motion-scale-modal` … `--motion-scale-tooltip` range
- Origin-aware surfaces (menus, popovers) scale from the **trigger**; centered modals stay center-origin
- Every recipe needs `@media (prefers-reduced-motion: reduce)` (opacity-only or instant)
- Keep durations on **CSS variables** so JS orchestration can `getComputedStyle` and stay in sync
- Replay animations with a reflow (`void el.offsetWidth`) between class remove and re-add when needed
- On close, use a closing class for the exit transition, then remove it after timeout — otherwise the next open jumps from the wrong scale

### Motion token usage (match by role, not by copying random ms)

| Role | Duration token | Easing | Notes |
| --- | --- | --- | --- |
| Per-item stagger | `--motion-stagger` (40ms) | — | Stacked text / list enter |
| Tooltip delay / shake segment | `--motion-micro` (80ms) | — | Small offsets |
| Menu/modal/tooltip **close**, in-place text swap | `--motion-quick` (150ms) | `--ease-smooth-out` or `--ease-out` | Snappy dismiss |
| Menu/modal **open**, icon swap, sliding tabs, page slide | `--motion-fast` (250ms) | `--ease-smooth-out` / `--ease-in-out` | Primary opens |
| Panel dismiss | `--motion-medium` (350ms) | `--ease-smooth-out` | Slightly heavier surfaces |
| Panel open, skeleton → content, clear dissolve | `--motion-slow` (400ms) | `--ease-smooth-out` / `--ease-in-out` | Contentful reveals |
| Badge appear, hero line reveal, rare success flourish | `--motion-emphasis` (500ms) | smooth-out or `--ease-pop` for badge only | Use sparingly on overlays |

**Blur on crossfade** (when two states overlap oddly): `--motion-blur-sm` (2px) for icon/text/number swaps; `--motion-blur-md` (3px) for page/panel slides. Keep blur off for simple opacity fades.

### Pattern → recipe (pick by what the user sees)

Match the **visible element + verb**, then implement with tokens above:

| If you see… | Use this transition recipe |
| --- | --- |
| Trigger + surface growing from it (settings, overflow) | **Origin-aware menu** — open `scale(--motion-scale-menu→1)` + opacity, `--motion-fast` / `--ease-smooth-out`; close faster (`--motion-quick`) with slightly higher scale floor (~0.99) |
| Centered pause / win / confirm dialog | **Modal** — open from `--motion-scale-modal`, `--motion-fast`; close `--motion-quick`; backdrop fades with opacity only |
| Sheet / drawer into an edge of the overlay | **Panel reveal** — translate on axis + optional `--motion-blur-sm` crossfade; open `--motion-slow`, close `--motion-medium` |
| List ↔ detail or step 1 ↔ step 2 in overlay | **Side-by-side page** — opposing translateX(`--motion-y-base`) + light blur; `--motion-fast` |
| Card / HUD cluster changing width or height | **Layout resize** — transition `width`/`height` or grid tracks with `--ease-smooth-out` and `--motion-fast` (avoid animating padding if possible) |
| Score / timer / counter updating | **Number pop-in** — per-digit re-enter with `--motion-y-micro` + `--motion-blur-sm`; stagger `--motion-stagger`; `tabular-nums` |
| Label / status text changing in one slot | **Text state swap** — outgoing slides/fades one way, incoming the other, light blur; `--motion-quick` |
| Two icons in one control (play/pause, mute) | **Icon swap** — cross-fade + scale + `--motion-blur-sm`; `--motion-fast` / `--ease-in-out`; keep both in DOM during swap |
| Small notification dot on a control | **Badge** — short diagonal/offset enter; optional `--ease-pop` once — not on every hover |
| Horizontal chips / avatar stack hover | **Distance falloff lift** — neighbors lift less; bouncy ease **only on mouseleave return**, set timing in JS before writing transform vars |
| Invalid field / failed action | **Error shake** — short segmented translateX with cubic segments; separate “error styling” class from “shake” class so shake can replay |
| Search/filter clear | **Clear dissolve** — control exits with motion; optional per-word streak; `--motion-slow` |
| Placeholder → loaded overlay content | **Skeleton reveal** — pulse with linear, then cross-fade + light blur to real content; `--motion-slow` |
| Loading / “thinking” status line | **Shimmer text** — masked highlight sweep, **linear**, loop; pure CSS; muted foreground |
| Segmented control / filter tabs | **Sliding pill** — move highlight with `transform` + width; first paint and resize with `transition: none` then restore |
| Hover hint on icon/control | **Tooltip** — delayed fade+scale in (`--motion-scale-tooltip`, `--motion-quick` / `--ease-out`); **instant or near-instant out**; subsequent siblings can skip delay (`data-instant`) |
| Title + subtitle entering a start screen | **Staggered text reveal** — blurred rise `--motion-y-medium`, stagger `--motion-stagger`–`--motion-micro`; quiet fade on exit |

**Tie-breakers:** prefer lower overhead (resize over panel, menu over modal, icon swap + checkmark over a full celebration modal). Don’t stack three recipes on one interaction.

### Overlay-specific guidance

- Pause menus open often — favor **modal/menu** recipes with **quick close**, not multi-second cinematic intros
- HUD counters use **number pop-in**, not full card animations every tick
- Start-screen copy uses **staggered text reveal** once; respect reduced motion
- Never animate the WebGL canvas with these recipes — only DOM overlay nodes

### Anti-patterns

| Avoid | Prefer |
| --- | --- |
| `transition: all` | Explicit `opacity, transform` (and blur only when needed) |
| One duration for open and close | Asymmetric open/close tokens |
| Keyframes for hover toggles | CSS transitions (interruptible) |
| Animating the outer page wrapper for a badge | Animate the badge/dot only |
| Hardcoded ms in JS timeouts that drift from CSS | Read `--motion-*` from computed styles |
| Success stroke `stroke-dasharray` guesses | `path.getTotalLength()` (+1) for your path |
| Mixing error styling + shake in one class | Orthogonal classes so shake can reflow-replay |

## Content tone in the UI

- Labels are **short, plain language**
- No emoji, no “✨ magic” marketing fluff in chrome
- Prefer verbs for actions: Continue, Resume, Settings, Quit

## Implementation checklist

Before calling UI done:

- [ ] No emoji in any chrome string or icon slot
- [ ] No purple / yellow / gold accent system
- [ ] No loud gradients on backgrounds or buttons
- [ ] Components default to shadcn/ui (https://ui.shadcn.com/) when React+Tailwind is available
- [ ] Tokens used for color, space, radius, type, motion
- [ ] Nested radii obey `outer = inner + padding`
- [ ] Clear type hierarchy (size + weight + color), ≤3 primary sizes
- [ ] Fluid type/spacing where layouts scale
- [ ] Tabular nums on live stats
- [ ] Focus visible; contrast AA for text
- [ ] Motion uses token durations/easings and a matching recipe (menu, modal, number pop-in, etc.)
- [ ] Open/close asymmetric where appropriate; no `transition: all`; reduced-motion guarded
- [ ] Gameplay / canvas untouched by these styles

## Review format

When reviewing UI against this skill, use a markdown table:

| Before | After | Why |
| --- | --- | --- |
| Purple gradient CTA | Flat near-white primary on dark | Removes slop accent; quieter hierarchy |
| Card and button both `12px` radius with 16px pad | Card `28px`, button `12px` | Concentric radii |
| Title with emoji | Title plain, weight 600 | No emoji in chrome |
| `transition: all 300ms` on menu | `opacity/transform` with `--motion-fast` open / `--motion-quick` close, `--ease-smooth-out` | Tokenized, asymmetric, interruptible |
| Score textContent swap with no motion | Per-digit pop-in, `--motion-y-micro`, tabular-nums | Matches counter recipe |
| Modal scales from `0` | From `--motion-scale-modal` (0.96) | Avoid zero-scale pop-in |
