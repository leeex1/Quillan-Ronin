# X feed card: `public/x-banner.jpg` (50:11)

**Games also ship a second, much wider card** that X uses in the feed (the
link-preview `og.jpg` is still required — this does not replace it). Write
`public/x-banner.jpg` at **exactly 50:11** (1200×264 — 1200 is the standard
max card width on the web, same as `og.jpg`). The injector emits
`x:game:image` (1200×264) from that file when the request has a public host.
Do **not** invent `x:game:og` or overload `og:image`. Do not add
`x:game:image` to `__root.tsx`.

Same JPEG / size discipline as `og.jpg` (keep it well under 600 KB).
**Custom `public/x-banner.jpg` — games only.** Use `imagine_text_to_image`
(or `imagine_image_to_image`) to generate a custom banner **only when the
app is a game**. Non-games omit the file — the injector then emits no
`x:game:image`. Do not paste `og.grok.me/v1/banner.png` (or any
`x:game:image`) into `__root.tsx`; the injector strips those tags.
Generate the art at 50:11 (`aspect_ratio: "50:11"` on
`imagine_text_to_image` — that ratio is on the Imagine schema; do **not**
substitute 16:9 and crop). Then normalize:

```sh
ffmpeg -y -i banner-raw.jpg \
  -vf "scale=1200:264:force_original_aspect_ratio=increase,crop=1200:264" \
  -q:v 4 /workspace/.grok/x-banner.jpg.tmp
node scripts/write-atomic.mjs /workspace/.grok/x-banner.jpg.tmp public/x-banner.jpg
```

**Safe lockup.** Unlike the centered `og.jpg` card, keep title, tagline, and
other critical content inside the **left-most 50% × top-most 80%** of the
finished 1200×264. Feed chrome overlays the **right ~25%** and **bottom
~20%**, so lettering there clashes. Raise the lockup above the midline (do
not vertically center it); comfortable left and top margins; never
edge-hug: a 50:11 frame decapitates edge-hugging lettering even faster than
the 1200×630 card. Scenery and characters may extend into those overlay
strips. Prompt Imagine in visual language — "title lockup in the left half,
sitting above the midline, empty strip along the bottom edge" — not
percentages. Reuse the link-cover art when you can (reframe the same scene
wider rather than inventing a second identity). Verify the 1200×264 JPEG
the same way as `og.jpg` (dimensions + under 600 KB + no clipped title).
Reject if any title or tagline glyph sits in the right half or the bottom
fifth of the frame.
