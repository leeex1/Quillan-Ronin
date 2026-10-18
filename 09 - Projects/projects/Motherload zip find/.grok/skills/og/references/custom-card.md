# Custom card: generate `public/og.jpg`

For prompt-craft, composition, and blind read-back verification, follow the
`imagine` and `game-asset-core` skills — this file owns the **card-specific**
contract only (size, lockup, wiring).

1. **Set the canvas with `aspect_ratio: "16:9"`.** The call looks like
   `{ "prompt": "…", "aspect_ratio": "16:9" }` — ratio words in the prompt do
   **not** set the canvas. At 2mp, 16:9 renders 1792×1008, so the normalize
   below cover-crops to 1200×630 trimming only ~3% vertically, which a
   centered title survives. A narrower canvas is what kills titles: from 3:2
   the same crop takes **~21% vertically**, straight through the lockup.
   Paths:

   - **Default — one call with `aspect_ratio: "16:9"`:** `imagine_text_to_image`
     with the art + baked title. **Check the output dimensions** via
     `read_file` / Pillow on the returned `file_path`; if the ratio missed, reframe or use
     the API path.
   - **Reframe if needed:** pass the prior `file_path` into `imagine_image_to_image`
     with **`aspect_ratio: "16:9"`** and a prompt like "extend the scenery
     left and right into a wider frame; keep the title lettering and
     central subject exactly as they are".
   - **Optional — true 2:1 via the xAI Images API:** `POST
     https://api.x.ai/v1/images/generations` with `"aspect_ratio": "2:1"`
     and `response_format: "b64_json"` using the injected `XAI_API_KEY`
     (see the `xai-api` skill). 2mp 2:1 is 1984×992; normalize then trims
     only ~2.4% per side and nothing vertical.

   Build the prompt from the app's theme, palette, and characters. If the
   app already has a key generated asset (hero sprite, title scene), pass
   its `file_path` into `imagine_image_to_image` so the card matches in-game art —
   same 16:9 + check-the-output rule applies.
   **Last resort only** (no `imagine_text_to_image` and no xAI Images API): stay
   on whatever canvas you have and keep the entire title block inside the
   **middle half** of the frame height, with the crop-clipping check in
   step 6 as the gate.
2. **Bake the title in like a game cover.** Store-page covers (Stardew Valley,
   Cuphead) lead with a short stylized logo-type title. Put the exact app name
   in quotes in the prompt; 1–3 strong words; optional short tagline under the
   title in smaller lettering (exact words in quotes).
   - **Stack multi-word titles** into a two-line lockup ("SKY" over "STRIKE").
   - **Center the block both ways** with generous margins — avoid "upper third"
     / percentage placement (models hug the edge).
   - **Bound the width**: lettering spans roughly half to two-thirds of the
     frame, never border to border.
   - **Keep comfortable margins anyway.** From a 16:9 canvas the normalize
     below trims only ~3% vertically; from a 2:1 API canvas it trims
     nothing vertical and ~2.4% per side. The crop turns destructive when
     the canvas comes back off-ratio — an edit that pinned to its input's
     ratio, or a model miss — so **check the raw canvas dimensions before
     cropping**, and re-ratio first if it isn't ~16:9 or 2:1.
3. **Verify glyphs *and* layout on read-back** (see `imagine` / `game-asset-core`
   for the blind-describe loop). On a garble or layout miss, **regenerate with a
   corrected prompt** — never try to move a logo with `imagine_image_to_image` (frame
   translation / seams). After two failed attempts, ship the card **artwork-only**
   (titleless).

   **Intentional exception vs `imagine`'s "rebuild text with code" rule:** the
   share card is a single static PNG; there is no reliable in-sandbox path to
   composite crisp code-drawn lettering onto generative art for this asset, so
   a clean titleless card is the correct fallback after two glyph failures.
4. **Normalize to exactly 1200×630 JPEG** with the baked-in ffmpeg (cover-crop —
   from 16:9 this shaves ~3% top/bottom; from 2:1 ~2.4% per side and
   nothing vertical). **JPEG, not PNG**: the card is
   photographic generative art, and a PNG of it lands at 1–2 MB — heavy
   enough that link scrapers (X card previews included) time out or skip the
   image, so the card silently fails to unfurl. JPEG at this quality is
   ~150–300 KB with no visible loss at unfurl size:

   ```sh
   ffmpeg -y -i card-raw.jpg \
     -vf "scale=1200:630:force_original_aspect_ratio=increase,crop=1200:630" \
     -q:v 4 /workspace/.grok/og.jpg.tmp
   node scripts/write-atomic.mjs /workspace/.grok/og.jpg.tmp public/og.jpg
   ```

5. **Tell the injector the card is custom** — set `"card": "custom"` in
   `src/lib/og/site.json`, handed over the same way, and keep `public/og.jpg`:

   ```sh
   node scripts/write-atomic.mjs /workspace/.grok/site.json.tmp src/lib/og/site.json
   ```

   Bake also infers custom
   from the file if the flag is missing, but brand-check still requires the
   field. The injector emits the absolute `https://${host}/og.jpg` URL. Do not
   add `og:image` to `__root.tsx`.
6. **Verify before finishing** (Pillow is installed; `ffprobe` is **not**):

   ```sh
   python3 -c "
   from PIL import Image; import os
   im = Image.open('public/og.jpg')
   kb = os.path.getsize('public/og.jpg') // 1024
   print(im.size, f'{kb} KB')"
   # expect: (1200, 630) and under 600 KB (keeps X and other scrapers
   # reliable; target <= 300 KB — if over, bump -q:v up a step and re-encode)
   ```

   **Read back the final `public/og.jpg`, not the pre-crop raw** — the crop
   is where clipping happens. This is a **hard gate, not an impression**:
   if any title glyph touches a frame edge or is visibly cut, the card is
   **rejected** — do not ship it, whatever else is right about it. Fix the
   ratio (step 1) or regenerate with the width bound restated; a shipped
   decapitated title is worse than the placeholder. Then confirm the card
   reads like *this* app at thumbnail size — clear subject, correctly
   spelled title (if any), comfortable margins.
