---
name: video2dsprite
description: >
  Grok Build only. Turn a 2D character still into denser animation sprites via
  imagine_text_to_image base → imagine_image_to_video (6s/10s run-in-place) → ffmpeg
  frames → magenta chroma-key → dense sampled strips/grids/GIFs. Use when the
  user wants video-to-sprite, smoother run/walk cycles, or denser intermediate
  poses. Prefer generate2dsprite for crisp production pixel sheets. Triggers on
  "video to sprite", "imagine_image_to_video sprite", "dense walk cycle", "smooth run
  animation from video".
metadata:
  short-description: "Video→dense sprites (imagine_image_to_video + chroma postprocess)"
user-invocable: false
---

# Video2dsprite (Grok Build only)

Convert a **base 2D character image** into **dense animation sprites** using Grok Build's native video tools.

## App-builder / Grok environment

| Item | Value |
| --- | --- |
| Skill dir | `.grok/skills/video2dsprite/` |
| Scripts | `python3 .grok/skills/video2dsprite/scripts/video2dsprite.py …` |
| Video tools | `imagine_image_to_video` — animate one base `file_path` (verify present) |
| Inspect | `read_file` on stills/frames; report paths for videos |
| Deps | ffmpeg + Pillow + numpy (preinstalled in app-builder image) |
| Output home | `assets/sprites/video2dsprite/<name>/` under `/workspace` |
| Default | Prefer **`generate2dsprite`** for production heroes; this is the denser-motion path |

```text
base still → imagine_image_to_video (in-place motion) → extract frames → chroma key → sample/normalize → strip / grid / GIF
```

## Platform gate (read first)

| Runtime | Supported? |
| --- | --- |
| **Grok Build** (xAI) | **Yes** — requires an image generator + an image→video tool |
| Codex / Claude / other agents | **No** — they lack Grok video tools. Tell the user this skill is Grok Build only and offer `$generate2dsprite` instead |

**Gate on the capability, not the exact tool name.** Image generation appears
as `imagine_text_to_image` / `imagine_image_to_image`; video as `imagine_image_to_video`
or (legacy) `generate_video`. Use whichever pair your tool list has. Stop and
explain only when you have no image→video tool at all. Do not fake motion with
code-drawn frames.

This skill is an **optional denser-motion path**. It does **not** replace `$generate2dsprite`:

| Use `$generate2dsprite` when… | Use `$video2dsprite` when… |
| --- | --- |
| Crisp pixel sheets, fixed grids, identity-critical heroes | User wants denser intermediate poses / smoother feeling loops |
| Attack/cast body sheets, prop packs, engine atlases | Experimenting with video-sourced run/walk/idle motion |
| Production default for most game sprites | User explicitly asks for video → frames → sprites |

Video softens pixels, drifts identity, and leaves chroma fringes. Always QC; for production heroes, prefer `$generate2dsprite` unless the user wants the video look.

## Parameters

Infer from the user request:

- `subject`: character / creature description, or path to existing still
- `action`: `run` | `walk` | `idle` | `attack` | custom motion phrase
- `view`: usually `side` (side-scroller). `topdown` is harder — warn and keep camera locked
- `duration`: `6` (default) or `10` seconds
- `frame_counts`: which denser sets to export, default `8,16,24,48`
- `cell_size`: output sprite cell, default `128`
- `anchor`: `feet` (default for side locomotion) | `center`
- `bg`: solid `#FF00FF` (required for chroma)
- `name`: output slug
- `out_dir`: working folder (default `./sprites/video2dsprite/<name>/` or project-relative)

## Agent rules

1. **Grok-only.** Refuse on non-Grok runtimes with a short explanation + `$generate2dsprite` alternative.
2. **Still → video, never text-to-video alone.** Stage frame 1 as a clean still with `imagine_text_to_image` (from a prompt, or from a reference `file_path`). Then call `imagine_image_to_video` with that still's `file_path`.
3. **In-place motion.** Prompt for run/walk **in place** facing a fixed direction. No camera pan, no background scroll, no scene change. Subject stays roughly centered.
4. **Solid magenta background** on the base and preserved in the video prompt (`#FF00FF` / pure magenta). Required for flood-fill chroma.
5. **Do not invent art with PIL/Canvas.** Base art comes from `imagine_text_to_image` or a user/local still. Scripts only postprocess.
6. **Do not put experimental outputs into the game** unless the user asks to integrate.
7. **Prefer one locomotion cycle for game use.** Dense sample across a full 6s multi-cycle clip is fine for previews; for engine sheets, optionally re-sample a single cycle (12–16 frames) after visual QC.
8. **Report absolute paths** of video, cleaned frames, strips, and preview GIFs when done.

## Workflow

### 1. Plan

Pick the smallest useful run:

- Side-view run/walk loop → this skill
- Multi-action hero kit → still use `$generate2dsprite` per action; only use video for locomotion if requested
- FX / projectile / prop packs → `$generate2dsprite`, not video

Create:

```text
<out_dir>/
  base/
  video/
  frames-raw/
  frames-clean/
  sprite/          # default 8-frame set + denser x16/x24/x48
  prompt-used.txt
  pipeline-meta.json
  README.txt
```

### 2. Build the base still

Options:

- **A. Existing sprite:** read the image, composite onto solid `#FF00FF` if needed, then pass that sandbox path to `imagine_image_to_image` / `imagine_image_to_video`
- **B. New character:** `imagine_text_to_image` with solid magenta background, full body, side view, centered
- **C. Match reference:** `imagine_image_to_image` with the reference `file_path`, moving it onto magenta and preserving identity

Base requirements:

- Full body visible, generous magenta margin
- Side view for run/walk (profile or 3/4 side), feet near bottom third
- Same art style as the rest of the project when a reference exists
- No text, UI, watermark, or second character

Copy the returned `file_path` and save as
`<out_dir>/base/<name>-base.png`; keep the base `file_path` for the video call.

Write the exact image prompt into `prompt-used.txt`.

### 3. Animate with `imagine_image_to_video`

Call Grok `imagine_image_to_video`:

- source path: `<base still file_path>` (sandbox path returned by T2I / I2I)
- `duration`: `6` (default) or `10`
- `resolution_name`: `480p` unless user asks `720p`
- `prompt`: one short present-tense shot (see [references/prompt-rules.md](references/prompt-rules.md))

Mandatory motion constraints in the prompt:

- Subject runs/walks **in place** (treadmill style)
- Camera **locked** — no pan, zoom, or orbit
- Background stays **flat solid magenta**
- Identity, costume, palette stable for the whole shot
- Single continuous action only

Copy the returned video `file_path` to
`<out_dir>/video/<name>-<duration>s.mp4`.

If video tools are unavailable, stop (platform gate).

### 4. Extract + chroma + sample (local script)

Run the processor (ffmpeg + Pillow + numpy):

```bash
python3 .grok/skills/video2dsprite/scripts/video2dsprite.py process \
  --video <out_dir>/video/<name>-6s.mp4 \
  --out-dir <out_dir> \
  --name <name> \
  --frame-counts 8,16,24,48 \
  --cell-size 128 \
  --body-height 100 \
  --foot-y 118 \
  --fps 0
```

Notes:

- `--fps 0` = extract every decoded frame (use source fps)
- Magenta flood-fill from corners + despill
- Even sampling for each count in `--frame-counts`
- Feet-normalized cells, horizontal strip, grid, loop GIF per count

Optional: only re-sample denser sets from existing cleaned frames:

```bash
python3 .grok/skills/video2dsprite/scripts/video2dsprite.py sample \
  --clean-dir <out_dir>/frames-clean \
  --out-dir <out_dir> \
  --frame-counts 16,24,48 \
  --cell-size 128
```

### 5. QC

Visually check:

- [ ] Preview GIF loops without huge pops
- [ ] Magenta gone (no solid pink blocks); fringe acceptable or re-key
- [ ] Feet stay on a stable baseline (no hop from bad crop)
- [ ] Identity roughly stable (face/clothes not morphing every frame)
- [ ] Action is in-place (not sliding out of frame)
- [ ] For game use: pick one count (often **16 or 24**) or cut one true cycle

If identity drifts hard or pixels are too soft, fall back to `$generate2dsprite` for production sheets and keep the video set as motion reference only.

### 6. Deliver

Report paths only (unless user asked to wire into a game):

- Video: `video/*.mp4`
- Dense sprites: `sprite/x16|x24|x48/`
- Strips / grids / GIFs: `sprite/run-strip-N.png`, `run-grid-N.png`, `run-preview-N.gif`
- Meta: `pipeline-meta.json`

Do **not** modify game code unless requested.

## Defaults

- Duration: **6s**
- Action: **side run in place**, facing right
- Export counts: **8, 16, 24, 48**
- Cell: **128²**, body height ~100, feet at y≈118
- Background: **#FF00FF**
- Prefer single-asset `imagine_image_to_video` over multi-ref (compose multi-ref with `imagine_text_to_image` first if needed)

## Tradeoffs (tell the user once)

**Pros:** denser intermediates → often feels smoother than 4–8 discrete gen poses.  
**Cons:** softer pixels, identity drift, chroma fringe, multi-cycle 6s clips are not a single perfect loop, heavier assets.  
**Rule of thumb:** 8→16→24 usually gains smoothness; 48 is often diminishing returns; 145 raw frames are for sampling, not all for runtime.

## Resources

- [references/prompt-rules.md](references/prompt-rules.md) — base still + video prompts
- [references/pipeline.md](references/pipeline.md) — folder layout, ffmpeg, sampling strategy
- [scripts/video2dsprite.py](scripts/video2dsprite.py) — extract, chroma, normalize, export

## Relationship to other skills

- `$generate2dsprite` — primary sheet pipeline (Codex + Grok when image gen exists)
- `$generate2dmap` — maps; not used here
- `$video2dsprite` — **Grok Build exclusive** motion densification path
- `$game-asset-core` / `$game-animation-frames` — loop/flip-test/motion laws and
  engine-ready defaults; still use this skill’s scripts for sandbox execution
  (magenta base + chroma), not a freeform background color
