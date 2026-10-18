---
name: game-animation-frames
description: >
  Deep guide for game ANIMATION assets: motion cycles, action keyframes,
  effect sequences, and animation sprite sheets — built around a
  video-first pipeline. In this app-builder sandbox, execute via the
  video2dsprite / generate2dsprite skills (magenta + scripts), not ad-hoc
  ffmpeg. Use whenever generating anything that moves: walk/run cycles,
  attacks, idles, FX, flags, fire, animation sheets. Complements
  game-asset-core.
metadata:
  short-description: "Video-first animation frames that actually cycle"
user-invocable: false
---

# Animation Frames — video-first

The image generator draws poses; the VIDEO generator understands motion —
leg alternation, arc continuity, cloth and fire dynamics come free because
video must animate them. So don't ask the image model to imagine
mid-motion poses: animate the base and harvest real frames.

## App-builder execution (read first)

This skill is **doctrine** (motion laws, loop QC, when to use video). In the
app-builder sandbox, **do not** run a freeform `ffmpeg` harvest or invent a
random keyable `#hex` background.

| Step | Do this |
| --- | --- |
| Production sprites / fixed grids | **`generate2dsprite`** — solid **`#FF00FF`** magenta sheets + chroma scripts |
| Denser locomotion from video | **`video2dsprite`** — base still on **`#FF00FF`** → `imagine_image_to_video` → skill scripts (ffmpeg + chroma) |
| Keyable background | Always **`#FF00FF`** when using either pipeline (required for chroma) |
| This skill | Loop / flip-test / motion laws below — apply after the pipeline runs |

Open `.grok/skills/video2dsprite/SKILL.md` or `.grok/skills/generate2dsprite/SKILL.md`
and follow their workflows for generation and postprocess. Then apply the
laws and flip test here before shipping frames into the game.

## Default pipeline (intent)

1. **Base frame.** Subject in neutral/starting pose, full style words, side /
   game-appropriate view, **solid `#FF00FF` background** (app-builder chroma
   key). game-asset-core defaults apply.
2. **Animate.** `imagine_image_to_video` from the base: one clear motion, in place,
   static camera ("the knight walks in place, side view, camera locked",
   6s). Keep the shot simple — one subject, one motion. Prefer running this
   through **`video2dsprite`** so harvest + chroma are consistent.
3. **Harvest + clean.** Use **`video2dsprite`** scripts (not ad-hoc
   `ffmpeg -i … fps=12` alone). Magenta flood-fill / despill lives there;
   do not re-key with a different flat `#hex` unless you leave the magenta
   pipeline entirely.
4. **Select.** Pick frames that (a) capture the motion's distinct phases and
   (b) LOOP — the sequence's end must flow back into its start. Don't force
   a count: if the motion reads best with 8, 10, or 12 frames, deliver that
   many (more frames = smoother in-engine). For a cycle, select one full
   period using motion landmarks (foot contacts, wing extremes, flame peaks).
5. **Package.** Deliver frames in play order (zero-padded names) and/or a
   sheet per game-asset-core rules (uniform cells, no dividers) — or the
   transparent strips/grids emitted by **`video2dsprite`** /
   **`generate2dsprite`**. State the intended fps.

Fall back to keyframe-by-keyframe `imagine_text_to_image` (still on `#FF00FF` when
postprocessing with the sprite scripts) only when video fails the motion
(rare: very stylized poses, single dramatic keyframes) — and then plan
phases yourself and obey the laws below. Prefer **`generate2dsprite`** for
crisp production multi-frame grids.

## Motion laws (verify against these, whatever the pipeline)

- Cycles loop; alternating gaits spend half the period mirrored.
- Continuity: limbs, props, anatomy, effects move on continuous paths —
  nothing teleports, vanishes, or duplicates between adjacent frames.
- Physics reads in stills: airborne shows air, anticipation compresses,
  follow-through overshoots; effects stay anchored to their origin unless
  the request moves them.
- Energy matches the ask: idle/subtle means barely-different frames.

## Verify — the flip test

View the final frames strictly in order and narrate the motion; check
loop closure explicitly (last→first). A hedge in your narration is a
failed frame. The video pipeline usually passes this on the first try —
that's why it's the default.
