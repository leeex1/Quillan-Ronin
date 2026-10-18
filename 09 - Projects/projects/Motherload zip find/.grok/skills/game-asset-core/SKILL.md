---
name: game-asset-core
description: >
  Core discipline for ANY game-asset generation with Imagine tools: the
  engine-ready defaults users don't state, spec checklists, style anchoring,
  read-back verification, honest defect flagging. Use whenever generating
  any game art (sprites, sheets, animations, tiles, UI, FX) — then ALSO load
  the matching specialist skill: game-animation-frames for anything that moves,
  game-tilesets for tiles/terrain, game-character-consistency for recurring characters,
  game-ui-icons for UI and icons.
metadata:
  short-description: "Core rules + engine-ready defaults for game assets"
user-invocable: false
---

# Asset Core

Game developers ask for WHAT they need, not HOW to make it engine-ready.
The how is your job. Apply these defaults whenever the request doesn't say
otherwise — an asset that needs manual cleanup is a miss even if the user
never mentioned the requirement.

## App-builder note

This skill is **doctrine / QC**. For 2D sprites and motion in this sandbox, also
run the pipeline skills:

- **`generate2dsprite`** / **`video2dsprite`** for sheets and video harvest
- Keyable background on those paths is solid **`#FF00FF`** (required for chroma)
  — not an arbitrary flat hex
- Motion laws and loop QC live in **`game-animation-frames`** (which defers
  execution to those pipelines)

## Unprompted engine-ready defaults

| When asked for... | Deliver, without being told... |
|---|---|
| a character/creature/prop sprite | isolated subject, flat single-color keyable background (**`#FF00FF`** when using generate2dsprite/video2dsprite), clean silhouette, no baked ground scene or cast shadow |
| anything that moves/animates | a frame SEQUENCE that loops cleanly (see game-animation-frames; execute via video2dsprite/generate2dsprite) |
| a sprite sheet | uniform implicit cells, NO divider lines, subject at the identical position per cell so frames crop at width/cols × height/rows — or build it yourself: frames + PIL composite |
| ground/terrain/water/walls | seamlessly tileable (verify with a real 2×2 composite), no landmark motifs, non-directional lighting where rotation might be used |
| UI panels/frames/buttons | scale-survivable (9-slice: corner ornament, uniform edges), no text ever (games localize), state variants geometry-identical |
| the same character/object again | edit-chained from your existing base image, never regenerated fresh |
| icons | one style contract across the set, uniform padding, legible at 32px |

Deliver organized, exactly-named files; if the request leaves counts or
naming to you, choose sensible names and document them in a manifest/record.

## Working discipline

1. **Spec checklist (private).** List every stated property PLUS the
   applicable defaults above. Verify against it; never paste it into
   prompts.
2. **Prompt in the generator's language.** 2–5 vivid sentences, always with
   style/medium words. Express geometry/quantity as nameable visual
   configurations (clock positions, pie wedges, colored markers), never as
   numbers or abstractions.
3. **Verify by describing blind, then diffing.** Write what the image shows
   before re-reading the spec. Every stated property AND every applicable
   default is pass/fail — no "good enough", no self-negotiated waivers. A
   hedge in your own description = failed check.
4. **Escalate representation, then strategy.** Retry once with a more
   concrete visual re-expression. If the generator repeats the same failure,
   it's a prior: build compositionally (parts + PIL rotate/mirror/assemble —
   mind mirrored asymmetries) or keep the best and FLAG it. ~2 discards max
   per point being proven.
5. **Deliver and report.** Final pass across all files for cohesion and the
   checklist; state every unfixed defect and every default you consciously
   deviated from.
