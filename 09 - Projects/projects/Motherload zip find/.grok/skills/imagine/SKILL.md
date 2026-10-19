---
name: imagine
description: >
  How to use the Imagine tools in Grok Build: imagine_text_to_image,
  imagine_image_to_image, imagine_reference_to_image, imagine_text_to_video,
  imagine_image_to_video, imagine_reference_to_video, and render_file for chat
  previews. When to build a visual with code instead of generating it,
  prompt-craft, reference-first handling of real people, factual grounding, and
  asset-consistency. Load this whenever generating or editing an image or video
  is on the table. Tool-usage-driven, not triggered by a user merely mentioning
  images.
metadata:
  short-description: "Prompting and workflow guidance for Imagine image/video tools"
user-invocable: false
---

# Imagine

Grok Build uses the **split Imagine computer stack** (`grok_computer` variants).
There is **no** consolidated `imagine_image` or `imagine_video` tool — always
call the modality-specific name from the table below.

| Tool | Role |
|------|------|
| `imagine_text_to_image` | New image from a text prompt only (no source). |
| `imagine_image_to_image` | Edit / restyle **one** existing image (sandbox path). |
| `imagine_reference_to_image` | Combine **2+** reference images (sandbox paths). |
| `imagine_text_to_video` | New video from a text prompt only (no source). |
| `imagine_image_to_video` | Animate **one** still (sandbox path) into a video. |
| `imagine_reference_to_video` | Video from **1+** reference images (sandbox paths). |
| `render_file` | Show a sandbox image/video path to the user in chat. |

**Path-based handles.** These tools read/write the shared sandbox:

- Generation returns a sandbox **`file_path`** (under artifacts). Open it with
  `read_file` / shell; show the user with **`render_file`**.
- Edit / animate tools take that path (or paths) as input — match the live
  schema (`image`, `images`, etc.). Never invent paths.
- Never call `imagine_image` / `imagine_video`. Asset-id helpers
  (`imagine_create_asset` / `imagine_view_media` / `render_imagine_media`) are
  **not** on this stack.

Apply this whenever you're considering or about to call any of these tools.

## Handle flow (mandatory mental model)

```text
generate → file_path → render_file (show user) / read_file or scripts (QC)
edit / animate → pass prior file_path(s) into image_to_* / reference_to_*
```

- **Use the path the tool actually returned** — do not invent filesystem paths.
- **Show the user media** with `render_file`, not ad-hoc markdown image links.

## Build accurate visuals with code, not the image tools

1. **Image models are unreliable at exact text, numbers, and structure.** They can handle short text or a simple layout, but they often garble words, invent numbers, draw chart bars that match no data, or point diagram arrows nowhere, and the more that has to be exact, the worse they do. A detailed prompt doesn't make it dependable, and another `imagine_image_to_image` edit usually won't fix it. So when a result needs specific text, data, or structure to be correct (charts from real numbers, labeled or technical diagrams, math explainers, tables, screens with real copy), construct the asset with code, where you control the exact content. Prefer HTML and CSS, which give much better layout, typography, and polish than Python plotting. When only the look matters (photos, illustrations, characters, scenes, decorative art), the image tools are the right choice. Which one fits depends on what the output needs to get right, not on how the request is worded.

## Verifying discrete accuracy (loop)

When the output must get specific text, numbers, data, or structure right, don't trust the first result - verify it in a loop:

1. Produce the result (generate, or per *Build accurate visuals with code*, construct it in code).
2. Inspect the actual output - use `read_file` (image understanding) on the result path - and confirm every word, number, label, and structural detail matches the requirement, and that nothing overlaps, clips, or runs off-canvas.
3. If anything is wrong, fix and re-verify:
   - Garbled text, invented numbers, or broken layout from an image model? Don't just re-prompt - it will likely garble it again. Rebuild it with code.
   - Overlapping or clipped elements in code-built output? Re-lay-out with auto-layout (HTML/CSS) rather than nudging coordinates by hand.
   - Otherwise make one targeted edit via `imagine_image_to_image` with the prior path.
4. Only finish when the discrete content is exactly correct. If it can't be made accurate, tell the user instead of shipping something wrong.

## Core Principles

1. **You own the prompt.** If the user gives a detailed prompt or asks you to use theirs, use it verbatim. Otherwise craft the final prompt: front-load the subject, give strong high-level direction for mood, composition, lighting, and style without over-specifying every detail, write natural prose rather than keyword tags, and describe positively instead of using negative prompts. For edits, describe only what changes. Target 2-5 sentences.
2. **Reference-first for real people.** Never use pure text-to-image for a named real person or group, including face swaps, posters, cartoons, and cinematic or editorial depictions. Use `imagine_image_to_image` **with a real reference path** instead, and never produce non-consensual, sexualized, or minor-involving likenesses. See Real People and References for the procedure.
3. **Ground facts with search first.** If any part of the request depends on a real-world fact, identity, brand or product, place, event, or top/latest/current result, search the web before generating and put the actual verified details into the prompt. Don't rely on memory, and don't write vague placeholders like "the current president"; write the verified name.
4. **Reuse a base for consistency.** When the same character, object, or setting must appear across multiple images, generate one base with `imagine_text_to_image`, keep its `file_path`, then pass that path to `imagine_image_to_image` for every variation. Don't re-run text-to-image from scratch for a recurring subject.
5. **Handle failures gracefully.** On a moderation or safety block, stop; don't retry and don't paraphrase the prompt to evade the filter. Tell the user it was blocked and offer a different direction. If a reference is weak or a result looks off-target, say so and ask for an upload or redirect rather than silently iterating.
6. **Plan multi-step workflows.** Sequence the steps; only parallelize generations that belong to the same step.
7. **Review at the end.** Confirm the generations you intended actually executed and match what was asked. Render final assets with `render_file`.
8. **Don't assume tool behavior.** Don't invent tool parameters, return values, or environment capabilities that aren't actually provided; verify rather than guess.

## Choosing the tool

| Situation | Call |
|-----------|------|
| New image, no source | `imagine_text_to_image` with `prompt` (+ `aspect_ratio`) |
| Edit / restyle / recolor one existing image | `imagine_image_to_image` with `prompt` + source path |
| Combine 2+ reference images into one | `imagine_reference_to_image` with `prompt` + source paths |
| Iterate on a previous result | `imagine_image_to_image` with prior path |
| Named real person or group | `imagine_image_to_image` with a real reference path after web search |
| Generic / invented subject from scratch | `imagine_text_to_image` |
| New video, no source | `imagine_text_to_video` with `prompt` |
| Animate one still | `imagine_image_to_video` with that still's path |
| Multi-ref video | `imagine_reference_to_video` with image path(s) |

Rule of thumb: **no refs → text_to_*; one ref → image_to_*; 2+ refs → reference_to_*.**

## `imagine_text_to_image`

Generate a new image from a text prompt.

Inputs:

- `prompt` (required) - full description of the desired image.
- `aspect_ratio` - one of `1:1`, `3:4`, `4:3`, `2:3`, `3:2`, `9:16`, `16:9`, `21:9`, `5:2`, `50:11`, or `unknown`. Use `16:9` for OG share cards and `50:11` for the X feed banner when generating a custom `public/x-banner.jpg` for **games** (every app wires `x:game:image`; only games call Imagine for it — see the `og` skill); for a true 2:1 canvas, call the xAI Images API.

To produce multiple variations, make multiple `imagine_text_to_image` calls with distinct prompts. The tool does not expose `n` or `count` parameters.

## `imagine_image_to_image`

Edit one existing image.

Inputs (names follow the live schema — typically a singular sandbox path):

- `prompt` (required) - what to change (describe only the edit).
- Source path field (required) - sandbox path returned by a prior generation or download.
- `aspect_ratio` - only set when the user explicitly wants a ratio change.

## `imagine_reference_to_image`

Compose one image from 2+ reference paths.

Inputs:

- `prompt` (required).
- Source paths (required) - 2+ sandbox paths.
- `aspect_ratio` - optional.

For a single source edit, use `imagine_image_to_image` instead.

## `imagine_text_to_video`

Generate a new video from a text prompt only (no source frame).

Inputs:

- `prompt` (required) - short present-tense shot description.
- `duration` / `aspect_ratio` / resolution fields as exposed by the live schema.

Prefer short shots; same prompt-craft rules as image-to-video below.

## `imagine_image_to_video`

Animate one still into a video.

Inputs:

- Singular source still path (required).
- `prompt` - short present-tense shot (recommended; required by some variants).
- `duration` - `6` (default), `10`, or `15` when exposed.
- `aspect_ratio` - optional; omit to keep the source ratio when animating one image.

**Prefer short shots.** Build video as a planned sequence of short clips, not one long take:

1. Plan the story as shots - one beat each.
2. Prefer more 6s shots over fewer long ones.
3. Create each shot's source still with `imagine_text_to_image` / `imagine_image_to_image` (keep character paths consistent).
4. Animate with `imagine_image_to_video` + that still path.

Key behaviors:

- **Prompt-craft:** one short, vivid moment in present tense with a clear camera movement, in 1-2 sentences.
- **Minimal but interesting:** one clear subject and a single simple motion or camera move.
- **Complex source?** Keep the subject fixed and move only the camera, or break into simpler shots.
- **Real people:** reference-first - drive from a verified reference; never animate a named person without one.
- Don't loop the same clip unless asked.
- Assemble multi-shot timelines with FFmpeg stream copy on the returned video paths.

## `imagine_reference_to_video`

Video from one or more reference image paths guided by a prompt. Prefer composing a single still with `imagine_reference_to_image` first, then `imagine_image_to_video`, when the goal is a clean first frame.

## Writing Strong Prompts

Describe, roughly in this order: **subject -> action/pose -> setting -> style -> composition -> lighting/mood -> key details.**

- Be specific and concrete; lead with the most important elements.
- State what to include rather than what to exclude.
- Use one coherent scene per prompt.
- Match `aspect_ratio` to the use case: `9:16` for phone/story, `16:9` for banner/video frame or OG share cards, `1:1` for avatar/icon.

## Real People and References

1. Search the web first to confirm identity, role, relationship, or event, even when it seems obvious.
2. Obtain a strong reference image on disk (user upload or search → sandbox path), then call `imagine_image_to_image` with that path. A user-uploaded photo is best.
3. If no suitable reference exists, ask the user to upload one rather than generating from a weak base.

## Showing results

- Call `render_file` with the sandbox `file_path` so the user sees the image/video in chat.
- For your own QC and scripts, `read_file` / shell on that path.

Game sprites and maps have their own pipelines — follow `generate2dsprite`,
`video2dsprite`, and `generate2dmap` for those.

## Failure modes to avoid

- Calling consolidated names `imagine_image` or `imagine_video` (they are not available).
- Passing a list to `imagine_image_to_image` / `imagine_image_to_video` (singular source only).
- Inventing a filesystem path that the tool never returned.
- Running chroma/ffmpeg on a path you made up without a real generation/`file_path`.
