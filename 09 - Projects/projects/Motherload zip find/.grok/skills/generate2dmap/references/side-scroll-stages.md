# Playable Stage Reference Rules

For playable side-view scrolling/action maps, an in-world stage reference mockup is mandatory before generating final scene objects or scene metadata. This applies across art styles and game styles, including pixel art, clean HD, side-scrollers, platformers, runners, shooters, brawlers, scrolling combat stages, and Megaman-like or Castlevania-like stages:

0. Choose and record one `stage_canvas`, for example `1536x864` for a default 16:9 HD side-scroller when the project has no explicit camera size. Use the engine's existing viewport aspect ratio when it exists. All primary parallax layers, the stage reference, and the stage preview must share this exact size unless a layer is explicitly marked as a repeatable strip.
1. Generate named parallax scenery layers as separate runtime images: `assets/map/<name>-sky.png`, `assets/map/<name>-far-bg.png`, `assets/map/<name>-mid-bg.png`, `assets/map/<name>-near-bg.png`, and optional `assets/map/<name>-foreground-overlay.png`.
- These layers are scenery only, not playable foreground. They may contain sky, clouds, mountains, distant buildings, distant castle walls, silhouettes, atmosphere, and non-colliding far depth.
- Do not collapse these layers into only `assets/map/<name>-background.png` for a playable `side_scroll_mode` stage. A single scenery background is allowed only when the user explicitly requests a flat/non-parallax background; in that case still continue with stage reference, separate objects, collision, camera bounds, and QA preview.
   - Each primary layer prompt must specify the same target canvas size/aspect ratio, same camera framing, same horizon height, and same top-left aligned composition. If image generation returns different sizes, regenerate or normalize them to `stage_canvas` before using them together.
   - Repeatable strips and foreground/object sprites may have different source dimensions, but they must declare display size, anchor point, repeat axis, and scale in metadata. They are not substitutes for the primary parallax plates.
   - It must not contain walkable floors, platform tops, terrain chunks, spike traps, pickups, crates, doors, gates, checkpoints, ladders, near fences, near stone walls, enemies, player characters, UI, labels, or any object that should later be edited, collided with, reused, or layered independently.
   - Keep the playable foreground lane visually open or neutral so separate platform/object layers can stack clearly over it.
2. Hand the background to `imagine_image_to_image` as a real reference: pass its sandbox `file_path`. Also `read_file` it so you can see it; do not rely on a path string or prompt text as the reference.
3. In the stage-reference prompt, explicitly say: use the provided background image as the visual reference, preserve exact camera/framing/dimensions/horizon/depth/entrances/exit direction, and generate an in-world stage reference mockup.
4. Generate `assets/map/<name>-stage-reference.png` from the visible background.
5. In the stage reference, visually place the intended scene layout as natural game-world objects or subtle blockout geometry: platforms or walkable lanes, terrain chunks, foreground occluders, hazards, pickups, doors, checkpoints, gates, and exits.
   - Use at most 9 distinct visible runtime object candidates in the stage reference unless the user explicitly asks for a larger object pass. Repeated placements of the same platform, terrain chunk, hazard, pickup, checkpoint, door, gate, or occluder count as one candidate and should be repeated later in metadata.
   - Prioritize objects that the final game must render or collide with separately. Avoid filling the mockup with many small decorative foreground props that will not become reusable assets.
6. Do not draw spawn markers, actor markers, arena trigger zones, camera bounds, arrows, labels, circles, outlines, numbered callouts, text, legends, or UI overlays in the reference image. Record player spawn, actor spawn markers, arena triggers, camera bounds, and exit links later as scene-hook metadata.
7. Use the stage reference to decide object identities, sizes, coordinates, render order, collision shapes, and camera bounds.
8. Continue through the post-reference object production gate: generate or define final platforms, terrain chunks, hazards, pickups, doors, checkpoints, foreground occluders, and other visible scene objects as separate assets, tile layers, or object layers. Compose the final runtime preview from the original background plus these separate runtime objects.

The stage reference is an in-world reference mockup. Do not ship it as the runtime map, do not infer collision from its pixels, and do not cut platform objects out of the baked reference image. If a platform must be reusable or collidable, generate it as a separate platform object, terrain chunk, tile, or engine-native object.

If the generated background already has obvious foreground gameplay pieces baked into it, do not use it as `background` in runtime data. Regenerate the scenery-only background or demote that image to a concept/reference artifact.

Scene hooks are metadata only. Do not generate enemy, boss, NPC, player, projectile, or animation sprites inside `generate2dmap`; call `$generate2dsprite` separately when the game needs those assets.

If a playable side-view scrolling/action run has already generated a background but has not generated `assets/map/<name>-stage-reference.png`, pause the platform/props pipeline and generate the stage reference next. Background plus props is not enough evidence that the level layout is coherent.
