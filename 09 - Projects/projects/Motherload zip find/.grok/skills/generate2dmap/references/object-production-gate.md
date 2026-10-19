# Post-Reference Object Production Gate

An in-world reference mockup is never the final deliverable by itself. After generating `dressed-reference` or `stage-reference`, continue with:

1. Make both images visible in conversation context before any object/prop generation:
   - the original `base` or `background`
   - the generated `dressed-reference` or `stage-reference` mockup
2. If either image is a local file, call `read_file` on it immediately before writing object lists or object/prop image prompts. Do not rely on file paths alone.
3. Create a concrete object list from the visible reference mockup while cross-checking the original base/background: object id, type, approximate position, approximate size, render layer, collision role, and asset strategy.
   - If the reference contains more than 9 distinct visible runtime object candidates, reduce the generated asset list to the 9 most gameplay-relevant candidates first, then represent extra repeats or low-value decorations through placement metadata or a later asset pass.
   - Classify every object before generation. Compact decorative props may be batched; wide/long, tall/large, collision-bearing, and tileset/strip objects must use one-by-one, strip, custom wide pack, tile/object-layer, or engine-native strategies.
4. For each visible runtime object, choose exactly one asset strategy:
   - generate a separate transparent asset with `$generate2dsprite` or direct `imagine_text_to_image`
   - extract it from a generated prop/object pack
   - represent it as a tile/object layer if the engine/editor pipeline is tile-based
5. For every object/prop generation that must match the map style, pass the base/background and/or reference mockup via `imagine_image_to_image` (`file_path`) or `imagine_reference_to_image` (path list for 2+) — and say in the prompt that the provided reference images are the visual context. The generated asset must match the original map style and correspond to an object visible in the reference mockup.
6. Generate or define the final platforms, terrain chunks, props, hazards, pickups, doors, gates, checkpoints, exits, foreground occluders, and other visible scene objects. Do not skip this step just because the reference mockup already contains them visually.
7. Write placement metadata such as `data/<name>-props.json`, `data/<name>-objects.json`, engine-native object layers, or tile/object data.
8. Write collision, zones, scene hooks, camera bounds, and exits as structured metadata.
9. Compose a QA preview from the original base/background plus final runtime objects.

Reference-only output is incomplete for any playable map, layered map with props, side-view stage, engine scene, or request that asks for separate props/editable objects. Only stop at a reference mockup if the user explicitly asks for a reference-only concept image.

For prop packs or object packs generated after a reference mockup, the prompt must be derived from the visible reference mockup and original base/background, not from memory or filenames. It should list the exact objects being generated and preserve the art style, lighting, perspective, and scale cues from the original base/background.
