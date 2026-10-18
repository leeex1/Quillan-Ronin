# Expected deliverables and validation

Check the deliverable list for the pipeline you chose before calling a map
done, then run the validation list against what you actually wrote.

For a baked raster map:

- `assets/map/<name>.png`
- optional `<name>.prompt.txt`
- optional `data/<name>-collision.json` or `data/<name>-zones.json`
- code changes that load/use the image

Use this deliverable only for non-playable backgrounds or explicitly requested flat images. If actors must move through the scene, collide with level geometry, jump on platforms, collect items, trigger doors, or edit the level later, upgrade to a layered, parallax-stage, tilemap, or engine-native deliverable.

For a layered raster map:

- `assets/map/<name>-base.png`
- `assets/map/<name>-base.prompt.txt`
- optional `assets/map/<name>-dressed-reference.png` for prop planning
- `assets/props/<prop>/prop.png` folders, from one-by-one props or extracted prop packs
- `data/<name>-props.json` placement metadata
- `data/<name>-collision.json` and/or `data/<name>-zones.json` when gameplay needs them
- `assets/map/<name>-layered-preview.png`
- code changes that load the base, props, y-sorted renderables, collision, and zones

For a tilemap or layered tilemap:

- image-generated or user-supplied `assets/tilesets/<name>.png`
- optional tile slicing/atlas metadata
- engine-native tile layer data such as Tiled JSON, LDtk data, Godot TileMap scene data, Unity tile placement data, or project-native JSON
- object layers for spawns, exits, interactables, blockers, and zones
- a flattened preview assembled from the visual tileset and layer data
- no script-drawn final tileset art unless the user explicitly asked for procedural placeholders

For a playable side-view scrolling/action stage:

- image-generated parallax scenery layers such as `assets/map/<name>-sky.png`, `assets/map/<name>-far-bg.png`, `assets/map/<name>-mid-bg.png`, `assets/map/<name>-near-bg.png`, and optional `assets/map/<name>-foreground-overlay.png`
- one recorded `stage_canvas` shared by the primary parallax layers, `stage-reference`, and `stage-preview`
- `assets/map/<name>-background.prompt.txt` and prompt files/manifests for other generated visual assets
- `assets/map/<name>-stage-reference.png` as an in-world reference mockup for platform/object placement
- separate image-generated platform, terrain-chunk, foreground-occluder, hazard, door, pickup, checkpoint, gate, and exit sprites when these are visible scene objects
- `data/<name>-objects.json` or engine-native object layers for platforms, terrain chunks, hazards, pickups, doors, checkpoints, gates, exits, and foreground occluders
- `data/<name>-scene-hooks.json` or engine-native metadata for player spawns, actor spawn marker metadata, encounter/arena triggers, camera bounds, and exit links
- `data/<name>-collision.json` with explicit platform/solid geometry independent from the background pixels
- `assets/map/<name>-stage-preview.png` composed from the background plus objects for QA only
- code or scene changes that load the background, render object layers, and use the collision/object data as runtime gameplay data

Do not accept a single generated side-view action/platformer stage image plus collision rectangles as the final playable map. The stage must expose platforms or walkable lanes, hazards, doors, pickups, checkpoints, gates, exits, scene hooks, and camera bounds as separate runtime objects, tile/object layers, or metadata. Runtime `background` fields must point to the scenery-only background or parallax layer, never to `stage-reference` or `stage-preview`; previews are QA artifacts only.

For `grid_mode`:

- image-generated or user-supplied tileset/grid art
- grid dimensions, cell size, and map data in project-native JSON, Tiled JSON, LDtk, Godot TileMap, Unity Tilemap, or equivalent
- cell metadata for walkable/buildable, movement cost, terrain effects, resources, collision, and placement rules
- object layers for units, buildings, machines, cards/board slots, exits, spawns, and triggers
- a QA preview that can show optional debug grid/collision overlays

For `room_chunk_mode`:

- reusable chunk art or tile/object layers
- chunk metadata with `chunk_id`, size, entrances/exits, connection sockets, spawn markers, blockers, hazards, and camera bounds
- collision and seam validation metadata
- a chunk preview and, when multiple chunks exist, an assembled layout preview

For `scene_mode`:

- foundation-only `assets/map/<name>-base.png`
- in-world `assets/map/<name>-dressed-reference.png`
- separate props/interactables/blockers from one-by-one assets or compact prop packs
- placement, collision, zones, exits, camera bounds, and scene-hook metadata
- a QA preview composed from the base plus final runtime objects

For a prop pack:

- raw generated sheet with solid `#FF00FF` background
- extracted `assets/props/<prop>/prop.png` files
- `prop-pack.json` extraction manifest
- no `edge_touch` entries for accepted props

## Validation

Always validate what the chosen pipeline requires:

- map files exist and have expected dimensions
- prompt files or prompt manifest fields exist for generated visible assets
- transparent props contain alpha
- prop pack manifests parse and accepted props do not touch cell edges
- placement JSON parses and referenced prop files exist
- collision/zones JSON parses when present
- critical spawn, path, entrance, blocker, and zone points behave as expected
- playable/editable layered maps use a foundation-only base/background and do not bake runtime-controlled props, interactables, hazards, doors, gates, pickups, actors, foreground occluders, or reusable scene objects into the base
- playable stages have explicit runtime objects or metadata for every gameplay-relevant platform or walkable lane, blocker, hazard, door, pickup, checkpoint, gate, exit, player spawn, actor spawn marker, encounter/arena trigger, and camera bound
- playable side-view backgrounds are scenery-only and do not contain baked-in foreground gameplay platforms, hazards, pickups, doors, gates, checkpoints, or other reusable runtime objects
- `side_scroll_mode` primary parallax layers, stage references, and stage previews match the recorded `stage_canvas`; any repeatable strips or differently sized foreground sprites declare display size, anchor, scale, and repeat policy
- `side_scroll_mode` parallax layers have explicit render order, scroll factors, dimensions, loop/repeat policy, and are not used as collision sources
- `grid_mode` outputs include grid dimensions, cell size, cell metadata, object layers, and validation of critical walkable/buildable cells
- `room_chunk_mode` outputs include chunk dimensions, exits/connection sockets, seam validation, collision, and at least one assembled or per-chunk preview
- stage-reference maps preserve the background dimensions and their object plan matches the final object/collision metadata
- stage-reference and dressed-reference mockups contain no more than 9 distinct visible runtime prop/object candidates unless the user explicitly requested a larger pass
- reference mockups are followed by final props/objects, placement metadata, collision/scene-hook metadata, and a QA preview unless the user explicitly requested reference-only output
- flattened preview looks coherent at the game's camera size
