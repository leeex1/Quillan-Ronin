# Genre Playbook — Voxel / Minecraft-like

Block worlds you can walk, build, and mine. In the browser this is a **Three.js/WebGL performance problem first, game second** — the naive approach (one cube mesh per block) melts instantly. The whole genre is about **chunking, face culling, greedy meshing, and voxel raycasting**. Read `../threejs-foundational.md` first (draw calls, disposal, instancing, first-person controls).

---

## 1. Core mechanics (minimal-but-good scope for a demo)

1. **A voxel world** stored as data (block IDs in a 3D grid), rendered as merged chunk meshes.
2. **First-person movement + collision** against blocks (AABB vs voxels).
3. **Break and place blocks** via a voxel raycast (crosshair targets a block/face).
4. **A few block types** with distinct textures (atlas).
5. **Some world generation** (flat, or simple noise terrain) and chunk streaming around the player.

Flat world + walk + place/break + 3–4 block types is already recognizably "Minecraft." Skip inventory depth, lighting/AO, water, and mobs until the core render + edit loop is solid. **If you render each block as its own mesh, nothing else matters — you'll be at single-digit FPS.**

---

## 2. Chunking (the foundational structure)

- **Divide the world into chunks** (commonly 16×16×256, or 16×16×16 sub-chunks). Store each chunk's blocks in a **flat typed array** (`Uint8Array` indexed `x + y*SX + z*SX*SY`), not nested objects — cache-friendly and memory-light.
- **One merged mesh per chunk** (a single `BufferGeometry` built from all visible faces), not one mesh per block. This is the single most important rule — it turns thousands of draw calls into one per chunk.
- **Rebuild a chunk's mesh only when its blocks change** (place/break), not every frame. Cache the geometry; dispose the old one on rebuild.
- **Stream chunks** around the player: load/generate + build meshes for chunks within a radius; unload (and **dispose geometry/material/textures**) chunks that fall out of range. Do meshing off the main thread (**Web Workers**) to avoid frame hitches; transfer the typed arrays.
- **Neighbor awareness:** meshing a chunk needs to know the blocks in adjacent chunks (to cull faces at chunk borders correctly) — pass neighbor data or a padded copy.

---

## 3. Face culling & greedy meshing (the performance core)

- **Face culling first (mandatory):** only emit a block face if the neighboring block in that direction is **empty/transparent**. Interior faces between two solid blocks are never seen — skipping them removes the vast majority of geometry. This alone makes voxel worlds viable.
- **Greedy meshing (big win):** after culling, **merge adjacent coplanar faces of the same block type into larger quads.** Instead of one quad per block face, a flat 16×16 grass top becomes (ideally) one quad. Algorithm (per 0fps.net "Meshing in a Minecraft Game"): for each of the 6 face directions, sweep the volume slice-by-slice, build a 2D mask of visible faces of each type, then greedily expand rectangles over the mask (grow width, then height while cells match), emitting one quad per merged rectangle. Cuts vertex count and draw work dramatically.
- **Trade-off:** greedy meshing complicates per-face texturing/UVs and lighting/AO (merged quads span multiple blocks). For a demo, culled per-face meshing is acceptable; add greedy meshing when you need scale. Use a **texture atlas** (or a texture array) so all block textures share one material → one draw call per chunk. Set atlas textures to `NearestFilter` and mind bleeding at atlas tile edges (add padding or use a texture array).
- **Don't use `InstancedMesh` of cubes for terrain** — it still submits all 6 faces per block and per-face culling/greedy meshing wins massively. Instancing is fine for sparse identical props.

---

## 4. Voxel raycasting: place & break blocks

- **Use a grid-DDA voxel traversal (Amanatides & Woo), not `Raycaster.intersectObjects` against block meshes.** Step the ray cell-by-cell through the voxel grid: track `tMax`/`tDelta` per axis, advance to whichever axis boundary is nearest, and check the block at each visited cell until you hit a solid block or reach max distance. This is exact, cheap, and independent of how the mesh is built.
- **Break:** set the hit cell to empty; rebuild that chunk's mesh (and any neighbor chunk if the block was on a border).
- **Place:** place the new block in the cell **adjacent to the hit face** — you must track *which face* you crossed to enter the solid cell (the last step axis + direction gives the face normal). Place at `hitCell + faceNormal`. Reject placement if that cell is non-empty or overlaps the player's AABB (don't let the player entomb themselves).
- **Highlight** the targeted block/face (wireframe or overlay) so the player sees what they'll hit.

---

## 5. Collision, feel & world gen

- **Collision = AABB vs voxels:** resolve the player capsule/box against the blocks it overlaps, axis-separated (X, then Y, then Z), like a 3D version of tile collision. Sub-step fast movement to avoid tunneling through thin walls at high speed/low FPS.
- **First-person feel:** pointer lock, gravity + jump, step-up over 1-block ledges optional; block-break should have a short "mining" delay + crack overlay + particle burst for satisfaction; place/break sounds.
- **World gen:** simplex/Perlin noise for heightmaps (e.g. `simplex-noise`), layered (base terrain + caves via 3D noise) — keep it cheap and run in the worker with meshing.
- **Ambient occlusion** (darkening block corners) adds huge visual quality but complicates greedy meshing; add later.

---

## 6. Common bugs to avoid (checklist)

- **One mesh per block** → thousands of draw calls, instant death. Merge into one mesh per chunk.
- **No face culling** → 6× the geometry, most of it hidden. Cull faces against neighbors (including across chunk borders).
- **Rebuilding chunk meshes every frame** → rebuild only on edit; cache geometry.
- **Not disposing unloaded chunk geometry/materials/textures** → memory leak/crash while exploring (Three.js doesn't GC GPU resources).
- **Meshing on the main thread** → frame hitches when chunks stream; use Web Workers + transferable typed arrays.
- **Chunk-border faces wrong** (missing or double faces at seams) → meshing must see neighbor chunk blocks.
- **Using `intersectObjects` for block picking** → slow/fragile; use grid-DDA voxel traversal and track the entry face for placement.
- **Placing blocks on the wrong side / inside the player** → place at hitCell + face normal; reject if occupied or overlapping the player AABB.
- **Texture atlas bleeding** between tiles → padding or texture arrays, `NearestFilter`.
- **Collision tunneling at speed** → axis-separated AABB resolution + sub-stepping.

---

## Defaults to apply

1. **One merged mesh per chunk, never one mesh per block.** Store blocks in flat typed arrays; chunk the world (e.g. 16³). This is non-negotiable for browser performance.
2. **Always do face culling** (skip faces adjacent to solid blocks, across chunk borders too). Add **greedy meshing** (0fps.net algorithm) for scale; culled-per-face is OK for a small demo.
3. **Use a texture atlas / texture array so each chunk is one draw call.**
4. **Rebuild chunk meshes only on edit; stream and dispose chunks by distance; mesh + generate in Web Workers.**
5. **Break/place via grid-DDA voxel raycast (Amanatides & Woo), tracking the entered face** so placement goes on the correct adjacent cell; reject placements inside the player or occupied cells.
6. **Minimal scope:** flat/simple-noise world, FP movement + AABB voxel collision, place/break, a few atlas-textured block types. Get render + edit solid before AO, water, lighting, or mobs.

---

## Sources
- 0fps.net — "Meshing in a Minecraft Game" (greedy meshing, part 1): https://0fps.net/2012/06/30/meshing-in-a-minecraft-game/
- 0fps.net — "Meshing in a Minecraft Game" (part 2, texturing/AO trade-offs): https://0fps.net/2012/07/07/meshing-minecraft-part-2/
- Amanatides & Woo — "A Fast Voxel Traversal Algorithm for Ray Tracing" (grid-DDA block picking): http://www.cse.yorku.ca/~amana/research/grid.pdf
- Three.js — voxel/Minecraft-style tutorial (chunk geometry, raycast place/break): https://threejs.org/manual/#en/voxel-geometry
- Three.js — disposing objects (chunk unload): https://threejs.org/manual/#en/cleanup
- MDN — Web Workers (off-thread meshing/gen): https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
- "Let's make a voxel engine" community wiki (chunking patterns): https://sites.google.com/site/letsmakeavoxelengine/home
