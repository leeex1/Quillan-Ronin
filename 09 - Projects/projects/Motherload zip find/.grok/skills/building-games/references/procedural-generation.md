# Procedural Generation (seeded RNG, noise, dungeons/mazes/terrain, wave function collapse)

Consolidated from Red Blob Games, the simplex-noise library, and WFC primary sources (see Sources). Focus: what an AI builder needs to generate **varied but reproducible** worlds without the classic bugs — non-deterministic output, disconnected dungeons, blobby/banded terrain.

---

## 1. Seeded RNG is the foundation — never use bare `Math.random()`

`Math.random()` is **not seedable**, so you can't reproduce a level, share a seed, debug a bad map, or sync generation across clients in multiplayer. **Always drive procedural generation from a seeded PRNG.**

**mulberry32** — the go-to small, fast, good-quality 32-bit seeded PRNG:
```js
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; // [0,1)
  };
}
const rng = mulberry32(12345); // same seed → identical sequence forever
```

Rules:
- **Hash string seeds to an int first** (e.g. `xmur3`) so `"cave-42"` maps to a stable number.
- **Use separate RNG streams for independent systems** (terrain vs loot vs enemy placement) so changing one doesn't reshuffle the others (`mulberry32(seed)`, `mulberry32(seed ^ 0x9e3779b9)`, …).
- Add helpers: `randRange(a,b)=a+rng()*(b-a)`, `randInt`, `pick(arr)`, `shuffle` (Fisher–Yates using `rng`).
- For **networked/deterministic** generation, both sides must run the *same* PRNG and consume it in the *same order* (deterministic multiplayer is out of scope on this deploy target). Order-of-consumption bugs are the #1 desync source.
- Alternatives: `alea`, PCG. mulberry32 is plenty for games.

---

## 2. Noise: value / Perlin / simplex (for terrain, textures, organic fields)

Random-per-cell is white noise (harsh, uncorrelated). **Coherent noise** produces smooth, natural-looking fields where nearby points are similar.

- **Value noise** — cheap, interpolate random lattice values. OK for simple stuff.
- **Perlin noise** — classic gradient noise; can show axis-aligned artifacts.
- **Simplex / OpenSimplex** — fewer directional artifacts, scales better to higher dimensions; preferred default. Use the **`simplex-noise` npm package**.
  - **v4 API changed:** no more `new SimplexNoise()`. Use factory functions and pass your seeded PRNG:
  ```js
  import { createNoise2D } from 'simplex-noise';
  const noise2D = createNoise2D(mulberry32(12345)); // returns ~[-1, 1]
  const v = noise2D(x * 0.01, y * 0.01);            // scale coords = "frequency"
  ```
  (Also `createNoise3D`/`createNoise4D`; passing a PRNG makes output reproducible — without one it uses `Math.random()` = non-reproducible.)

**Key techniques (from Red Blob Games "Making maps with noise"):**
- **Frequency:** multiply input coords by a scale; small scale = large smooth features.
- **Octaves / fBm:** sum several noise layers at increasing frequency and decreasing amplitude (`amplitude *= persistence(≈0.5)`, `frequency *= lacunarity(≈2)`) for natural detail.
- **Normalize** noise from [−1,1] to [0,1] before thresholding.
- **Redistribution:** `elev = Math.pow(elev, k)` to bias toward valleys/plains vs peaks.
- **Island mask:** multiply by a radial falloff so edges are water.
- **Biomes:** sample **two independent noise fields** (elevation + moisture) and look up a Whittaker-style biome table — don't derive moisture from elevation (they'd correlate and look wrong).

---

## 3. Dungeon / room generation

Common approaches, pick per game:
- **Rooms + corridors (BSP or random placement):** place non-overlapping rooms, then connect. BSP recursively splits space and puts a room in each leaf; connect sibling leaves → guaranteed structure. Random placement: drop rooms, reject overlaps, connect with L-shaped/A* corridors.
- **Cellular automata caves:** fill grid randomly (~45% wall), run several smoothing passes (`cell = neighbors≥5 ? wall : floor`) → organic caverns. Then **flood-fill and keep only the largest connected region** (or carve tunnels to connect regions).
- **Drunkard's walk / random walk:** carve floors along a random walk for winding caves.

**Critical rule: guarantee connectivity.** After generation, **flood-fill from the entrance and verify every important tile (exit, key rooms, loot) is reachable.** Discard/regenerate or carve connectors otherwise. "Unreachable exit / locked-in player" is the #1 procgen bug. Also validate: min room count, spawn ≠ exit, no soft-locks.

---

## 4. Maze generation

- Grid where each cell has walls; carve passages.
- **Recursive backtracker (randomized DFS):** deep, winding mazes with long corridors — most common.
- **Randomized Prim's / Kruskal's:** more uniform, bushier mazes.
- All produce a "perfect maze" (exactly one path between any two cells). Add **loops** by knocking out extra walls if you want multiple routes.
- Use your seeded `rng` for neighbor choice so the maze is reproducible.

---

## 5. Terrain generation

- **Heightmap from fBm simplex** (see §2): sample noise per cell → elevation → color/mesh. Threshold into water/sand/grass/rock/snow bands.
- **Domain warping** (offset sample coords by another noise) for more natural, less "blobby" coastlines.
- For 3D: feed the heightmap into a plane geometry's vertex Y (Three.js `PlaneGeometry` + displace vertices) and `computeVertexNormals()`; consider chunking + LOD for large worlds.
- Keep generation **deterministic per chunk**: derive each chunk's local seed from `(worldSeed, chunkX, chunkY)` so infinite/streamed worlds are stable and reproducible.

---

## 6. Wave Function Collapse (WFC) basics

WFC is a constraint-solver that generates output where every local neighborhood matches examples/adjacency rules — great for tile maps, textures, and levels that must "fit together."

Algorithm (overlapping or simple-tiled):
1. Every cell starts in **superposition** (all tiles possible).
2. **Observe:** pick the lowest-entropy cell (fewest remaining options) and **collapse** it to one tile (weighted-random).
3. **Propagate:** remove now-impossible neighbors' options based on adjacency rules; repeat until stable.
4. Loop until all cells collapsed, or **backtrack/restart on contradiction** (a cell with zero options).

Rules & gotchas:
- Adjacency rules come from a sample image (overlapping model) or hand-authored tile edges (simple-tiled model).
- **Contradictions happen** — implement restart or backtracking; without it, generation hangs/throws.
- Weight tiles for aesthetics; use noise (from §2) to bias regional weights for large-scale structure, then WFC for coherent local detail.
- For JS: **mxgmn/WaveFunctionCollapse** (original), Boris the Brave's write-up + **DeBroglie**, or ndarray-based JS ports.

> Note: Red Blob Games does **not** have a WFC tutorial (a common false citation). Cite Maxim Gumin's repo and Boris the Brave instead. Red Blob Games *is* the canonical source for **noise maps** and **A\*/grids**.

---

## 7. Bug-prevention checklist
- **`Math.random()` for generation** → non-reproducible, unshareable, undebuggable, desyncs multiplayer; use a seeded PRNG.
- **String seed used directly** → NaN/garbage; hash to int (xmur3) first.
- **Shared single RNG stream across systems** → tweaking one system reshuffles all; use separate streams.
- **No connectivity check** → unreachable exits / trapped player; flood-fill validate & regenerate.
- **Forgetting to seed simplex-noise (v4)** → falls back to `Math.random()`, non-reproducible; pass your PRNG to `createNoise2D`.
- **Single-octave noise** → smooth but featureless; add fBm octaves.
- **Deriving moisture from elevation** → correlated, unrealistic biomes; use independent noise fields.
- **WFC with no contradiction handling** → infinite loop / crash; add restart/backtrack.
- **Per-frame regeneration or huge synchronous gen** → jank; generate in chunks / a web worker / on level load.
- **Non-deterministic chunk seeds** → seams/flicker in streamed worlds; derive chunk seed from `(worldSeed, cx, cy)`.

---

## Defaults to apply
- **Every generated procgen game gets a visible/optional seed** driven by **mulberry32** (with xmur3 for string seeds), plus `randRange/randInt/pick/shuffle` helpers and **separate RNG streams** per system. Reproducibility is free QA, shareable content, and multiplayer-safe.
- **Terrain/organic → seeded `simplex-noise` v4 (`createNoise2D(rng)`) with fBm octaves + redistribution + island mask + independent elevation/moisture** per Red Blob Games.
- **Dungeons → rooms+corridors or cellular-automata caves, ALWAYS followed by a flood-fill connectivity guarantee** (regenerate on failure). Ship this validation step by default — it prevents the worst procgen bug.
- **Offer WFC** for tile-based levels/textures (with contradiction restart), optionally seeded by noise for macro structure. Do heavy generation in a worker/on load, not per frame.

---

## Sources
- Red Blob Games — Making maps with noise functions: https://www.redblobgames.com/maps/terrain-from-noise/
- Red Blob Games — Noise (value/Perlin/simplex intro): https://www.redblobgames.com/articles/noise/introduction.html
- Red Blob Games — Procedural map/dungeon & grids index: https://www.redblobgames.com/
- simplex-noise (npm, v4 `createNoise2D` API): https://www.npmjs.com/package/simplex-noise , https://github.com/jwagner/simplex-noise.js
- mulberry32 / xmur3 seeded PRNGs (bryc's gist): https://github.com/bryc/code/blob/master/jshash/PRNGs.md
- Stefan Gustavson — Simplex noise demystified (PDF): https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
- Ken Perlin — Improving Noise (SIGGRAPH 2002): https://mrl.cs.nyu.edu/~perlin/paper445.pdf
- Cellular-automata cave generation (RogueBasin): https://www.roguebasin.com/index.php/Cellular_Automata_Method_for_Generating_Random_Cave-Like_Levels
- BSP dungeon generation (RogueBasin): https://www.roguebasin.com/index.php/Basic_BSP_Dungeon_generation
- Maze generation algorithms — Jamis Buck ("Buckblog") & "Mazes for Programmers": https://weblog.jamisbuck.org/2011/2/7/maze-generation-algorithm-recap
- Wave Function Collapse — Maxim Gumin (original): https://github.com/mxgmn/WaveFunctionCollapse ; Boris the Brave, "WFC explained": https://www.boristhebrave.com/2020/04/13/wave-function-collapse-explained/
