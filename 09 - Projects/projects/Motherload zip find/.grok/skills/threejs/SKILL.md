---
name: threejs
description: >
  Official Three.js API and TSL (Three.js Shading Language) reference for LLM
  code generation. Load when writing or debugging three.js / WebGL / WebGPU /
  custom materials / shaders / GLTF / advanced three APIs beyond basic game
  loop/controls. Prefer building-games for game correctness (loop, WASD,
  camera, orientation); use this skill for full API/TSL depth. Triggers on
  "three.js", "threejs", "WebGPU", "TSL", "NodeMaterial", "shader", "GLTF",
  "MeshStandard", "OrbitControls", "WebGLRenderer".
metadata:
  short-description: "Three.js + TSL full API (official llms-full reference)"
user-invocable: false
---

# Three.js (official LLM reference)

This skill vendors the **official** Three.js LLM documentation pack so the agent
can generate correct modern three.js without inventing outdated CDN/r128 APIs.

## Stack adaptation (this app builder)

You are in a **TanStack Start + React** workspace, not a bare HTML page:

| Official doc pattern | Do this here instead |
| --- | --- |
| `<script type="importmap">` + CDN three | **`npm install three`** (+ `@types/three` if needed); import from `"three"` / `"three/addons/…"` |
| Raw HTML canvas bootstrap | Prefer **@react-three/fiber + drei** for games/UI integration (`building-games` + `3d-libs.md`) |
| Standalone `WebGLRenderer` demo | Fine for a self-contained canvas module; still install three via npm so Vercel build has it |
| Always “latest” CDN version | Pin via **package.json** so dev and deploy match |

**three is not preinstalled** — add it with npm and leave it in `package.json`.

## When to load what

1. **Game / interactive 3D product** → start with **`building-games`** (loop, controls, orientation, camera, first-run, steer sign).
2. **R3F / drei / rapier wiring** → `building-games/references/3d-libs.md`.
3. **Deep three API, TSL, WebGPU, materials, loaders, postprocessing** → load  
   **`references/llms-full.txt`** (this skill’s full official dump).

Do **not** load `llms-full.txt` for simple 2D canvas games (Pong, tetris, etc.).

## Full reference

**Read on demand:**

```text
references/llms-full.txt
```

Source: https://threejs.org/docs/llms-full.txt (pinned copy for offline sandbox use).

Contents include: modern imports, WebGLRenderer vs WebGPURenderer, TSL complete
reference, NodeMaterial, loaders, post-processing, compute, and API tables.

## Quick defaults for this product

- Prefer **WebGLRenderer** (or R3F default) unless the user needs TSL/WebGPU compute.
- Cap pixel ratio (`renderer.setPixelRatio(Math.min(devicePixelRatio, 2))` or R3F `dpr={[1,2]}`).
- Dispose geometries/materials/textures on teardown (three does not GC GPU resources).
- For games: still obey **`building-games`** control and orientation self-tests.

## Finish check

- three (and R3F stack if used) is in `package.json` and imports resolve.
- No r128 / cdnjs script-tag patterns.
- If TSL/WebGPU used: materials and imports match `llms-full` (node materials, `await renderer.init()`).
