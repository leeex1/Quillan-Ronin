---
name: building-games
description: >
  Build browser games and interactive/canvas/3D experiences in this TanStack
  Start + React app. Use for any game, simulation, or WebGL/Canvas experience —
  2D or 3D, single-player. Covers the game loop & timing, 3D orientation/camera
  conventions, collision, performance, assets, audio, save, game feel, and
  per-genre playbooks. For WASD / vehicle / flight **input signs and inverted
  A/D**, open the **`controls`** skill — do not rely on this file or racing-kart
  alone. Triggers on "game", "minecraft", "fps", "platformer", "racing",
  "tetris", "snake", "shooter", "3d", "three.js", "canvas", "voxel", "physics".
metadata:
  short-description: "Browser games: loop, 3D orientation, camera, perf, assets, genres"
user-invocable: false
---

# Building Games

Build a **playable, correct** browser game — not a static screenshot. A game is
just a React route with a `<canvas>` (or `<Canvas>` for R3F) plus DOM overlay UI.
Style the overlay (start screen, HUD, menus) with the **`design-ui`** skill; this
skill owns the gameplay loop and world.

**Controls / inverted A/D:** open **`.grok/skills/controls/SKILL.md`** **before**
writing WASD, steering, or flight input. Vehicle/flight demos often ship with
A/D flipped if you only read this file or a single genre playbook.

**Scope note — single-player, bots, or small P2P co-op:**
- Ship **single-player** or **single-player + AI/bots** by default.
- **2–8 player co-op / casual realtime** (shared cursors, party games, casual
  action among friends) is supported — use the **`multiplayer-p2p` skill**
  (WebRTC mesh, signaled at `/api/rtc`). Read its trust model first.
- P2P is the only supported multiplayer right now. Do not half-build sockets
  that can’t connect.

**References (load on demand):**
- **`controls` skill** (`../controls/`) — **required** for movement/steer/flight:
  player-visible A/D, inverted-steer anti-pattern, flight ailerons, mandatory
  self-test + `window.__controlsTest`. Not optional for vehicles/planes.
- `references/threejs-foundational.md` — the deep 3D/loop/perf reference. Read for 3D.
- `references/3d-libs.md` — three.js + @react-three/fiber + drei + rapier usage.
- **`threejs` skill** (`../threejs/`) — official full Three.js + TSL API dump
  (`llms-full.txt`). Load for advanced materials/shaders/WebGPU/loaders; not for
  simple 2D canvas games.
- `references/babylon.md` — Babylon.js, the batteries-included 3D engine alternative.
- `references/phaser.md` — Phaser 3, the default engine for 2D games.
- `references/ecs-architecture.md` — entity-component-system structure for larger games.
- `references/genres/*.md` — per-genre playbooks (fps, platformer-2d, racing-kart,
  puzzle-match3-tetris, voxel-minecraft, endless-runner, topdown-twin-stick,
  tower-defense, board-card-chess). Genre files **do not** replace **`controls`**.
- `references/game-feel-juice.md`, `input.md`, `audio.md`, `collision-physics.md`,
  `save-persistence.md`, `procedural-generation.md`, `ai-pathfinding.md`.
- **`game-asset-core`** (+ `game-animation-frames` / `game-tilesets` /
  `game-character-consistency` / `game-ui-icons`) — engine-ready 2D art defaults
  and verification when generating sprites, sheets, tiles, or UI (see §6).

Pick the specific genre/topic reference for the build; this file is the universal
core for loop/world. **Input signs → `controls`.** **2D game art → `game-asset-core`.**

---

## 1. Game loop & timing (the #1 correctness issue)

- Drive the loop with the engine's RAF loop (`renderer.setAnimationLoop`, R3F
  `useFrame`, or `requestAnimationFrame` for 2D canvas). **Never** `setInterval`/
  `setTimeout`/`Date.now()` for game timing.
- **Scale ALL movement/animation by delta time** (seconds) so speed is frame-rate
  independent (60Hz vs 144Hz). Compute delta **once per frame** and reuse it.
  - three.js: use `THREE.Timer` (not `Clock` — `Clock.getDelta()` returns ~0 on a
    second call in the same frame, a classic freeze bug).
- **Cap delta** (`min(delta, 0.1)`) so a backgrounded tab doesn't teleport things.
- **Fixed timestep for physics/gameplay:** accumulate delta and step simulation at
  a fixed rate (e.g. 1/60) while rendering at display rate — prevents tunneling and
  non-determinism.

## 2. Controls (delegate to the `controls` skill)

**Open `.grok/skills/controls/SKILL.md` before implementing any WASD / steer /
flight code.** That skill is the source of truth for:

- Player-visible **A = left / D = right** (chase cam, while moving forward)
- Why **`KeyA → steer−` + `yaw += steer * +rate` inverts** (most common bug)
- Vehicle vs FPS (strafe ≠ steer), fixed-wing ailerons, heli/drone notes
- Mandatory self-test + `window.__controlsTest` probe

Do **not** treat `genres/racing-kart.md` as the only place steer signs live —
planes, jetskis, and mechs never open it.

**Short reminder (full detail in `controls`):**

```
// Vehicle yaw body (chase cam): A must increase yaw with this basis
forward = (-sin(yaw), 0, -cos(yaw))
// KeyA → steer = +1;  yaw += steer * turnRate * speedFactor * dt
// WRONG (ships inverted): KeyA → steer = -1; yaw += steer * turnRate * dt
```

- **Pointer lock:** mouse-look only — implement WASD yourself; click-to-play
  overlay; dismiss on lock.
- Track keys with held state + **dt**; unify devices via `references/input.md`.
- **Finish:** run the `controls` skill checklist. Screenshot-only is insufficient
  for vehicles/flight.

## 3. 3D orientation & world objects (the "sideways/backwards" bugs)

- three.js is **right-handed, +Y up**: +X right, +Y up, +Z toward viewer. **Meshes
  face +Z; cameras look −Z** (the classic "camera backwards" gotcha).
- Primitives like `Cone`/`Cylinder` point **+Y** by default → rotate to align a
  tip with forward (`geo.rotateX(Math.PI/2)`).
- **Orienting a mesh to face `forward`** (meshes face **+Z**): simplest correct way
  is `mesh.lookAt(mesh.position.clone().add(forward))`. To build the basis by hand,
  set the **+Z column to `forward`** and choose the x-axis that keeps it a *proper*
  right-handed rotation (`det = +1`):
  `xAxis = normalize(cross(up, forward))`, then `makeBasis(xAxis, up, forward)`.
  Note this is `cross(up, forward)` — **not** the movement `right = cross(forward, up)`
  from §2. Targeting +Z (instead of a camera's −Z) flips the x-axis sign so that
  `xAxis × up = forward`; the frame stays right-handed (no mirroring). Do **not** use
  `makeBasis(xAxis, up, -forward)` for a mesh — that targets −forward (the *camera*
  convention) so the mesh faces **backwards**.
- **Orienting a camera** to look along `forward` (cameras look **−Z**): set the +Z
  column to `-forward` — `xAxis = normalize(cross(forward, up))`, then
  `makeBasis(xAxis, up, -forward)` (or just `camera.lookAt(target)`).
- Keep a consistent world `up` so objects stay upright; only rotate flat primitives
  to stand up.
- Verify glTF import orientation; debug with `AxesHelper`/`ArrowHelper`. Upright
  self-test: characters stand on the ground plane, not lying/sunk.

## 4. Camera must agree with movement

- Keep a **dedicated `moveForward`/`moveRight`** for movement, computed once and
  never mutated by camera code (aliasing a shared temp vector makes camera and
  movement disagree — a real repro bug).
- Third-person follow: `desired = playerPos + up*height + moveForward*(-followDist)`;
  lerp the camera toward it (use exp-based smoothing, delta-scaled), `lookAt(player)`.
- Isolate-the-layer debug order: (1) keys register → (2) movement signs correct →
  (3) camera agrees. Fix in that order.

## 5. Performance

- **Minimize draw calls** (`renderer.info.render.calls`, target <100): share
  materials, `InstancedMesh`/`BatchedMesh` for repeated objects, atlases.
- **Dispose GPU resources yourself** (`geometry/material/texture.dispose()`) on
  level change — three.js does not GC them. **Pool** bullets/enemies/particles.
- No per-frame allocations (reuse temp vectors). Compress textures; LOD for distance.

## 6. Assets (avoid the generated-photo trap)

- **Interactive 3D elements** (weapon viewmodels, characters, props, projectiles)
  → build from **3D geometry / glTF**, not a generated image. A flat photorealistic
  JPG of a gun-in-hands used as an FPS viewmodel looks wrong, can't animate, and
  (JPG has **no alpha**) renders as a black box. Parent a real 3D viewmodel to the
  camera as an overlay render layer.
- Reserve image generation for **flat 2D** assets only (textures, sky/menu
  backgrounds, 2D sprites, UI art). **Never** use a generated photo as a 3D mesh,
  viewmodel, or character substitute — build those in 3D geometry / glTF.
  Set `crossOrigin="anonymous"` on images drawn to canvas/textures.
  See the **`imagine`** skill (2D only — image tools cannot produce real 3D).
- **Engine-ready game art doctrine** → open **`game-asset-core`**
  (`../game-asset-core/`) for defaults + blind verify + retry discipline, then the
  matching specialist: **`game-animation-frames`** (loop / motion laws),
  **`game-tilesets`**, **`game-character-consistency`**, **`game-ui-icons`**.
  These are **QC/doctrine**, not the export pipeline. Do not ship stick-figure
  placeholders when real art is expected.
- **2D game sprites / animation sheets** → run **`generate2dsprite`**
  (`.grok/skills/generate2dsprite/SKILL.md`): solid **`#FF00FF`** magenta
  `imagine_text_to_image` sheets + chroma postprocess scripts (magenta is required for the
  processor). Wire transparent PNGs/GIFs into Canvas/Phaser. Still apply
  **`game-asset-core`** (+ animation/character specialists when relevant).
- **2D maps / levels / prop packs** → open **`generate2dmap`**
  (`.grok/skills/generate2dmap/SKILL.md`). Prefer foundation-only base + separate
  props/collision for playable maps. Browser default: `raw_canvas` / Phaser.
  Tileable ground/walls → also **`game-tilesets`** for 2×2 seam checks.
- **Optional denser locomotion** → run **`video2dsprite`** (Grok
  `imagine_image_to_video` + sandbox scripts; magenta base). Prefer `generate2dsprite`
  for crisp production heroes. Use **`game-animation-frames`** for loop/flip-test
  laws; prefer **`video2dsprite`** over ad-hoc ffmpeg-only harvest in this
  sandbox.

## 7. Audio, save, feel
- **Audio**: unlock `AudioContext` on the first user gesture (tap-to-start) or iOS
  is silent; re-resume on `visibilitychange`. (`references/audio.md`)
- **Save**: `localStorage`/IndexedDB with a `version` field + migrations.
- **Juice**: screen shake, hit-stop, eased tweens, particles — cheap, huge
  perceived-quality lift. Keep presentation separate from simulation.

## 8. Mobile
- Distinguish canvas buffer size from CSS size; respect `devicePixelRatio`.
- `touch-action: none`, letterbox-fit to a base resolution, handle orientation.
- Touch controls (virtual joystick + action buttons), ≥44px targets.

---

## Stack / engine choice
- **3D → three.js**, ideally via **@react-three/fiber + drei** (fits the React
  app; drei gives pointer-lock/controls/loaders) + **@react-three/rapier** for
  physics/character controllers. See `references/3d-libs.md`.
- **2D →** native Canvas 2D is enough for snake/tetris/flappy/platformer; reach for
  Phaser only when the genre needs it.
- These game deps are **not preinstalled** — `npm install` them (and make sure they
  land in `package.json` so the Vercel build has them).

## Finish criteria (before "done")
- Loads with **no console errors**; visible gameplay (not a blank canvas).
- **`controls` skill self-test passed** (A = left / D = right from chase cam
  while moving forward; flip one sign if inverted). Not screenshot-only.
- 3D upright & camera-agrees self-tests pass (§3, §4).
- Runs on mobile viewport with touch controls.
- Production build (`npm run build`) renders the built output, not just dev.
- **Share / X card:** open the **`og`** skill — custom `public/og.jpg` **and**
  `"type": "x:game"` in `src/lib/og/site.json`. X uses `og:type="x:game"`
  to present the unfurl as a game card; do not use `twitter:card` or invent
  `x:type` for this. `browser-smoke` / `brand-check` warn when canvas apps omit
  the `site.json` field.
