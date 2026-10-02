# Browser Game Best Practices — Three.js / Babylon.js / Phaser (controls, camera, orientation, loop, assets, perf)

Consolidated from official docs, well-regarded tutorials, and engine forums (see Sources at bottom). Focus: what an AI builder needs to get **browser games that feel right and don't have the usual bugs.**

---

## 1. Game loop & timing (the #1 correctness issue)

**Rules:**
- Drive the loop with **`renderer.setAnimationLoop(fn)`** in Three.js (internally uses `requestAnimationFrame`, plays nicely with WebXR) or the engine's built-in loop (Phaser `update(time, delta)`, Babylon `scene.onBeforeRenderObservable` / `engine.runRenderLoop`). Avoid `setTimeout`/`setInterval` and `Date.now()`.
- **Scale ALL movement/animation by delta time** (seconds since last frame) so speed is frame-rate independent (30fps laptop vs 144Hz monitor). e.g. `mesh.position.x += speed * delta`.
- **Cap delta** to avoid huge jumps after a backgrounded tab: `delta = Math.min(delta, 0.1)`.
- **Compute delta exactly once per frame** and reuse it everywhere.
  - Three.js gotcha: `THREE.Clock.getDelta()` mutates state — calling it more than once per frame returns ~0 on later calls (very common bug). Newer `THREE.Timer` (Clock deprecated ~r183) has an explicit `.update()` so delta can be read multiple times safely.
- **For physics/gameplay stability use a fixed timestep + accumulator** ("Fix Your Timestep"): run `fixedUpdate(FIXED_STEP)` (e.g. 1/60) in a `while (accumulator >= FIXED_STEP)` loop, render at display rate, optionally interpolate. Variable delta is fine for purely visual demos.
- **Modularize**: give objects a `.update(delta)`/`.tick(delta)` method and iterate an "updatables" list; keep the top-level loop lean.
- **On-demand rendering**: if nothing is animating, stop the loop / render only on change to save battery (Three.js `setAnimationLoop(null)`; R3F `frameloop="demand"`).

**Reference pattern (Three.js, Timer + capped delta + fixed step):**
```js
const timer = new THREE.Timer();
let accumulator = 0; const FIXED = 1/60;
function animate() {
  timer.update();
  let delta = Math.min(timer.getDelta(), 0.25);
  accumulator += delta;
  while (accumulator >= FIXED) { fixedUpdate(FIXED); accumulator -= FIXED; } // physics/AI
  updateVisuals(delta);
  renderer.render(scene, camera);
}
renderer.setAnimationLoop(animate);
```

---

## 2. First-person / WASD + mouse-look controls (Three.js)

The canonical reference is the official `misc_controls_pointerlock` example (https://threejs.org/examples/misc_controls_pointerlock.html).

**Key facts & rules:**
- **`PointerLockControls` only handles the Pointer Lock API + mouse-look (yaw/pitch). It does NOT include WASD movement — you implement keyboard + translation yourself.**
- Add **`controls.getObject()`** (the yaw container) to the scene, not the camera directly; the camera is a child for pitch.
- Track keys with **boolean state flags** set in `keydown`/`keyup` (support both `KeyW`/`ArrowUp`, etc.). Do NOT move on the keydown event itself.
- Move with **velocity + delta time** (acceleration + friction/damping), not a fixed distance per press — feels far better.
- Use **`controls.moveForward()` / `controls.moveRight()`** — they respect current yaw (local XZ plane). Do NOT modify `camera.rotation`/`camera.position` directly for horizontal movement.
- Pointer lock **requires a user gesture**: always show a "Click to play" blocker overlay; call `controls.lock()` on click; handle `'lock'`/`'unlock'` events and `pointerlockerror`; provide escape-to-unlock and a crosshair.
- **Avoid the deprecated `FirstPersonControls`** (behaves like fly controls).
- For real collisions/slopes/stairs, integrate a physics character controller: **Rapier.js `KinematicCharacterController`** (currently favored), or cannon-es (`PointerLockControlsCannon`), Ammo.js. Sync the body to `controls.getObject()`.
- Encapsulate as a `FirstPersonController` class with `update(delta)`, `lock()`, `dispose()`, and tunable `speed`/`jumpHeight`. Add sprint (Shift), crouch, head-bob for feel.
- R3F: use drei `<PointerLockControls />` + `useKeyboardControls` + `useFrame`.

**Other camera rigs:**
- **Orbit / product inspection:** Three.js `OrbitControls`; Babylon `ArcRotateCamera` (set `lower/upperRadiusLimit`, `lower/upperBetaLimit`, `inertia`, `panningSensibility`).
- **Third-person follow:** lerp camera toward a target offset behind the player each frame; `camera.lookAt(player)`; Babylon has `FollowCamera`.
- Keep the far plane / `camera.maxZ` (Babylon) as small as practical — improves culling, overdraw, and depth precision.

---

## 3. 3D orientation & coordinate conventions (a top source of "why is my model sideways/backwards" bugs)

- **Three.js is right-handed, +Y up:** +X right, +Y up, +Z toward the viewer. Default camera sits at +Z looking toward −Z. (Differs from Unreal's Z-up, etc.) Changing global up (`Object3D.DefaultUp`) is discouraged — it affects helpers/grids.
- **Forward-axis gotcha:**
  - Most `Object3D` (Mesh/Group): **local forward is +Z** `(0,0,1)`. Model/rotate so the "front" aligns with +Z.
  - **Camera** (and some lights): forward is **local −Z** `(0,0,-1)`. This is the classic "camera looks backwards" gotcha.
- Many primitives (`ConeGeometry`, `CylinderGeometry`) point along **+Y** by default → rotate before use, e.g. `geo.rotateX(Math.PI/2)` to make +Y → +Z (align an arrow/cone's tip with forward).
- **`object.lookAt(target)`** rotates in **world space** so the forward axis points at the target (uses `.up`, quaternions internally; does not move the object). For nested targets, first `target.getWorldPosition(v)`. To face a *direction* vector: `obj.lookAt(obj.position.clone().add(dir))`.
- Imported glTF/GLB models may need an initial rotation to match +Z-forward. glTF/Blender export is right-handed Y-up, consistent with Three.js.
- Debug with `AxesHelper` and `ArrowHelper` to visualize orientation quickly.

---

## 4. Performance (Three.js-centric, principles apply broadly)

**The #1 killer is draw calls** (`renderer.info.render.calls`). Target **<100/frame** for smooth 60fps on most hardware; 500+ stutters; mobile is stricter.

- **Share materials** aggressively (reuse one material instance across meshes).
- **Instancing (`InstancedMesh`)** for many *identical* objects (trees, crowds, bullets) → one draw call; update via `setMatrixAt` + `instanceMatrix.needsUpdate`. Note: default frustum culling applies to the whole instanced mesh.
- **`BatchedMesh`** (r156+) for *varied* geometries sharing one material, with per-object visibility + `perObjectFrustumCulled`.
- **Merge geometries** (`BufferGeometryUtils.mergeGeometries`) for fully static scenery sharing a material (loses per-object culling — one bounding volume).
- **Frustum culling** is automatic (`mesh.frustumCulled`); after modifying geometry call `computeBoundingBox()`/`computeBoundingSphere()`. Reduce `camera.far` where possible.
- **Dispose GPU resources manually** — Three.js does NOT GC them. On removing objects/switching levels: `geometry.dispose()`, `material.dispose()`, and dispose all textures. Watch `renderer.info.memory`. Use **object pooling** for particles/enemies instead of create/destroy churn.
- **Avoid allocations in the hot loop** (no `new Vector3()`/`new Matrix4()` per frame; reuse temporaries; Babylon has `TmpVectors` and `xxxToRef()`).
- **LOD** (`THREE.LOD`) — swap lower-poly at distance. Compress geometry (Draco) and textures (**KTX2/Basis**); prefer texture atlases.
- Limit lights; bake lighting/AO where possible; keep shadow maps small (and `RENDER_ONCE` refresh for static lights).
- Add **Stats.js**/FPS counter during dev; profile with Spector.js / engine inspector; profile on target devices.

**Babylon.js specifics:** prefer WebGPU w/ fallback + `powerPreference:"high-performance"`; `scene.performancePriority` (Intermediate/Aggressive); `scene.autoClear=false` when meshes cover viewport; `scene.skipPointerMovePicking=true` if no hover-picking; **thin instances** for static crowds; `mesh.freezeWorldMatrix()`, `material.freeze()`, `scene.freezeActiveMeshes()` for static content; `engine.setHardwareScalingLevel(2)` on low-end/mobile; `SceneOptimizer` to auto-degrade quality to hit a target FPS; keep `camera.maxZ` low. Camera "Behaviors" (Framing/Bouncing/AutoRotation) give polish for free.

**Phaser specifics:** heavy **object pooling** (Groups with `maxSize`, recycle via `active/visible=false`), process only active objects, texture atlases (Texture Packer), destroy tweens/emitters on `SHUTDOWN`, smaller canvas + CSS upscaling, lazy-load per-level assets.

---

## 5. Engine / stack choice

- **2D games → Phaser** (Phaser 3/4): scenes, Arcade/Matter physics, Scale Manager, huge ecosystem, great mobile support. Best default for 2D.
- **3D games → Three.js** (most control, largest community) or **Babylon.js** (batteries-included: inspector, physics, WebGPU, camera behaviors, SceneOptimizer). **PlayCanvas** if you want a full editor/engine.
- Physics: 2D → Arcade (simple) / Matter (rigid bodies); 3D → **Rapier.js** (fast, WASM, popular) / cannon-es / Ammo.js.
- Tooling: **Vite + TypeScript**; PWA (manifest + service worker) for installable/offline mobile.

**Phaser architecture (from best-practice write-ups):** one file per Scene extending `Phaser.Scene`; a scene stack of Boot → Preload (progress bar) → Menu → Game → parallel UI/HUD (`scene.launch`) → Pause/GameOver. Lifecycle: `init(data)` → `preload()` → `create(data)` → `update(time,delta)`; clean up on `SHUTDOWN`. Constants file for all string keys. Decouple scenes via an EventEmitter "EventsCenter" or `registry`, not direct references. Prefabs (custom GameObject subclasses) + FSM/ECS for complex logic instead of a giant `update()`.

---

## 6. Assets: generate vs. code vs. link

- **Generate real images** for sprites/textures/backgrounds/UI where the builder can (matches Lovable/v0 philosophy: no placeholder images in the final product). For pixel art request specific specs ("clean 32×32 side-view run cycle, 8 frames, sprite sheet, transparent background").
- **Don't hand-draw complex SVG/geometry** for illustrations or maps — use real assets or a library (echoes v0/Claude anti-slop rules).
- **Optimize for the browser:** compressed textures (KTX2/Basis/WebP), power-of-two, atlases, keep total download small (aim <20–50MB). Draco/Meshopt for glTF.
- **Placeholder > bad attempt** (Claude): if a good asset can't be produced, use a clean placeholder rather than an ugly hand-rolled one.
- Set **`crossOrigin='anonymous'`** on images drawn to `<canvas>`/textures to avoid CORS taint (v0 rule).
- 3D model pipeline: glTF/GLB (Y-up, right-handed) with Draco/Meshopt/KTX2; verify forward-axis orientation on import.

---

## 7. Mobile / responsive / touch

- Distinguish the **canvas drawing buffer** (`canvas.width/height`) from **CSS display size**; account for `devicePixelRatio` to avoid blurry/tiny output. Rendering at a smaller internal resolution + CSS upscale is a valid perf win; `image-rendering: pixelated` for pixel art.
- Choose a **base design resolution** + aspect ratio; **letterbox (FIT)** to avoid stretching. Phaser: `scale: { mode: Phaser.Scale.FIT, autoCenter: CENTER_BOTH, width, height }` and handle the `resize` event; listen for `orientationchange`.
- Viewport meta: `width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no`; CSS `touch-action: none` and `preventDefault()` on touch to stop scroll/zoom.
- **Touch controls:** virtual joystick (e.g. **nipplejs**, `mode:'static'` or `'dynamic'`), split screen (left = move, right = actions/aim), normalize input to −1..1 and apply with delta time, visual feedback on active. **Minimum touch/hit target 44px** (Claude/Apple HIG). Unify keyboard + touch + Gamepad API into one input state.
- Test on real iOS Safari (audio unlock on gesture, WebGL context loss, touch coalescing) and low-end Android.

---

## 8. Common browser-game bugs to prevent (checklist)
- Movement tied to frame rate (no delta) → runs too fast/slow on different displays.
- `Clock.getDelta()` called multiple times per frame → things freeze/jitter.
- Model faces wrong way → forgot +Z-forward convention / camera −Z; fix geometry rotation or `lookAt`.
- Pointer lock never engages → missing user-gesture click / blocker overlay.
- Memory leak / crash after level reload → geometries/materials/textures not disposed.
- Frame drops → too many draw calls (needs instancing/batching/atlases), per-frame allocations, uncompressed textures.
- Blurry or misaligned on mobile/retina → not handling devicePixelRatio / canvas resize.
- CORS-tainted canvas → missing `crossOrigin='anonymous'`.
- Physics tunneling/instability → variable timestep instead of fixed step.
- Audio doesn't play on mobile → not unlocked on first user gesture.

---

## Sources
- Three.js official: PointerLock example (https://threejs.org/examples/misc_controls_pointerlock.html), docs for `Timer`, `Object3D.lookAt`, `InstancedMesh`, `BatchedMesh` (https://threejs.org/docs/).
- "Discover three.js" — animation loop / Loop class (https://discoverthreejs.com/book/first-steps/animation-loop/).
- Three.js Discourse (Clock→Timer r183, setAnimationLoop vs RAF, coordinate-system threads) — https://discourse.threejs.org/.
- Three.js perf write-ups: utsubo.com "100 tips" (https://www.utsubo.com/blog/threejs-best-practices-100-tips), threejsroadmap.com draw calls.
- StackOverflow / Medium tutorials on PointerLock WASD and Rapier character controllers (https://medium.com/javascript-alliance/creating-a-first-person-character-controller-in-three-js-5d96534edfd8).
- Babylon.js: official optimization guide (https://doc.babylonjs.com/features/featuresDeepDive/scene/optimize_your_scene), SceneOptimizer, camera input docs; 2025 forum best-practices thread (https://forum.babylonjs.com/t/best-practices-for-optimizing-babylon-js-scenes-not-just-on-lower-end-devices/58688); Babylon 9.0 notes.
- Phaser: docs.phaser.io (Scenes, Arcade Physics), Phaser Discourse best-practices threads, franzeus.medium.com "How I optimized my Phaser 3 action game in 2025".
- Mobile/touch: nipplejs (https://github.com/yoannmoinet/nipplejs), joshmorony.com Phaser scaling guide.
- Fix Your Timestep (Gaffer On Games) — classic fixed-timestep reference.
