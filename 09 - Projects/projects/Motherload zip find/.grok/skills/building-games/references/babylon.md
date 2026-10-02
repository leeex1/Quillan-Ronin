# Babylon.js — Deep Engine Guide (Havok physics, cameras, SceneOptimizer, WebGPU, assets, GUI, freezing)

Babylon.js is the "batteries-included" 3D engine: built-in physics, inspector, GUI, node materials, camera behaviors, SceneOptimizer, and a first-class WebGPU backend. Use it when you want an integrated engine rather than assembling Three.js + libraries.

This file assumes the general loop/orientation/perf rules and the short Babylon perf bullets in `threejs-foundational.md` and goes **much deeper** without repeating them.

---

## 1. Engine setup & the render loop

```js
import { Engine, Scene, Vector3 } from "@babylonjs/core";
const engine = new Engine(canvas, true, { preserveDrawingBuffer: false, stencil: true, powerPreference: "high-performance" });
const scene = new Scene(engine);
engine.runRenderLoop(() => scene.render());
window.addEventListener("resize", () => engine.resize());
```

- `engine.runRenderLoop` is Babylon's RAF loop; hook game logic into **`scene.onBeforeRenderObservable`** and read delta with **`engine.getDeltaTime()`** (ms) or `scene.getAnimationRatio()` (frame-rate compensation factor). Scale movement by delta as always.
- Use **`scene.onBeforeRenderObservable.add(fn)`** for per-frame logic instead of stuffing everything in the render loop closure — it's observable, disposable, and composable.
- `engine.setHardwareScalingLevel(n)` renders at 1/n resolution (n>1 = lower res, big mobile win; n<1 = supersample).

---

## 2. WebGPU (real, production-ready — prefer it with fallback)

Babylon has a mature **`WebGPUEngine`** that dramatically cuts CPU-side draw-call overhead and enables compute shaders.

```js
import { WebGPUEngine, Engine } from "@babylonjs/core";
async function createEngine(canvas) {
  if (await WebGPUEngine.IsSupportedAsync) {
    const e = new WebGPUEngine(canvas, { antialias: true });
    await e.initAsync();            // REQUIRED before creating a Scene
    return e;
  }
  return new Engine(canvas, true, { powerPreference: "high-performance" });   // WebGL2 fallback
}
```

- `WebGPUEngine.initAsync()` **must be awaited** before you create the scene — a top setup bug.
- **Snapshot rendering** (`engine.snapshotRendering = true`, WebGPU only) records the command stream for a static scene and replays it, slashing CPU cost for scenes whose draw list doesn't change. Set `snapshotRenderingMode` and refresh when the scene structure changes.
- WebGPU supports **compute shaders** (particles, culling, GPU picking) — not available on WebGL.

---

## 3. Cameras & Camera Behaviors (free polish)

- **ArcRotateCamera** — orbit/inspection. Set limits and inertia so it feels good: `lowerRadiusLimit`/`upperRadiusLimit`, `lowerBetaLimit`/`upperBetaLimit`, `wheelPrecision`, `panningSensibility`, `inertia`. `camera.attachControl(canvas, true)`.
- **UniversalCamera** — FPS/free camera (WASD + mouse; combines FreeCamera + touch/gamepad). `camera.applyGravity`, `camera.checkCollisions`, `camera.ellipsoid` for simple capsule collision against meshes with `mesh.checkCollisions = true` (built-in, no physics engine needed).
- **FollowCamera** — third-person chase; set `radius`, `heightOffset`, `rotationOffset`, `cameraAcceleration`, `maxCameraSpeed`.
- **Keep `camera.maxZ` as small as practical** and `minZ` as large as practical — improves depth precision, culling, and overdraw.

**Behaviors** attach polished motion for free (this is a Babylon differentiator):
- **FramingBehavior** — auto-frames a target mesh nicely (`camera.useFramingBehavior = true`).
- **BouncingBehavior** — soft bounce at radius limits.
- **AutoRotationBehavior** — idle turntable spin (great for product/menu scenes).
Enable via `camera.useAutoRotationBehavior = true` etc., then tune the behavior object.

---

## 4. Physics — Havok (V2 plugin, the current default)

Havok is the recommended physics engine (WASM, fast, deterministic-ish). Use the **Physics V2** API (`PhysicsAggregate` / `PhysicsBody` / `PhysicsShape`).

```js
import { HavokPlugin, PhysicsAggregate, PhysicsShapeType } from "@babylonjs/core";
import HavokPhysics from "@babylonjs/havok";

const havok = await HavokPhysics();                 // MUST await — loads the .wasm
scene.enablePhysics(new Vector3(0, -9.81, 0), new HavokPlugin(true, havok));

// dynamic body:
new PhysicsAggregate(sphere, PhysicsShapeType.SPHERE, { mass: 1, restitution: 0.6, friction: 0.5 }, scene);
// static body: mass 0
new PhysicsAggregate(ground, PhysicsShapeType.BOX, { mass: 0 }, scene);
```

- **`await HavokPhysics()` is mandatory** and the #1 init error. First `HavokPlugin` arg (`_useDeltaForWorldStep`, usually `true`) makes physics use engine delta for smoother variable framerate.
- **Bundler gotcha (Vite/webpack):** the `HavokPhysics.wasm` file must be served. Use `await HavokPhysics({ locateFile: f => \`/assets/${f}\` })` or configure the bundler to copy the wasm.
- **PhysicsAggregate** is the easy path: it builds the matching `PhysicsShape` + `PhysicsBody` automatically. For fine control (compound shapes, sharing shapes across many bodies) build `PhysicsBody` + `PhysicsShape` yourself.
- Read/write motion: `aggregate.body.setLinearVelocity(v)`, `applyImpulse`, `setMassProperties`, `body.setMotionType(PhysicsMotionType.ANIMATED/DYNAMIC/STATIC)`.
- Collisions/triggers: `body.setCollisionCallbackEnabled(true)` + `body.getCollisionObservable().add(...)`; use `PhysicsShapeType.CONTAINER`/mesh shapes sparingly (mesh colliders are expensive — prefer primitives or convex hulls).
- **glTF imports:** apply physics to the actual mesh, not the `__root__` transform node — often re-parent or use the child mesh; otherwise transforms get double-applied.
- Debug: `new BABYLON.Debug.PhysicsViewer(scene).showBody(aggregate.body)`.
- Older engines (Cannon, Ammo, Oimo) still exist via the V1 plugin, but **Havok V2 is the recommended choice** for new games. Ammo remains an option if you need its specific features/soft bodies.

---

## 5. Asset loading — containers & glTF (do it right for level switching)

- **glTF/GLB is the format.** `import "@babylonjs/loaders";` then `await ImportMeshAsync("model.glb", scene)` (or `SceneLoader.ImportMeshAsync` / `AppendSceneAsync`). Use Draco + KTX2/Basis compression.
- **AssetContainer is the key tool for reusable/levels:** `LoadAssetContainerAsync(url, scene)` loads meshes/materials/textures **without adding them to the scene**. Then `container.addAllToScene()` / `container.removeAllFromScene()` and, crucially, **`container.dispose()`** to fully free GPU resources when unloading a level. This gives clean load/unload cycles and lets you **instantiate** copies:
  - `container.instantiateModelsToScene(name => name, cloneMaterials)` clones a loaded model (with skeletons/animations) cheaply for spawning many enemies from one load.
- **AssetsManager** gives progress-tracked batch loading with tasks (mesh, texture, binary) and `onProgress`/`onFinish` — good for a loading screen.
- **Babylon does not GC GPU resources** any more than Three.js does — dispose meshes/materials/textures or (better) dispose the AssetContainer when leaving a level.

---

## 6. Materials & Node Material Editor

- **PBRMaterial** for realistic; **StandardMaterial** for simpler/cheaper. Reuse material instances across meshes (shared material = fewer state changes).
- **Node Material** — visual shader graph (Node Material Editor at nme.babylonjs.com). Export as JSON and load with `NodeMaterial.ParseFromSnippetAsync("snippetId", scene)` or from file. Great for custom effects (dissolve, water, force-fields) without writing GLSL/WGSL by hand, and it compiles to both WebGL and WebGPU.
- **Material plugins** and `material.freeze()` (see freezing below) reduce per-frame shader/uniform recompute.

---

## 7. GUI (built-in — no DOM juggling)

`import { AdvancedDynamicTexture, Button, ... } from "@babylonjs/gui";`

- **Fullscreen UI:** `AdvancedDynamicTexture.CreateFullscreenUI("ui")` — a 2D overlay in screen space for HUD/menus (buttons, sliders, text, stack panels, grids). Resolution-independent, handles pointer input, works on mobile.
- **In-world UI:** apply an ADT to a mesh's material for diegetic screens/labels; use `linkWithMesh` to attach floating labels/health bars that track 3D objects.
- Prefer Babylon GUI over hand-rolled DOM overlays for game HUDs — it integrates with the scene, scales with the canvas, and doesn't fight CSS.

---

## 8. Performance — FREEZING is the Babylon superpower

For anything static, tell Babylon to stop recomputing it. This is the highest-leverage Babylon optimization and under-used in generated code.

- **`mesh.freezeWorldMatrix()`** — mesh never moves ⇒ skip world-matrix recompute each frame. Call `unfreezeWorldMatrix()` if it needs to move again. Also set `mesh.doNotSyncBoundingInfo = true` for static meshes.
- **`material.freeze()`** — material params fixed ⇒ skip uniform/shader re-evaluation. `unfreeze()` to change it.
- **`scene.freezeActiveMeshes()`** — locks the active-mesh (culling) list; use when the set of visible meshes doesn't change (e.g., fixed camera or after setup). `unfreezeActiveMeshes()` when it changes. Huge CPU savings on large static scenes.
- **`scene.blockMaterialDirtyMechanism = true`** while bulk-creating materials to avoid repeated recompiles.

Other big levers (deeper than the three.js file's bullets):
- **`scene.performancePriority = ScenePerformancePriority.Intermediate` (or `Aggressive`)** — auto-applies a bundle of optimizations (freezes active meshes, skips some checks). `Aggressive` disables per-mesh picking/some features — verify nothing you need breaks.
- **Thin instances** (`mesh.thinInstanceAdd(matrix)` / `thinInstanceSetBuffer`) for thousands of identical static objects → one draw call, cheaper than regular instances. Regular **instances** (`mesh.createInstance()`) when you need per-instance picking/parenting.
- **`scene.autoClear = false`** (and `autoClearDepthAndStencil`) when the scene fully covers the viewport — skips the clear.
- **`scene.skipPointerMovePicking = true`** if you don't need hover/move picking (picking on every pointer move is a silent CPU cost). Set `mesh.isPickable = false` on non-interactive meshes.
- **`scene.skipFrustumClipping`**, `mesh.alwaysSelectAsActiveMesh` — micro-tune culling for known-visible meshes.
- **Merge meshes** (`Mesh.MergeMeshes([...], true, true, undefined, false, true)`) for static geometry sharing a material.
- **LOD:** `mesh.addLODLevel(distance, lowMesh)` and `addLODLevel(farDist, null)` to cull entirely far away.
- Keep **`camera.maxZ` low**; limit lights (each real-time light multiplies shader cost); use **shadow generators sparingly** with small map sizes and `RefreshRate` for static lights.
- Avoid per-frame allocations: use **`TmpVectors.Vector3[0]`** scratch vectors and the `...ToRef()` method variants (`addToRef`, `scaleToRef`) instead of returning new objects.

---

## 9. SceneOptimizer — adaptive quality to hit a target FPS

Instead of guessing settings per device, let Babylon degrade quality until it reaches a target framerate.

```js
import { SceneOptimizer, SceneOptimizerOptions } from "@babylonjs/core";
// preset tiers try progressively harder optimizations to reach the target FPS:
SceneOptimizer.OptimizeAsync(scene,
  SceneOptimizerOptions.HighDegradationAllowed(60),   // target 60 fps
  () => console.log("reached target"),
  () => console.log("could not reach target"));
```

- Presets: `LowDegradationAllowed`, `ModerateDegradationAllowed`, `HighDegradationAllowed(targetFps)`. They apply, in escalating order, optimizations like: reduce hardware scaling, merge meshes, disable shadows, reduce texture size, disable post-processes, cut particle counts.
- You can build a custom `SceneOptimizerOptions` with specific `SceneOptimization` steps and priorities. Great for shipping one build that adapts from low-end mobile to desktop automatically.

---

## 10. Inspector & profiling

- **`import "@babylonjs/inspector"; scene.debugLayer.show()`** — the in-browser inspector: scene tree, per-mesh stats, materials, textures, and a **Statistics/Performance** tab (draw calls, active meshes, frame breakdown, GPU/CPU time). This is the fastest way to find what's slow.
- Watch **draw calls** and **active meshes** counts; use the profiler to see whether you're CPU-bound (draw calls, matrix updates → freeze/instance) or GPU-bound (overdraw, shader cost, texture size → LOD/compression/fewer lights).

---

## 11. Common Babylon pitfalls (checklist)

- Forgetting `await WebGPUEngine.initAsync()` / `await HavokPhysics()` → cryptic init failures.
- Havok `.wasm` not served by the bundler → physics silently absent; use `locateFile` / copy the wasm.
- Not disposing meshes/materials/textures or the AssetContainer on level switch → GPU memory leak.
- Applying physics to a glTF `__root__` node → double transforms / wrong positions.
- Static scene running full cost because nothing is frozen → freeze world matrices, materials, active meshes.
- Picking on every pointer move (`skipPointerMovePicking` not set; meshes left `isPickable`) → CPU drain.
- Too many real-time lights / large shadow maps → shader + fill cost explosion.
- Per-frame `new Vector3()`/`Matrix` allocations → GC churn; use `TmpVectors` + `...ToRef()`.
- Huge `camera.maxZ` → z-fighting and wasted culling range.
- Mesh colliders for everything instead of primitive/convex shapes → physics perf collapse.

---

## Defaults to apply

- **Reach for Babylon when the game wants an integrated engine** (built-in physics, GUI, inspector, camera behaviors) rather than the assemble-it-yourself Three.js route.
- **Always await `initAsync()` (WebGPU) and `HavokPhysics()`**, and wire the **WebGPU-with-WebGL2-fallback** pattern by default.
- **Emit freezing by default for static content**: `freezeWorldMatrix()` on non-moving meshes, `material.freeze()` on static materials, and `scene.freezeActiveMeshes()` / `performancePriority` once the scene is set up. This is the single biggest Babylon win most generated code misses.
- **Ship `SceneOptimizer.OptimizeAsync(scene, HighDegradationAllowed(60))`** so one build adapts from low-end mobile to desktop instead of hardcoding quality.
- **Use AssetContainers** for level load/unload (`instantiateModelsToScene` to spawn many from one load; `dispose()` to free on unload) to avoid the memory leaks that plague generated 3D games.
- **Use Babylon GUI** (fullscreen ADT) for HUDs/menus instead of DOM overlays; use **camera Behaviors** (Framing/AutoRotation) for free menu/product polish.
- Set **`skipPointerMovePicking`, low `maxZ`, `isPickable=false`** on non-interactive meshes as defaults.

---

## Sources
- Babylon.js docs — Optimizing your scene (freeze APIs, autoClear, performancePriority, active meshes): https://doc.babylonjs.com/features/featuresDeepDive/scene/optimize_your_scene
- Babylon.js docs — Physics V2 / Havok (`enablePhysics`, `PhysicsAggregate`, `PhysicsBody/Shape`): https://doc.babylonjs.com/features/featuresDeepDive/physics/usingPhysicsEngine and https://github.com/BabylonJS/Documentation/blob/master/content/features/featuresDeepDive/physics/v2/usingPhysicsEngine.md
- `@babylonjs/havok` npm (init pattern, `await HavokPhysics()`, `locateFile`): https://www.npmjs.com/package/@babylonjs/havok
- Babylon.js docs — WebGPU (`WebGPUEngine`, `initAsync`, snapshot rendering): https://doc.babylonjs.com/setup/support/webGPU
- Babylon.js docs — Cameras & Camera Behaviors: https://doc.babylonjs.com/features/featuresDeepDive/cameras/camera_introduction and https://doc.babylonjs.com/features/featuresDeepDive/behaviors/cameraBehaviors
- Babylon.js docs — Asset Containers & glTF loading: https://doc.babylonjs.com/features/featuresDeepDive/importers/assetContainers and https://doc.babylonjs.com/features/featuresDeepDive/importers/loadingFileTypes
- Babylon.js docs — SceneOptimizer: https://doc.babylonjs.com/features/featuresDeepDive/scene/sceneOptimizer
- Babylon.js docs — GUI: https://doc.babylonjs.com/features/featuresDeepDive/gui/gui ; Node Material: https://doc.babylonjs.com/features/featuresDeepDive/materials/node_material/nodeMaterial
- Babylon.js docs — Inspector: https://doc.babylonjs.com/toolsAndResources/inspector
- Babylon.js forum — optimization best practices thread: https://forum.babylonjs.com/t/best-practices-for-optimizing-babylon-js-scenes-not-just-on-lower-end-devices/58688
