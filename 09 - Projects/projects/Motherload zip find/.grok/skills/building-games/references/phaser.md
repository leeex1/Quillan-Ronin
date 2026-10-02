# Phaser 3 / 4 — Deep Engine Guide (scenes, physics, scale, tilemaps, pooling, architecture)

Phaser is the default choice for **2D browser games**. This file goes deep on the parts that actually break in generated games. It assumes the general loop/mobile/perf rules in `threejs-foundational.md` and does **not** repeat them.

> **Version note (verified):** Phaser **3** is the mature, ubiquitous line (3.90+). Phaser **4** shipped in 2025 with a rewritten WebGL renderer (internally "Beam") and optional **WebGPU** path, plus the core split into modular packages. Phaser 4 is *intentionally near-API-compatible with Phaser 3* — the Scene lifecycle, Arcade/Matter physics, Scale Manager, Loader and GameObject APIs described here carry over. Prefer Phaser 3 (3.90.x) for maximum stability/plugin compatibility today; use Phaser 4 when you want the faster renderer/WebGPU and can tolerate a younger ecosystem. When unsure, target v3 API surface — it runs on both.

---

## 1. Scene model — the thing to get right first

A Phaser game is a **stack of Scenes**, each an almost self-contained world (own display list, cameras, input, tweens, physics, clock). The `Phaser.Game` instance owns only truly global systems: **Renderer, Animation Manager, global Cache, Registry (global DataManager), Input Manager, Scene Manager, Sound Manager, TimeStep**. Everything else (`this.add`, `this.tweens`, `this.physics`, `this.input`, `this.cameras`) is **per-scene**. A tween created in Scene A is unrelated to one in Scene B.

### Lifecycle callbacks (exact order)

```js
class GameScene extends Phaser.Scene {
  constructor() { super('game'); }          // unique string key
  init(data) {}      // (1) reset per-run state HERE, receives data from start/launch
  preload() {}       // (2) queue asset loads only
  create(data) {}    // (3) build objects; assets from preload are now ready
  update(time, delta){} // (4) every tick while RUNNING (delta in ms)
}
```

Status flow: `PENDING → INIT (booted) → START → LOADING → CREATING → RUNNING`. `RUNNING` can go to `PAUSED` (renders, no update) or `SLEEPING` (no update, no render), then back. `stop()` → `SHUTDOWN` (can restart). `remove()` → `DESTROYED` (gone).

### The scene stack pattern (best-practice architecture)

Run **multiple scenes at once**, layered bottom→top by config order:

- **Boot** — load the tiny assets needed for the loading screen (bar image, logo); set Scale/registry defaults; `this.scene.start('preload')`.
- **Preload** — load everything with a visible progress bar (`this.load.on('progress', v => …)`), then `start('menu')`.
- **Menu / Title** — UI, start button.
- **Game** — actual gameplay.
- **UI / HUD** — run **in parallel** over Game via `this.scene.launch('ui')`; keeps HUD stable while Game restarts. Communicate via events/registry, never direct references.
- **Pause / GameOver** — overlay scenes launched over Game.

### Scene control verbs (get these exactly right — a top bug source)

| Method | Effect on target | Effect on caller |
|---|---|---|
| `start('k')` | **stops** then starts k | **stops** caller |
| `launch('k')` | starts k (parallel) | caller keeps running |
| `switch('k')` | starts or **wakes** k | **sleeps** caller |
| `run('k')` | resume if paused / wake if sleeping / restart if running / else start | caller keeps running |
| `pause`/`resume` | freeze update, keep render | — |
| `sleep`/`wake` | freeze update + render | — |
| `stop` | shutdown | — |

Rules of thumb: gameplay you replay from scratch → **start/stop** or **sleep/wake**; a modal (pause menu, shop) over live gameplay → **pause/resume** or **launch** an overlay; menus you revisit → **sleep/wake** is easier to reason about than start/stop.

### The #1 Phaser correctness bug: state not reset on restart

Scenes are **booted once, started many times**. Module-level or constructor-set flags persist across restarts.

```js
// BROKEN: gameOver stays true after restart → instant game over
class S extends Phaser.Scene { constructor(){ super('s'); this.gameOver = false; } }

// CORRECT: reset run state in init()
class S extends Phaser.Scene {
  constructor(){ super('s'); }
  init(){ this.gameOver = false; this.score = 0; }   // runs on every start
}
```

Related: arrays of destroyed game objects survive restart. **Clean up on SHUTDOWN**:
```js
this.events.once('shutdown', () => { this.enemies.length = 0; });
```

### Cleanup rule

On `shutdown`, Phaser auto-destroys the scene's display list, tweens, timers and input. But **external references, global timers, DOM listeners, EventEmitter subscriptions on cross-scene emitters, and pooled arrays are yours to clear**. A destroyed GameObject still registered on an emitter is a classic crash. Always `this.events.once('shutdown', cleanup)`.

---

## 2. Physics: Arcade vs Matter (choose deliberately)

Phaser ships two physics systems. **Pick Arcade unless you specifically need Matter** — mixing/over-choosing is a common performance and complexity mistake.

### Arcade Physics — AABB, fast, for platformers/shooters/arcade

- **Only axis-aligned bounding boxes (rectangles) and circles.** No rotation of bodies, no slopes, no polygons. Extremely fast; handles thousands of bodies.
- Two body types: **Dynamic** (`this.physics.add.sprite`) responds to velocity/gravity/collisions; **Static** (`this.physics.add.staticGroup`) never moves (platforms, walls) and is cheaper.
- Core calls: `body.setVelocity(x,y)`, `setGravityY`, `setBounce`, `setCollideWorldBounds(true)`, `setImmovable(true)`, `setAllowGravity(false)`.
- **Collide vs Overlap:**
  - `this.physics.add.collider(a, b, cb)` — separates bodies (they block each other), optional callback.
  - `this.physics.add.overlap(a, b, cb)` — detects overlap without separation (pickups, hitboxes, triggers).
  - Both accept Groups, arrays, or single objects; add a `processCallback` (4th arg) to conditionally allow/deny a collision.
- Config: `physics: { default: 'arcade', arcade: { gravity: { y: 300 }, debug: true } }`. Turn on `debug` while building — it draws body outlines and velocity vectors and reveals 90% of "collision doesn't work" issues.
- Set body size explicitly for non-rectangular sprites: `sprite.body.setSize(w,h).setOffset(x,y)` or `setCircle(r)`.
- **Fast small objects tunnel through thin walls** (AABB, discrete step). Mitigate: thicker colliders, cap max velocity, or set world bounds; Arcade has limited continuous collision.

### Matter.js — full rigid-body, for physics-toys/ragdolls/complex shapes

- Real rigid-body dynamics: rotation, arbitrary convex/compound polygons, constraints/joints, springs, restitution, friction, sleeping bodies.
- Use when the *physics itself is the game* (Angry-Birds-like, stacking, contraptions, vehicles) or you need realistic collisions/rotation.
- Config: `physics: { default: 'matter', matter: { gravity: { y: 1 }, debug: true } }`.
- `this.matter.add.sprite(x,y,key,null,{ shape:'circle', restitution:0.6 })`. Use collision events: `this.matter.world.on('collisionstart', (e)=>{ for (const p of e.pairs){…} })`, plus **collision filters/categories** for what hits what.
- Heavier than Arcade; watch body counts on mobile. Enable body sleeping for idle stacks.

**Decision:** platformer, top-down shooter, endless runner, breakout, most casual games → **Arcade**. Physics sandbox, realistic stacking/rotation, joints → **Matter**. Don't reach for Matter just for "better collisions" — 90% of 2D games are Arcade games.

---

## 3. Scale Manager — responsive/mobile without stretching

Set once in game config. The two modes that matter:

```js
scale: {
  mode: Phaser.Scale.FIT,          // letterbox: keep aspect, fit inside parent
  autoCenter: Phaser.Scale.CENTER_BOTH,
  width: 800, height: 600,         // your fixed design resolution
  parent: 'game',
  // min/max clamp the FIT scaling:
  min: { width: 400, height: 300 }, max: { width: 1600, height: 1200 }
}
```

- **`FIT`** — canvas scales to fit the parent preserving aspect ratio; letterbox bars appear. Your game logic runs at a **fixed design resolution** (800×600 here) — simplest and most predictable. Best default for most games.
- **`RESIZE`** — canvas always matches parent size; **no fixed resolution**, game world dimensions change. You must handle the `resize` event and reposition/relayout UI yourself (`this.scale.on('resize', (gameSize)=>{…})`). Use for games that should truly fill the screen (strategy, infinite canvas) where you can reflow layout.
- `ENVELOP` — cover the parent (may crop). `WIDTH_CONTROLS_HEIGHT` / `HEIGHT_CONTROLS_WIDTH` — lock one axis.
- `NONE` — fixed, no scaling.

**Orientation:** listen `this.scale.on('orientationchange', o => …)` or check `this.scale.isPortrait`. For a landscape-only game, show a "rotate your device" overlay in portrait rather than fighting the layout. Combine with `viewport` meta + `touch-action: none` (see three.js file §7).

**Retina/DPR:** Phaser respects `resolution`/DPR via the renderer; for pixel-art keep `pixelArt: true` (sets nearest-neighbor + disables antialias) and `roundPixels: true` to avoid sub-pixel shimmer. Rendering at a modest internal resolution and letting FIT upscale is a legit perf win on mobile.

---

## 4. Sprites, atlases & animations

- **Always pack sprites into a texture atlas** (TexturePacker → JSON Hash/Array, or free tools). One atlas = one texture bind = fewer draw calls and no seams. `this.load.atlas('sheet','sheet.png','sheet.json')`, then `this.add.sprite(x,y,'sheet','frameName')`.
- Uniform grid frames → `this.load.spritesheet('run','run.png',{ frameWidth:32, frameHeight:32 })`.
- Animations are **global** (stored on the Animation Manager), defined once, reused by any sprite in any scene:
  ```js
  this.anims.create({ key:'run', frames:this.anims.generateFrameNames('sheet',{prefix:'run_',start:0,end:7,zeroPad:2}), frameRate:12, repeat:-1 });
  sprite.play('run');
  ```
- Batching: sprites sharing the **same texture** batch into few draw calls. Interleaving textures (sprite from atlas A, then B, then A) breaks the batch. Group same-texture sprites and set depth thoughtfully.
- Bitmap fonts (`load.bitmapFont`) render far cheaper than lots of dynamic `Text` objects (each `Text` is its own canvas texture — expensive to update every frame). For frequently-changing text prefer bitmap fonts or update sparingly.

---

## 5. Tilemaps

For levels, use **Tiled** (`.tmx`/exported JSON) rather than hand-placing sprites.

```js
// preload
this.load.image('tiles','tileset.png');
this.load.tilemapTiledJSON('map','level1.json');
// create
const map = this.make.tilemap({ key:'map' });
const tileset = map.addTilesetImage('tilesetNameInTiled','tiles');
const ground = map.createLayer('Ground', tileset, 0, 0);
ground.setCollisionByProperty({ collides:true });        // set per-tile in Tiled
this.physics.add.collider(player, ground);
// object layer for spawns/enemies:
const objs = map.getObjectLayer('Objects').objects;
```

- **Static layers** (`createLayer`) are fast (culled, batched). Dynamic tile edits use the same layer API (`putTileAt`, `removeTileAt`) — layers are dynamic in current Phaser.
- Set collision by **property** or by tile index/array; visualize with `layer.renderDebug(graphics)`.
- Cull off-screen tiles automatically via the camera; keep tile layers reasonably sized, split huge worlds into chunks.
- Use **object layers** for spawn points, triggers, and entity placement instead of hardcoding coordinates.

---

## 6. Object pooling with Groups (the mobile perf lever)

Creating/destroying bullets, enemies, particles every frame thrashes GC and stutters. **Pool them.**

```js
this.bullets = this.physics.add.group({
  classType: Bullet, maxSize: 64, runChildUpdate: true
});
// fire:
const b = this.bullets.get(x, y);        // reuse dead one or make new (up to maxSize)
if (!b) return;                          // pool exhausted → skip
b.enableBody(true, x, y, true, true);    // reactivate + show
// on expire/off-screen: DON'T destroy — recycle:
b.disableBody(true, true);               // deactivate + hide, returns to pool
```

- `get()` returns an **inactive** member (recycled) or creates a new one until `maxSize`; returns `null` when full — always null-check.
- `killAndHide(child)` / `disableBody(true,true)` recycles; avoid `destroy()` for pooled objects.
- `runChildUpdate:true` calls each active child's `preUpdate`/`update` — put per-object logic in the class.
- Only iterate **active** members: `this.bullets.getChildren()` and skip `!child.active`, or use `group.getMatching('active', true)`.
- Reuse tween/particle emitters too; destroy emitters and tweens on `SHUTDOWN`.

---

## 7. Input

- Keyboard: `this.cursors = this.input.keyboard.createCursorKeys()` (arrows + space/shift), or `this.keys = this.input.keyboard.addKeys('W,A,S,D')`. **Read state in `update` via `.isDown`**; use `Phaser.Input.Keyboard.JustDown(key)` for single-press actions. Don't drive movement off `keydown` events.
- Pointer/touch: `this.input.on('pointerdown', p => …)`; enable object interaction with `sprite.setInteractive()` then `sprite.on('pointerdown', …)`. Set explicit hit areas for irregular sprites.
- Unify keyboard + touch + **Gamepad API** (`this.input.gamepad`) into one input-state object your `update` reads, so control code doesn't branch per device.
- Drag: `this.input.setDraggable(sprite)` + `'drag'` event. For virtual joysticks on mobile, integrate nipplejs or the rex virtual joystick plugin and feed normalized -1..1 into your input state.

---

## 8. Cameras & tweens

- Each scene has a main camera (`this.cameras.main`). Follow the player: `camera.startFollow(player, true, 0.08, 0.08)` (lerp for smooth follow), set `camera.setBounds(0,0,mapW,mapH)` and `setZoom()`. `setDeadzone()` avoids jitter around the player.
- Effects for free polish: `camera.shake(200, 0.01)`, `camera.flash()`, `camera.fade()`. Multiple cameras enable split-screen / minimaps (`camera.ignore(objects)` to exclude).
- Tweens for juice: `this.tweens.add({ targets:sprite, scale:1.2, yoyo:true, duration:120, ease:'Sine.easeInOut' })`. **Always destroy long-lived/looping tweens on SHUTDOWN** or they reference dead objects.

---

## 9. Cross-scene communication (avoid spaghetti)

- **Registry** (global DataManager): `this.registry.set('score', 0)` / `get` — shared across all scenes; emits `changedata` events.
- **Scene events / a shared EventEmitter** ("EventsCenter"): create one `new Phaser.Events.EventEmitter()` in a module, import into scenes, emit/subscribe. Lets the UI scene react to Game scene events with **zero direct coupling**. Always `off()` listeners on shutdown.
- `this.scene.get('ui').events.emit(...)` works but couples scenes — prefer the shared emitter/registry.
- Pass one-shot data via `this.scene.start('game', { level: 3 })` → arrives in `init(data)` / `create(data)`.

---

## 10. Performance checklist (Phaser-specific)

- **Object pooling** for anything spawned repeatedly (bullets, enemies, particles, floating text).
- **Texture atlases** everywhere; group same-texture draws to keep batches intact.
- Turn **off physics debug** in production; only add bodies to objects that need them.
- Process **only active** objects; set `sprite.active=false`/`visible=false` for off-screen entities instead of updating them.
- Prefer **static** physics bodies / static tile layers for non-moving geometry.
- Minimize per-frame `Text` updates and `Graphics` re-draws (redrawing a Graphics every frame is expensive) — cache to a texture (`generateTexture`) or use bitmap fonts.
- Avoid per-frame allocations in `update`; reuse vectors/objects.
- Destroy tweens, timers (`this.time` events), and particle emitters on `SHUTDOWN`; clear pooled arrays.
- Lazy-load per-level assets in that level's preload; unload with `this.textures.remove` / `this.cache` when leaving big levels.
- Smaller canvas + FIT upscale on low-end mobile; consider `Phaser.CANVAS` fallback only if WebGL is unavailable (WebGL is default and much faster).
- Profile with the browser FPS meter and `game.loop.actualFps`; Phaser 4's WebGPU path reduces CPU draw-call overhead further.

---

## 11. Common Phaser pitfalls (checklist)

- Run-state not reset in `init()` → instant game-over / carried-over score after restart.
- Not cleaning arrays/emitter listeners on `shutdown` → crashes on destroyed objects after restart.
- Using `this.scene.start` when you meant `launch` (start stops the caller — kills your HUD).
- Movement/animation not scaled by `delta` (Phaser `update(time, delta)` gives delta in **milliseconds**).
- Physics body doesn't match sprite art → set `body.setSize/offset`; wrong/no collider group; forgetting `setCollideWorldBounds`.
- Fast objects tunneling through thin walls in Arcade → thicker walls / velocity cap.
- Text objects updated every frame tanking FPS → bitmap fonts / update on change only.
- Creating instead of pooling bullets/enemies → GC stutter on mobile.
- `RESIZE` scale mode without repositioning UI → misaligned HUD; use `FIT` unless you handle `resize`.
- Loading assets in `create` instead of `preload` → assets not ready / race conditions.
- Mixing Matter + Arcade needlessly, or choosing Matter when Arcade suffices → perf and complexity cost.

---

## Defaults to apply

- **Default to Phaser (v3 API) for any 2D game**; scaffold the **Boot → Preload(progress bar) → Menu → Game → parallel UI → Pause/GameOver** scene stack automatically.
- **Always put per-run state reset in `init()`** and register a `this.events.once('shutdown', cleanup)` in every gameplay scene — this alone eliminates the most common generated-game bug (broken restart).
- **Choose Arcade physics by default; only emit Matter when the physics is the game.** Encode the decision table above in the builder's engine-selection step.
- **Emit object pools (Groups with `maxSize` + recycle via `disableBody`) for bullets/enemies/particles** rather than create/destroy, and **texture atlases** for all sprites.
- Use **`Scale.FIT` + fixed design resolution + `CENTER_BOTH`** as the responsive default; add a rotate-device overlay for orientation-locked games.
- Load levels from **Tiled JSON tilemaps** with collision-by-property and object layers for spawns, instead of hardcoded coordinates.
- Decouple scenes via a **shared EventEmitter + Registry**, never direct scene references.
- Turn **physics `debug` on during generation/testing, off in the shipped build.**

---

## Sources
- Phaser official docs — Scenes (lifecycle, systems, control methods, restart pitfalls): https://docs.phaser.io/phaser/concepts/scenes
- Phaser docs — Arcade Physics: https://docs.phaser.io/phaser/concepts/physics/arcade ; Matter Physics: https://docs.phaser.io/phaser/concepts/physics/matter
- Phaser docs — Scale Manager: https://docs.phaser.io/phaser/concepts/scale-manager
- Phaser docs — Tilemaps: https://docs.phaser.io/phaser/concepts/tilemaps ; Loader: https://docs.phaser.io/phaser/concepts/loader
- Phaser docs — Groups & object pooling: https://docs.phaser.io/phaser/concepts/gameobjects/group ; Animations: https://docs.phaser.io/phaser/concepts/animations
- Phaser docs — Input (keyboard/pointer/gamepad): https://docs.phaser.io/phaser/concepts/input
- Phaser 4 announcement / release: https://phaser.io/news (Phaser 4 + Beam renderer / WebGPU) and https://github.com/phaserjs/phaser/releases
- API reference: https://newdocs.phaser.io/docs/latest
- franzeus.medium.com — "How I optimized my Phaser 3 action game" (pooling, atlases, active-only processing)
- joshmorony.com — Phaser scaling / responsive guides
