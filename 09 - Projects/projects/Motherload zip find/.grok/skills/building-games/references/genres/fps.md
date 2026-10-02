# Genre Playbook — First-Person Shooter (FPS)

How to build a browser FPS that *feels* right (fast, snappy, weighty) and avoids the classic bugs. Assumes Three.js/Babylon.js. Read `../threejs-foundational.md` first — this file only adds FPS-specific rules (it already covers pointer lock + WASD basics, delta time, disposal, draw calls).

---

## 1. Core mechanics (minimal-but-good scope for a demo)

A convincing FPS demo needs, in priority order:
1. **Mouse-look + WASD movement** with pointer lock (yaw/pitch, clamped pitch).
2. **One weapon that shoots** with a crosshair, muzzle flash, and hit feedback.
3. **Something to shoot** — 3–8 targets or simple enemies with a hit reaction + death.
4. **A hit indicator + score/ammo HUD.** Feedback is the game.
5. **A bounded arena** with walls/cover and collision so you can't walk through geometry.

Do NOT try to ship AI pathfinding, multiple weapons with full inventory UI, reload animations, or netcode in a first demo. A single hitscan weapon + stationary/patrolling targets + juice (screen shake, flash, sound) reads as a real game.

---

## 2. Controls & camera

- **Pointer Lock** is mandatory (see MDN below). Request it from a **user gesture** (`canvas.addEventListener('click', () => canvas.requestPointerLock())`), show a "Click to play" blocker, and re-show it on `pointerlockchange` when `document.pointerLockElement === null`. Prefer `requestPointerLock({ unadjustedMovement: true })` to disable OS mouse acceleration for consistent aim; **fall back gracefully** if the returned Promise rejects with `NotSupportedError`.
- **Read `movementX`/`movementY` on `mousemove`, not `clientX/Y`.** While locked, `clientX/Y` are frozen; only the deltas update. `yaw -= movementX * sensitivity; pitch -= movementY * sensitivity`.
- **Clamp pitch to just under ±90°** (`±89°`, i.e. `Math.PI/2 - 0.01`) so the camera never flips/gimbal-locks looking straight up/down.
- **Yaw the body (a container), pitch the camera.** Standard rig: a yaw `Object3D` holding the camera; apply yaw to the container, pitch to the camera. Movement (`moveForward/moveRight`) uses body yaw only — never pitch — so looking up doesn't make you fly.
- **Sensitivity + optional ADS zoom:** store a base sensitivity; when aiming-down-sights, lower FOV *and* scale sensitivity down proportionally so aim feels consistent.
- **Add ESC-to-unlock** and pause. Only re-attach the `mousemove` handler while locked; detach on unlock so a paused game doesn't rotate.
- **Movement feel:** acceleration + friction (not instant velocity), add sprint (Shift), crouch, and subtle **head-bob** driven by a sin wave of distance traveled (disable when idle). Optional: coyote-style step-up for small ledges. Air control should be reduced vs ground.
- **FOV:** 75–90° vertical feels good for FPS; too low feels claustrophobic, too high distorts. Let the player adjust.

---

## 3. Weapon handling & the viewmodel

### ⚠️ Do NOT use a generated PHOTO as the weapon viewmodel.
The single most common failure mode: the builder generates an **opaque JPG/PNG photo of a gun** and slaps it as a flat sprite/plane in the bottom-right of a 3D scene. It looks *wrong* — no parallax, wrong perspective, a hard rectangular edge, lighting that doesn't match the world, and it z-fights or clips into walls. **The world is 3D; the gun must be 3D too.**

**Instead:**
- **Build the viewmodel from 3D geometry/code** — a few boxes/cylinders (`BoxGeometry`, `CylinderGeometry`) assembled into a low-poly gun, or a proper **glTF/GLB model** (right-handed, Y-up; verify +Z-forward orientation). Even a crude boxy gun made of primitives reads far better than a photo because it has real perspective and lighting.
- **Parent the viewmodel to the camera** (add it as a child of the camera, offset to e.g. `(0.3, -0.3, -0.6)`) so it tracks the view. Because the camera's local forward is **−Z**, the gun sits at negative Z in front of the camera.
- **Render it so it never clips into walls.** Two robust options: (a) a **separate overlay scene + second camera** rendered after the main scene with `renderer.autoClear=false` and `renderer.clearDepth()` between passes, or (b) put the viewmodel on its own **layer** with a dedicated camera. This guarantees the gun draws on top and never intersects level geometry.
- If you must use a texture, it must be a **transparent PNG on correctly-perspectived geometry**, never a raw opaque photo on a screen-aligned quad.

### Shooting model
- **Hitscan (raycast) for fast bullets** (pistols/rifles): on fire, `raycaster.setFromCamera({x:0,y:0}, camera)` (center of screen = crosshair) and `intersectObjects(targets)`. Take the nearest hit; apply damage. Instant, cheap, and what most shooters use.
- **Projectiles for slow/arcing shots** (rockets, grenades): spawn a pooled mesh, move by velocity·delta, integrate gravity, and sweep-test for collisions (avoid tunneling — see bugs).
- **Fire rate:** gate with a cooldown timer (`if (now - lastShot < fireInterval) return`), not per-frame or per-mousedown-event alone. Support hold-to-fire for automatics via a boolean set on mousedown/up.
- **Recoil & spread:** kick the camera pitch/yaw up slightly per shot and recover over time (lerp back); add a small random cone to the ray direction that grows while firing continuously and shrinks when idle. This is 80% of "feel."
- **Ammo + reload:** track `magazine`/`reserve`; block firing at 0; a timed reload that refills. Even a fake reload with a timer + HUD reads well.

---

## 4. Genre-specific "feel" (the juice)

FPS feel is almost entirely feedback and responsiveness:
- **Muzzle flash** (a brief additive sprite/light at the barrel, 1–2 frames), **shell/particle**, and a **tracer** for projectiles.
- **Screen shake** on fire and on taking damage (small, decaying camera offset).
- **Hitmarker** (crosshair flashes / an X appears) + **hit sound** the instant a shot lands — the game must confirm every hit.
- **Enemy hit reaction**: flash the material, knockback, a death animation/ragdoll or a satisfying pop.
- **Weapon sway & bob**: viewmodel lags slightly behind fast mouse movement (lerp toward target offset) and bobs while walking.
- **Sound is half the feel**: distinct fire, impact, reload, footstep, and empty-click sounds; unlock the audio context on first gesture.
- **Snappy input**: never add input lag; process shooting on the event but resolve in the fixed/render step. Aim assist is optional on mobile/gamepad.

---

## 5. Common bugs to avoid (checklist)

- **Photo-as-viewmodel** (see §3) → use 3D geometry/model, render as an overlay layer.
- **Reading `clientX/Y` instead of `movementX/Y`** while locked → camera doesn't rotate.
- **Pitch not clamped** → camera flips upside down at the poles.
- **Applying pitch to movement** → you "fly" when looking up. Movement uses yaw only.
- **Firing tied to frame rate / no cooldown** → 144Hz machine empties the mag instantly.
- **Projectile tunneling** through thin walls at high speed → use raycast sweep between last and current position, or a fixed timestep + continuous collision, not just point-in-frame checks.
- **Raycasting from the mouse position instead of screen center** → shots miss the crosshair. Use `{x:0, y:0}` NDC (center).
- **Viewmodel clips into walls / z-fights** → separate overlay render pass or dedicated layer + `clearDepth()`.
- **Pointer lock never re-engages after ESC** → you didn't re-show the blocker / re-bind click on `pointerlockchange`.
- **Not disposing enemies/bullets** → memory leak; pool projectiles and enemies.
- **No collision** → walking through walls/floor. Add a capsule/AABB character collider (Rapier `KinematicCharacterController`).
- **Mobile:** pointer lock is unsupported on iOS/most mobile — provide touch look/joystick + fire button fallback and don't hard-require lock.

---

## Defaults to apply

Concrete rules to apply:

1. **NEVER render a weapon as a flat generated photo/JPG in a 3D FPS.** Build the viewmodel from Three.js primitives or load a glTF model, parent it to the camera, and render it as a separate overlay pass/layer so it never clips. A boxy code-built gun > a photorealistic sprite.
2. **Pointer lock is mandatory and gesture-gated:** click-to-play blocker, `requestPointerLock({ unadjustedMovement:true })` with fallback, read `movementX/Y`, clamp pitch to ±89°, re-show blocker on unlock.
3. **Separate yaw (body) from pitch (camera); movement uses yaw only.**
4. **Default to hitscan** with a raycast from screen center; gate fire with a cooldown timer; add recoil + spread + recovery. Projectiles only for slow/arcing weapons, with sweep tests to prevent tunneling.
5. **Ship juice, not features:** muzzle flash, hitmarker, hit sound, screen shake, enemy flash-on-hit. This makes a 1-weapon demo feel like a game.
6. **Minimal scope:** one weapon, a handful of targets/enemies, an arena with collision, and a HUD (ammo + score). Pool bullets/enemies; dispose on cleanup.
7. **Provide a mobile fallback** (touch look + fire button) since pointer lock is desktop-only.

---

## Sources
- MDN — Pointer Lock API: https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API
- MDN — `Element.requestPointerLock()` (options, `unadjustedMovement`, Promise): https://developer.mozilla.org/en-US/docs/Web/API/Element/requestPointerLock
- MDN — pointer lock live demo: https://mdn.github.io/dom-examples/pointer-lock/
- Three.js — `PointerLockControls` example: https://threejs.org/examples/misc_controls_pointerlock.html
- Three.js — `Raycaster` (`setFromCamera`, `intersectObjects`): https://threejs.org/docs/#api/en/core/Raycaster
- Three.js — layered/overlay rendering (`autoClear`, `clearDepth`, `Layers`): https://threejs.org/docs/#api/en/core/Layers
- Rapier — Kinematic character controller: https://rapier.rs/docs/user_guides/javascript/character_controller
- Gaffer On Games — Fix Your Timestep (projectile/physics stability): https://gafferongames.com/post/fix_your_timestep/
