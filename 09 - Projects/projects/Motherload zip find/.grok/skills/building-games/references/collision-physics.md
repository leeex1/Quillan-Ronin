# Collision & Physics for Browser Games (2D AABB/SAT/broadphase, swept collision, 3D Rapier/cannon, fixed timestep)

Consolidated from MDN game collision docs, Gaffer On Games, Rapier/cannon-es docs, and the SAT/swept-AABB canon (see Sources). Focus: what an AI builder needs so collisions are **correct and stable** — the classic bugs are tunneling through walls at speed, jitter, sinking into the ground, and O(n²) slowdowns.

---

## 1. Do you even need a physics engine?

- **Simple arcade games** (platformers, top-down, breakout, shooters) → hand-rolled **AABB** collision + resolution is often better: fully deterministic, no dependency, and you control feel. Most 2D games don't need Box2D/Matter.
- **Rigid-body simulation** (stacking, ragdolls, joints, realistic bouncing, vehicles) → use an engine: **2D → Matter.js / Planck.js (Box2D) / Rapier2D**; **3D → Rapier (rapier.js, WASM, fast, favored) / cannon-es / Ammo.js**.
- **Rule:** don't pull in a full rigid-body engine for a game that just needs "does the player box overlap the wall box." Physics engines add non-determinism and tuning overhead. Match the tool to the game.

---

## 2. 2D collision detection primitives

- **AABB (axis-aligned bounding box)** — the workhorse. Two boxes overlap iff they overlap on **both** axes:
  ```js
  const hit = a.x < b.x+b.w && a.x+a.w > b.x && a.y < b.y+b.h && a.y+a.h > b.y;
  ```
  Cheap, ideal for tile maps, most 2D gameplay. Only valid for **non-rotated** boxes.
- **Circle/distance** — for round objects: compare squared distance to summed radii (`dx*dx+dy*dy < (r1+r2)²` — avoid `sqrt`).
- **SAT (Separating Axis Theorem)** — for **rotated boxes / convex polygons**: two convex shapes don't intersect iff there exists an axis (a face normal of either shape) on which their projections don't overlap. Test all face normals; if any gap exists → no collision. The **minimum overlap axis gives the collision normal + penetration depth** (the MTV, minimum translation vector) for resolution.
  - SAT only works on **convex** shapes — decompose concave shapes into convex pieces.
  - For circles vs polygon, also test the axis from circle center to nearest vertex.

---

## 3. Broadphase: don't test every pair (O(n²) killer)

Checking all pairs is O(n²) — fine for dozens, catastrophic for hundreds/thousands. Split into **broadphase** (cheap culling of pairs that can't touch) → **narrowphase** (exact AABB/SAT on survivors).

- **Uniform spatial hash grid** — bucket objects into cells by position; only test objects sharing a cell. Best default for many similar-sized, evenly-distributed objects (bullets, particles, .io games). Cell size ≈ average object size (2–3×).
- **Quadtree** — recursively subdivide space; good for **non-uniform / clustered / varied-size** objects. Rebuild or update each frame.
- **Sort and sweep (sweep-and-prune)** — sort AABB endpoints on an axis; good when objects move coherently.
- **Rule:** the moment you have >~100 dynamic colliders, add a broadphase. Physics engines do this internally; hand-rolled collision needs it explicitly.

---

## 4. Tunneling & swept collision (the top correctness bug)

**Tunneling:** a fast object moves so far in one frame that it passes *through* a thin wall without ever overlapping it on any frame. Discrete AABB checks miss it entirely.

Fixes (in order of preference):
- **Swept collision (continuous / CCD):** instead of testing the box at its new position, test the **motion path**. **Swept AABB** computes the fraction of the frame `t (0..1)` at which the moving box first touches the target, then moves the object to exactly that contact point and resolves — no overlap ever occurs. Essential for bullets, fast platformers, thin walls.
- **Substepping / smaller fixed timestep:** break a big move into several small steps and test each. Simpler than swept, good enough for moderate speeds.
- **Raycast** fast/small projectiles (bullets) instead of moving a body — cast a ray along the travel and hit the first collider.
- **Minimum wall thickness** ≥ max per-step travel is a cheap band-aid but fragile.
- In engines, **enable CCD** on fast bodies (Rapier `setCcdEnabled(true)`, cannon-es CCD options).

---

## 5. Collision *resolution* (not just detection)

- **AABB resolution:** compute penetration on each axis; push the object out along the **axis of least penetration** (this makes walls/floors behave). Zero the velocity component on the axis you resolved (hitting the floor kills downward velocity).
- **Resolve axes separately** for platformers ("move X, resolve X, then move Y, resolve Y") — prevents catching on tile seams and gives clean wall-slide behavior.
- **Ground detection:** you're grounded if resolving downward this frame; set a small `coyoteTime` (see input skill) for feel.
- **Avoid sinking/jitter:** apply a tiny "skin"/epsilon and don't over-correct; with engines, tune restitution/friction and let the solver settle. Resting jitter usually means variable timestep or fighting corrections.
- With **SAT**, resolve along the MTV (normal × penetration depth); reflect/scale velocity along the normal for bounces.

---

## 6. Fixed timestep — the stability foundation (tie-in)

**Physics MUST run on a fixed timestep with an accumulator** ("Fix Your Timestep!", Gaffer On Games). Variable `dt` makes collision/physics non-deterministic and unstable (tunneling worsens, springs explode, replays/netcode desync).
```js
const STEP = 1/60; let acc = 0;
function frame(dt){ acc += Math.min(dt, 0.25);        // clamp to avoid spiral-of-death after tab-out
  while (acc >= STEP){ physicsStep(STEP); acc -= STEP; }
  const alpha = acc / STEP; render(alpha);            // interpolate between last two states
}
```
- **Clamp accumulated dt** (e.g. ≤0.25s) so a backgrounded tab doesn't trigger a "spiral of death" of catch-up steps.
- **Interpolate rendering** between the previous and current physics state using `alpha` for smooth visuals at any display rate.
- All physics engines expect a fixed step — call `world.step()` at a fixed rate, not per rAF with variable dt. Rapier is deterministic given fixed steps + same inputs (good for netcode).

---

## 7. 3D character controllers

Falling capsules from raw rigid-body dynamics feel bad for players — use a **kinematic character controller**:
- **Rapier `KinematicCharacterController`** (favored): create via `world.createCharacterController(offset)`, use a kinematic body + capsule/collider, then each fixed step call `computeColliderMovement(collider, desiredTranslation)` and read `computedMovement()` to get the collision-corrected move. Built-in **autostep** (stairs), **snap-to-ground** (don't float off slopes/ramps), **max slope climb angle**, and slide-along-walls. Set gravity/jump yourself (kinematic = you control motion).
- **cannon-es**: `PointerLockControlsCannon` example / a sphere or capsule body; more manual.
- Rules: use a **capsule** (not a box) so the player slides over small steps and around corners; apply movement each **fixed step**; do ground checks via the controller's grounded flag or a short downward ray; keep the visual mesh synced to the body each frame (with interpolation).

---

## 8. Bug-prevention checklist
- **Variable timestep for physics** → non-deterministic, jitter, worse tunneling, netcode desync; fixed step + accumulator.
- **No dt clamp** → "spiral of death" after tab-out; clamp accumulated dt.
- **Fast objects passing through walls** → tunneling; use swept AABB / CCD / raycast / substeps.
- **SAT on concave shapes** → wrong results; decompose into convex pieces.
- **AABB test on rotated boxes** → false/missed hits; use SAT for rotation.
- **O(n²) pair testing** → frame drops at scale; add spatial hash/quadtree broadphase.
- **Resolving both axes at once** → catching on tile seams / clipping corners; resolve X then Y.
- **Not zeroing velocity on the resolved axis** → gravity accumulates, object shudders into ground.
- **Player floats off ramps / trips on stairs** → missing snap-to-ground / autostep; use a character controller (Rapier) with those enabled.
- **Box character controller** → snags on edges; use a capsule.
- **`sqrt` in hot distance checks** → wasted perf; compare squared distances.
- **Reading physics state without render interpolation** → visible stutter at high refresh rates; interpolate with `alpha`.

---

## Defaults to apply
- **Right-size the solution:** hand-rolled **AABB + separate-axis resolution + swept collision** for arcade/2D/platformers; **Rapier** (2D or 3D) when we need real rigid bodies, joints, or a character controller. Don't default to a heavy engine.
- **Always run physics on a fixed timestep with a clamped accumulator + render interpolation** — this one pattern prevents tunneling instability and jitter, and it ties directly into the fixed-timestep game loop (see `threejs-foundational.md`).
- **Ship anti-tunneling by default** for fast objects (swept AABB or CCD/raycast for projectiles) and a **broadphase (spatial hash or quadtree)** once there are many colliders.
- **For 3D player movement, generate a Rapier `KinematicCharacterController` with a capsule, autostep, snap-to-ground, and slope limits** — not a raw dynamic body — so movement feels right out of the box.

---

## Sources
- MDN — 2D collision detection (AABB, circle, SAT): https://developer.mozilla.org/en-US/docs/Games/Techniques/2D_collision_detection
- Gaffer On Games — "Fix Your Timestep!": https://gafferongames.com/post/fix_your_timestep/ ; "Collision Response and Coulomb Friction": https://gafferongames.com/post/collision_response_and_coulomb_friction/
- Swept AABB collision — jitter physics / "Swept AABB Collision Detection and Response" (gamedev): https://www.gamedev.net/tutorials/programming/general-and-gameplay-programming/swept-aabb-collision-detection-and-response-r3084/
- SAT reference — Metanet/N tutorial & dyn4j "SAT (Separating Axis Theorem)": https://dyn4j.org/2010/01/sat/ ; https://www.metanetsoftware.com/technique/tutorialA.html
- Spatial hashing / broadphase — "Spatial Hashing" (gamedev) & quadtree collision (GameDev Academy): https://gameprogrammingpatterns.com/spatial-partition.html
- Rapier — docs (rigid bodies, CCD, character controller): https://rapier.rs/docs/ ; character controller: https://rapier.rs/docs/user_guides/javascript/character_controller ; JS bindings: https://github.com/dimforge/rapier.js
- cannon-es — repo/examples (incl. PointerLockControlsCannon): https://github.com/pmndrs/cannon-es ; https://pmndrs.github.io/cannon-es/
- Matter.js: https://brm.io/matter-js/ ; Planck.js (Box2D): https://github.com/piqnt/planck.js
- Box2D docs (continuous collision / solver background): https://box2d.org/documentation/
- Game Programming Patterns — Spatial Partition & Game Loop: https://gameprogrammingpatterns.com/spatial-partition.html , https://gameprogrammingpatterns.com/game-loop.html
