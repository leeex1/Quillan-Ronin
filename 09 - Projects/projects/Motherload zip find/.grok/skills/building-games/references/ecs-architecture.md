# ECS & Game State Architecture for JS Games (bitECS, miniplex, ECS vs OOP, state structure)

Consolidated from webgamedev.com, the bitECS and miniplex docs, and data-oriented-design canon (see Sources). Focus: how an AI builder should **structure game state so it stays fast and untangled** as a game grows — and when *not* to reach for ECS.

---

## 1. What ECS is (and why it exists)

**Entity-Component-System** separates data from behavior:
- **Entity** — an ID (or a plain object) with no logic; just a thing that exists.
- **Component** — pure **data**, no methods (Position, Velocity, Health, Sprite, Collider).
- **System** — pure **behavior** that iterates all entities having a given set of components and updates them (MovementSystem reads Position+Velocity).

Benefits: **composition over inheritance** (add/remove capabilities by adding/removing components — no deep class hierarchies), cache-friendly iteration, easy serialization (state = plain data), and clean separation that scales to thousands of entities.

The classic problem ECS solves: the "deadly diamond" of OOP inheritance (`FlyingEnemy` vs `SwimmingEnemy` vs `FlyingSwimmingShootingEnemy`...). With ECS you just attach `Flying`, `Swimming`, `Shooting` components in any combination.

---

## 2. When ECS vs OOP (don't cargo-cult ECS)

**Use plain OOP / a simple object list when:**
- Small games, jams, prototypes, or a handful of distinct entity types.
- Entity count is low (dozens–low hundreds) and perf isn't a concern.
- The team/model iterates faster with intuitive `player.jump()` classes.
- The engine already gives you a good model (Phaser GameObjects/prefabs, Three.js `Object3D` scene graph) — you can go a long way with a component-ish mixin approach without a full ECS.

**Reach for ECS when:**
- **Many similar entities** (bullet-hell, RTS, particles, simulations, .io games) — thousands of entities needing fast batch updates.
- Highly **combinatorial** entity capabilities (lots of mix-and-match behaviors/status effects).
- You need **determinism/serialization** (netcode, replays, save states) — pure-data components serialize trivially.

**Rule:** ECS is an optimization/organization tool, not a moral requirement. A giant `update()` switch or a tidy class hierarchy is fine for small games. Adopt ECS when entity count/combinatorial complexity actually bites — premature ECS adds boilerplate for no gain.

---

## 3. Library choice: bitECS vs miniplex

**miniplex** — DX-first, entities are **plain JS objects**, components are just properties. Best default for most indie/web games and rapid prototyping; excellent TypeScript + React bindings.
```js
import { World } from 'miniplex';
const world = new World();
const player = world.add({ position:{x:0,y:0}, velocity:{x:100,y:0}, health:{cur:100,max:100} });
const moving = world.with('position','velocity');   // live archetype query
function movementSystem(dt){ for (const e of moving){ e.position.x += e.velocity.x*dt; e.position.y += e.velocity.y*dt; } }
world.remove(player);
```
- Use `world.addComponent(e,'velocity',{...})` / `removeComponent` so queries re-index (mutating a bare property can skip re-indexing).
- `for...of` over a query is fast and **safe for removal during iteration**. No built-in scheduler — you call systems from your loop.
- Object property access is slightly slower at extreme entity counts than bitECS's typed arrays.

**bitECS** — performance-first, data-oriented: entities are **integer IDs**, components are **Structure-of-Arrays (typed arrays)**, archetype/bitmask queries. Best for tens of thousands of entities, WebGPU, perf-critical sims. ~5kb, zero deps (used in Hubs/Third Room; eyed by Phaser 4).
```js
import { createWorld, addEntity, addComponent, query } from 'bitecs';
const world = createWorld({ Position:{x:new Float32Array(1e4),y:new Float32Array(1e4)}, Velocity:{x:[],y:[]} });
const { Position, Velocity } = world.components;
const eid = addEntity(world); addComponent(world,eid,Position); addComponent(world,eid,Velocity);
const moving = query(world,[Position,Velocity]);
for (const e of moving){ Position.x[e] += Velocity.x[e]*dt; Position.y[e] += Velocity.y[e]*dt; }
```
- **Note the API moved on:** older tutorials use `defineComponent`/`defineQuery`/`Types`; current bitECS (0.4+) uses `createWorld({...schemas})` + `query(world,[...])`, plus relationships/observers/prefabs. Verify the version you install.
- SoA layout = cache-friendly; entities-as-IDs = trivial to serialize/network.

**Others:** Becsy (multithreaded), Koota, or the engine's own model. Some projects mix (bitECS core sim + object layer for UI).

---

## 4. Structuring game state (even without a formal ECS)

- **Single source of truth:** keep a central, serializable game-state object (or ECS world). Avoid scattering authoritative data across DOM, closures, and random globals — it makes save/load, undo, and netcode nearly impossible.
- **Separate data from rendering:** gameplay state (positions, health) should be independent of the visual objects (meshes/sprites). Systems update data; a render/sync step pushes data → Three.js/Phaser objects each frame. This keeps the sim testable/deterministic and lets you do fixed-timestep + interpolation.
- **Systems run in a defined order each tick:** input → AI → movement → collision → damage → cleanup → render-sync. Order matters and should be explicit, not incidental.
- **Object pooling instead of create/destroy churn:** for bullets/enemies/particles, recycle entities (mark inactive) rather than allocating/GCing every frame (pairs with ECS well; see `threejs-foundational.md`).
- **Deferred structural changes:** don't add/remove entities in the middle of iterating a query in ways the lib doesn't support — queue spawns/despawns and apply between systems (miniplex `for...of` is removal-safe; still queue mass changes for clarity).
- **Fixed timestep for the sim** (see game-loop/collision skills) so ECS systems are deterministic — required for netcode/replays.
- **Events/messaging** between systems via a small event queue rather than direct cross-system calls, to keep systems decoupled.

---

## 5. Bug-prevention checklist
- **Putting logic in components / data in systems** → defeats ECS; keep components pure data, systems pure behavior.
- **Adopting ECS for a tiny game** → boilerplate with no benefit; use a simple object list/classes until entity count/complexity warrants it.
- **Deep inheritance for entity variety** → combinatorial class explosion; compose with components instead.
- **Mutating a miniplex property directly and expecting query updates** → use `addComponent`/`removeComponent` so archetypes re-index.
- **Using stale bitECS `defineComponent` tutorials against a new version** → API mismatch; check installed version's docs.
- **Mixing render objects into authoritative state** → non-serializable, non-deterministic; separate sim data from view.
- **Undefined system order** → order-dependent bugs (moving before collision, etc.); define an explicit pipeline.
- **create/destroy per frame** → GC stalls; pool entities.
- **Structural changes mid-iteration** → skipped/duplicated updates; defer to a queue.
- **Scattered global state** → save/load and netcode become impossible; centralize the world.

---

## Defaults to apply
- **Default to miniplex** for generated games that need ECS (plain objects, great TS/React DX); **switch to bitECS** only when the game genuinely has thousands of entities or needs SoA perf. For small/simple games, a plain object list or engine prefabs is fine — don't force ECS.
- **Always structure state as a single serializable world with sim/render separation** and an **explicit system pipeline** (input→AI→movement→collision→damage→cleanup→render-sync) run on a fixed timestep. This unlocks save/load, replays, and netcode for free.
- **Bake in pooling + deferred spawn/despawn** for entity-heavy genres.
- When generating ECS code, **verify the library version's current API** (bitECS especially changed) rather than emitting old `defineComponent` patterns from memory.

---

## Sources
- Web Game Dev — Code architecture / ECS overview: https://www.webgamedev.com/code-architecture/ecs
- bitECS — repo & docs: https://github.com/NateTheGreatt/bitECS , https://bitecs.dev/docs/introduction
- miniplex — repo & docs: https://github.com/hmans/miniplex , miniplex-react: https://github.com/hmans/miniplex/tree/main/packages/miniplex-react
- Becsy (multithreaded ECS): https://lastolivegames.github.io/becsy/
- Sander Mertens — "Entity Component System FAQ": https://github.com/SanderMertens/ecs-faq
- Adam Martin — "Entity Systems are the future of MMOG development": http://t-machine.org/index.php/2007/09/03/entity-systems-are-the-future-of-mmog-development-part-1/
- Mick West — "Evolve Your Hierarchy" (components over inheritance): https://cowboyprogramming.com/2007/01/05/evolve-your-hierarchy/
- Richard Fabian — Data-Oriented Design (book): https://www.dataorienteddesign.com/dodbook/
- Game Programming Patterns — Component pattern: https://gameprogrammingpatterns.com/component.html
