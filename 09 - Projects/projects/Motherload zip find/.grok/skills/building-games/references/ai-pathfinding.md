# Game AI & Pathfinding (A* on grids, steering, FSM, behavior trees, navmesh)

Consolidated from Red Blob Games (the canonical A*/grid reference), Craig Reynolds' steering work, and Game AI Pro / behavior-tree canon (see Sources). Focus: what an AI builder needs so enemies/agents **move sensibly and don't jitter, get stuck, or tank the frame rate.**

---

## 1. Pathfinding: A* on a grid (the default)

**A\*** finds the shortest path by expanding the node that minimizes `f = g + h`, where `g` = cost so far and `h` = heuristic estimate to goal. It's Dijkstra + a goal-directed heuristic.

Rules:
- **Heuristic must be admissible** (never overestimate true cost) or A* may return a non-optimal path.
  - **4-directional grid → Manhattan distance** `|dx|+|dy|`.
  - **8-directional grid → octile/Chebyshev distance** (diagonal cost ≈ 1.414). Using Manhattan on an 8-dir grid overestimates → wrong paths; using Euclidean on a 4-dir grid underestimates → slow.
- **Use a binary heap / priority queue** for the open set. A linear scan of the open set is the #1 A* perf bug — turns O(E log V) into O(V²) and stutters on big maps.
- **Track a closed set / best-g per node** so you don't reprocess nodes; update `g` if a cheaper path to a node is found.
- **Weighted tiles:** give terrain costs (mud=5, road=1) via `g`; A* naturally routes around expensive terrain.
- **Diagonal corner-cutting:** forbid diagonal moves that clip through two wall corners, or agents visually cut through walls.
- **Reconstruct the path** by following `cameFrom` parents from goal back to start, then reverse.

Related tools (Red Blob Games):
- **Breadth-First Search / Dijkstra maps ("flow fields"):** when *many* agents chase *one* goal (tower defense, RTS), compute one BFS/Dijkstra field from the goal and have every agent follow the gradient — far cheaper than one A* per agent.
- **Greedy Best-First** is faster but not optimal; A* is the balanced default.
- **JPS (Jump Point Search)** optimizes A* on uniform-cost grids.

Libraries: **PathFinding.js** (grid A*/JPS/BFS, easy), or roll your own for control.

**Path-following gotchas:** smooth the raw grid path (string-pulling / line-of-sight simplification) so agents don't zig-zag along cell centers; recompute paths sparingly (on target move / periodically, not every frame); handle "no path exists" explicitly.

---

## 2. Steering behaviors (smooth local movement)

A* gives *where* to go; **steering** (Craig Reynolds) makes agents *move there naturally* by accumulating steering forces on velocity:
- **Seek / Arrive** — move toward a target; Arrive decelerates within a slowing radius (prevents overshoot/orbit).
- **Flee / Evade** — away from a threat.
- **Pursuit** — seek the target's *predicted future* position.
- **Wander** — smooth random roaming (jitter a point on a circle ahead), not teleporting random targets.
- **Obstacle avoidance** — steer around obstacles detected by a look-ahead feeler.
- **Separation / Alignment / Cohesion** = **Flocking (boids)** for crowds/schools/swarms.
- **Path following** — follow a computed A* path smoothly rather than snapping cell-to-cell.

Rules: **combine forces** (weighted sum or priority), **clamp to max force and max speed**, apply with **delta time**. Steering handles the smooth "how"; A*/flow fields handle the strategic "where." Use both together — A* alone looks robotic; steering alone gets stuck on walls.

---

## 3. Finite State Machines (FSM) — decision-making for simple agents

- Agent is in exactly one **state** (Idle, Patrol, Chase, Attack, Flee); **transitions** fire on conditions (saw player → Chase; lost player → Patrol; low HP → Flee).
- Each state has `enter()`, `update(dt)`, `exit()`. Keep transition logic centralized/table-driven.
- Great for enemies with a handful of behaviors. **Downside:** transitions explode combinatorially (n states → up to n² transitions) — becomes spaghetti past ~6–8 states.
- **Hierarchical FSMs** (states containing sub-states) tame some of that.

---

## 4. Behavior Trees (BT) — scalable decision-making

When FSMs get unwieldy, use a **behavior tree**: a tree of nodes evaluated each tick, each returning **Success / Failure / Running**.
- **Composites:** **Sequence** (run children in order, fail-fast — "AND"), **Selector/Fallback** (try children until one succeeds — "OR"), **Parallel**.
- **Decorators:** Inverter, Repeat, Cooldown, Succeeder, condition guards.
- **Leaves:** Actions (MoveTo, Attack, Reload) and Conditions (IsPlayerVisible, HasAmmo).
- **`Running` is the key concept:** long actions (walking a path) return `Running` and resume next tick — don't block.
- Far more **modular/reusable/authorable** than FSMs; the industry standard for complex NPCs. Consider a lib (behaviortree.js) or a small hand-rolled implementation.
- **Utility AI / GOAP** are alternatives for emergent/goal-driven behavior at higher complexity.

**Rule:** simple enemies → FSM; complex/multi-goal NPCs → behavior tree. Don't build a BT framework for a single 3-state slime.

---

## 5. Navmesh basics (3D / open spaces)

- Grids are natural for tile games; for **open 3D/continuous** worlds, a **navigation mesh** (convex polygons covering walkable surfaces) is more efficient and gives smoother paths than a dense grid.
- Run A* over the polygon graph (portal edges), then **string-pull ("funnel algorithm")** through polygon portals for a smooth, natural path.
- In JS: **recast-navigation-js** (WASM port of industry-standard Recast/Detour) bakes navmeshes from level geometry and provides crowd/agent avoidance; three-pathfinding for a simpler Three.js navmesh follower.
- Navmeshes handle multi-level geometry (ramps, bridges) that flat grids can't; support off-mesh links (jumps, ladders).

---

## 6. Performance & correctness rules
- **Don't A* every agent every frame.** Cache paths; recompute on target movement or a throttled interval; **time-slice** pathfinding across frames or use a **web worker** for big searches so the main thread doesn't hitch.
- **Flow field / Dijkstra map** when many agents share one goal — one computation for all.
- **Binary heap** open set, **typed arrays** for large grids, reuse buffers (no per-search allocation churn).
- **Run AI on a fixed timestep** (decoupled from render) for determinism/netcode; often at a lower rate than render (e.g. 10–20Hz for decisions) with steering interpolating between.
- **Stagger** AI updates across agents (LOD-AI): distant/off-screen agents think less often.
- Handle **dynamic obstacles**: mark blocked cells / re-path when the map changes; use local steering avoidance for moving agents.

---

## 7. Bug-prevention checklist
- **Non-admissible heuristic** (Euclidean on 4-dir, Manhattan on 8-dir) → non-optimal or slow paths; match heuristic to movement.
- **Linear open-set scan** → severe stutter on big maps; use a binary heap.
- **A* per agent per frame** → frame drops; cache/throttle/flow-field/worker.
- **Following raw grid path** → robotic zig-zag; smooth/string-pull and add steering.
- **No Arrive/slowing radius** → agents overshoot and orbit the target.
- **Corner-cutting diagonals** → agents clip through wall corners; forbid unsafe diagonals.
- **FSM sprawl** → unmaintainable transitions; switch to a behavior tree.
- **Blocking actions in a BT** → tree stalls; return `Running` for long actions.
- **Ignoring "no path"** → crashes/agents freeze; handle unreachable goals explicitly.
- **Not re-pathing on map change** → agents walk into new walls; invalidate cached paths.
- **AI tied to render frame rate** → non-deterministic/uneven; fixed timestep, staggered.

---

## Defaults to apply
- **Default AI stack:** **A\* on a grid** (binary-heap open set, movement-matched heuristic, weighted tiles) for "where," **steering behaviors** (seek/arrive/avoid/separation) for smooth "how," and an **FSM for simple enemies / behavior tree for complex NPCs** for "what to do."
- **Use flow fields (BFS/Dijkstra map) when many agents chase one target** (tower defense/RTS) — big perf win over per-agent A*.
- **For 3D/open worlds, generate navmeshes via recast-navigation-js**; smooth paths with the funnel algorithm.
- **Bake in the perf guardrails**: cache paths, throttle/time-slice/worker pathfinding, run AI on a fixed lower-rate tick with steering interpolation, and stagger distant agents. Always handle "no path found."

---

## Sources
- Red Blob Games — Introduction to A* (canonical, interactive): https://www.redblobgames.com/pathfinding/a-star/introduction.html ; A* Implementation Guide: https://www.redblobgames.com/pathfinding/a-star/implementation.html
- Red Blob Games — Grids/heuristics & "Pathfinding for Tower Defense" (flow fields): https://www.redblobgames.com/pathfinding/tower-defense/ ; grid pathfinding tricks: https://www.redblobgames.com/pathfinding/grids/algorithms.html
- Craig Reynolds — "Steering Behaviors For Autonomous Characters": https://www.red3d.com/cwr/steer/ ; Boids: https://www.red3d.com/cwr/boids/
- Amit Patel / Red Blob — heuristics guide: https://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
- PathFinding.js (grid A*/JPS/BFS): https://github.com/qiao/PathFinding.js
- Jump Point Search (Harabor & Grastien): https://harablog.wordpress.com/2011/09/07/jump-point-search/
- Behavior Trees — Chris Simpson, "Behavior trees for AI: How they work": https://www.gamedeveloper.com/programming/behavior-trees-for-ai-how-they-work ; behaviortree.js: https://github.com/Calamari/BehaviorTree.js
- Game Programming Patterns — State (FSM): https://gameprogrammingpatterns.com/state.html
- Recast/Detour navmesh: https://github.com/recastnavigation/recastnavigation ; recast-navigation-js: https://github.com/isaac-mason/recast-navigation-js ; three-pathfinding: https://github.com/donmccurdy/three-pathfinding
- GOAP (Jeff Orkin, F.E.A.R.): https://alumni.media.mit.edu/~jorkin/goap.html
