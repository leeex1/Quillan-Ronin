# Genre Playbook — Tower Defense

Place towers on a grid to stop waves of enemies walking a path to your base. The genre is fundamentally about **a grid, pathfinding, targeting, and economy**. 2D (Phaser/Canvas) or 3D top-down (Three.js). Read `../threejs-foundational.md` first (delta time, pooling, top-down camera).

---

## 1. Core mechanics (minimal-but-good scope for a demo)

1. **A grid map** with a defined enemy **path** from spawn to the base.
2. **Enemies that walk the path** in timed **waves**, with HP and speed.
3. **Tower placement** on buildable cells, costing money.
4. **Towers that target + shoot** enemies in range (projectile or hitscan), dealing damage.
5. **Economy + lives:** kills give gold; leaks cost lives; game over at 0 lives. Wave counter + "start next wave" button.

One tower type, one enemy type, a fixed path, 5–10 escalating waves, gold, and lives is a complete game. Add tower types/upgrades, enemy variety (fast/armored/flying), and maze-building later. **Fixed-path TD is much simpler and less bug-prone than maze/free-build TD — default to it for demos.**

---

## 2. Map, path & pathfinding

- **Two flavors — pick deliberately:**
  - **Fixed path (recommended default):** the path is authored (a polyline / ordered list of waypoints). Enemies just follow waypoints. No runtime pathfinding, no way to trap enemies — far fewer bugs.
  - **Maze / free-build:** players place towers anywhere and enemies pathfind around them (Bloons-style is fixed; Desktop TD is maze). Requires **A\*** re-pathing and a critical rule: **never allow a placement that fully blocks the path** — validate with a pathfinding check before committing, and reject if no route remains.
- **Grid model:** a 2D array of cells with types (path / buildable / blocked / spawn / base). Convert world↔grid with `floor(pos / cellSize)`. Snap tower placement to cell centers.
- **A\* (maze TD):** grid neighbors (4- or 8-connected), Manhattan/octile heuristic. Recompute affected enemies' paths when a tower is placed/sold. Cache the path; only recompute on change. See Red Blob Games' A* guide (canonical).
- **Waypoint following (fixed path):** each enemy stores its target waypoint index; move toward it by `speed*delta`; when within a small epsilon, advance to the next; reaching the end = a "leak" (lose a life, despawn).

---

## 3. Towers: targeting, range & firing

- **Range check:** distance (squared, to avoid sqrt) from tower to enemy ≤ range². Use a **spatial grid/quadtree** for broad-phase if there are many enemies — don't test every tower against every enemy each frame (O(towers×enemies)).
- **Targeting policy (make it explicit & selectable):** common options — **First** (furthest along the path, the usual default), **Last**, **Closest**, **Strongest (most HP)**, **Weakest**. "First" is the standard because it stops leaks best. Ambiguous/implicit targeting is a common complaint — pick and document one.
- **Fire rate:** cooldown timer per tower (`if (now - lastFire >= 1/fireRate) fire()`), delta-based, not per-frame.
- **Projectiles vs hitscan:** projectiles (pooled) that travel and can **home** on the target or aim at a **lead/predicted position** feel better and enable "miss if enemy dies mid-flight." Hitscan (instant damage + a drawn beam) is simpler and fine for lasers. Pool projectiles.
- **Damage types & AoE:** splash damage hits enemies within a radius of impact; slow/poison apply timed status effects. Even one splash tower adds a lot of depth.
- **Rotate the tower/turret to face its target** (mind facing conventions). Show range as a ring on hover/selection.
- **Upgrades/selling:** click a tower → upgrade (more damage/range/rate) or sell for partial refund.

---

## 4. Waves & economy (the balance core)

- **Wave data as configuration**, not hardcoded logic: a list of waves, each a list of `{enemyType, count, spawnInterval, delay}`. Easy to tune and extend.
- **Escalation:** later waves have more/tougher/faster enemies; introduce armored (reduce non-piercing damage), fast, and flying (only certain towers hit) types over time.
- **Economy loop:** enemies give gold on death; towers/upgrades cost gold; a leak costs a life. Tune so the player is always a little short — that tension is the game. Give inter-wave prep time and optionally a bonus for starting the next wave early.
- **Lives + win/lose:** lose a life per leak, game over at 0; win by surviving all waves (or endless mode with scaling).
- **Boss waves** for pacing spikes.

---

## 5. Common bugs to avoid (checklist)

- **(Maze TD) Placement that fully blocks the path** → always run a pathfinding validity check before allowing placement; reject if no route to base remains.
- **Enemies overshoot/orbit waypoints** → advance when within an epsilon distance and clamp the final step so they don't jitter around the point.
- **O(towers×enemies) targeting each frame** → use squared-distance checks + a spatial grid/quadtree for broad-phase.
- **Ambiguous targeting** → choose an explicit policy (default "First"), make it consistent and ideally player-selectable.
- **Projectiles chasing dead enemies / never expiring** → give projectiles a target ref + TTL; on target death, either retarget, continue to last position, or despawn.
- **Fire rate frame-rate dependent** → cooldown by delta time, not per frame.
- **Building on path/occupied cells** → validate cell type + occupancy before placing; snap to grid.
- **Economy exploits** → full-refund sell + rebuild loops; use partial refunds; validate gold on every transaction.
- **Not pooling enemies/projectiles** → GC stutter in big waves.
- **Enemies overlapping into one blob** → optional separation, or just accept single-file on a fixed path.

---

## Defaults to apply

1. **Default to fixed-path (waypoint) TD** for demos — no runtime pathfinding, dramatically fewer bugs. Offer maze/A* only when explicitly wanted.
2. **If maze/free-build: NEVER allow a tower placement that fully blocks the path** — validate with an A* reachability check before committing.
3. **Grid-based map + snap placement; validate cell buildability + gold before placing.**
4. **Explicit, selectable targeting policy (default "First"); squared-distance range checks + spatial grid** for performance. Pool projectiles/enemies.
5. **Wave and enemy stats as data/config, not code** — makes balancing and expansion trivial.
6. **Tune the economy tight** (gold from kills, cost of towers, lives from leaks) — the shortage-driven tension is the fun. Minimal scope: 1 tower, 1 enemy, fixed path, escalating waves, gold + lives.

---

## Sources
- Red Blob Games — Introduction to A* / pathfinding (the canonical grid pathfinding reference): https://www.redblobgames.com/pathfinding/a-star/introduction.html
- Red Blob Games — Grids, hexagons, and tile math: https://www.redblobgames.com/grids/
- MDN — 2D collision / distance checks: https://developer.mozilla.org/en-US/docs/Games/Techniques/2D_collision_detection
- Phaser — Groups & pooling (projectiles/enemies): https://docs.phaser.io/phaser/concepts/gameobjects/group
- "Tower Defense targeting priorities" (Bloons/Desktop TD community references): https://bloons.fandom.com/wiki/Targeting_Priority
- Amit Patel / Red Blob — implementation notes on flow fields for many-agent pathing: https://www.redblobgames.com/pathfinding/tower-defense/
