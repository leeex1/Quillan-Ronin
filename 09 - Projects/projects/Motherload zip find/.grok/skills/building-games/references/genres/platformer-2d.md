# Genre Playbook — 2D Platformer

How to build a browser 2D platformer whose jump *feels* good and whose collisions don't glitch. Default engine: **Phaser** (Arcade Physics) or a hand-rolled AABB loop on Canvas. Read `../threejs-foundational.md` first (delta time, Phaser scene/pooling rules). This file focuses on the two things that make or break a platformer: **jump feel** and **tile collision**.

---

## 1. Core mechanics (minimal-but-good scope for a demo)

A great-feeling platformer demo needs:
1. **A responsive jump** with the "forgiveness" features below (coyote time, jump buffer, variable height).
2. **Solid tilemap/AABB collision** — no sticking to walls, no falling through floors, no tunneling at speed.
3. **One-way (drop-through) platforms.**
4. **A few hazards + a goal** (spikes/pit → respawn at last checkpoint; flag/door to win).
5. **A camera that follows with a deadzone** and doesn't jitter.

That's a complete micro-game. Skip enemies with AI, wall-jump/dash, and moving-platform edge cases until the core jump + collision feel perfect. **Jump feel is the whole genre — spend your polish budget there.**

---

## 2. Jump feel — the forgiveness features (do ALL of these)

These are the difference between "floaty/frustrating" and "tight." Popularized by Celeste and GMTK's "Platformer Toolkit."

- **Coyote time (~80–120 ms):** allow a jump for a few frames *after* the player walks off a ledge. Store `coyoteTimer`; reset to `COYOTE_TIME` while grounded, decrement each frame, and allow a jump if `coyoteTimer > 0`. Prevents "I pressed jump but it didn't register at the edge."
- **Jump buffering (~100–150 ms):** if the player presses jump slightly *before* landing, remember it and jump the instant they touch ground. Store `jumpBufferTimer` on keypress; on landing, if `jumpBufferTimer > 0`, jump immediately.
- **Variable jump height:** short tap = short hop, hold = full jump. On jump, set upward velocity to max. On **key release while still rising** (`velocityY < 0`), cut the velocity (e.g. `velocityY *= 0.5`). Do NOT set velocity to 0 or it looks abrupt.
- **Asymmetric gravity:** apply **higher gravity while falling** than while rising (e.g. fall gravity ×1.5–2.5), and often a lower gravity near the jump apex ("apex hang") plus a small horizontal speed boost at apex. This is the single biggest "feel" upgrade — a pure parabola feels floaty.
- **Clamp fall speed** (terminal velocity) so long falls stay controllable and collision stays stable.
- **Instant-ish horizontal control:** high ground acceleration + high friction so direction changes feel snappy; reduced acceleration + a little air drag in the air. Avoid slippery ice-feel unless intended.

Combine coyote + buffer + variable height + asymmetric gravity and even a placeholder box "feels like Mario."

---

## 3. Collision — tilemap / AABB (the #1 bug source)

- **Separate the axes: resolve X, then Y (or Y then X) independently.** Move on X, resolve X collisions; then move on Y, resolve Y collisions. Resolving both at once with a single overlap vector causes corner snags and wall-sticking.
- **Use AABB (axis-aligned bounding box) vs the tile grid.** For each move, compute which tiles the player's box overlaps (world→tile via `floor(pos / tileSize)`), and push the player out of solid tiles along the axis being resolved. This is O(overlapped tiles), not O(all tiles).
- **Grounded detection:** you are grounded when a downward Y-resolution stopped you this frame (or a 1px probe below finds a solid tile). Set `velocityY = 0` on landing and on hitting a ceiling.
- **Prevent tunneling at high speed:** if the player can move more than ~half a tile per frame, do **swept collision** or sub-step the movement (loop in small increments) so you never skip over a thin floor. Fixed timestep helps.
- **Don't get stuck in seams between tiles:** when scanning multiple solid tiles, resolve using the collision on the axis of motion and skip internal edges (a tile that is adjacent to another solid tile shouldn't push you sideways). Merging colliders for runs of tiles, or ignoring faces between two solids, fixes the classic "snag on a flat floor" bug.
- **Phaser:** use `map.setCollisionByProperty({ collides: true })` / `setCollisionBetween`, then `this.physics.add.collider(player, layer)`. Set `tile.setCollision(...)` per-side for one-ways. Arcade physics handles separation for you — trust it before hand-rolling.

---

## 4. One-way (drop-through) platforms

- **Collide only when moving downward and coming from above.** The player passes up through the platform but lands on top.
- **Implementation:** only resolve the collision if `velocityY >= 0` (falling) AND the player's *previous* bottom was above the platform's top last frame. Otherwise ignore it.
- **Drop-through:** press Down (+ jump) to temporarily disable the current one-way platform for a few frames so the player falls through.
- **Phaser Arcade:** set `tile.setCollision(false,false,true,false)` (top-only) or use `body.checkCollision.down`; or use `collider`'s `processCallback` to return false when the player is below the platform.

---

## 5. Camera

- **Follow with a deadzone / soft zone**, not rigid lock — the camera only moves when the player leaves a central box, which kills micro-jitter. Phaser: `this.cameras.main.startFollow(player, true, lerpX, lerpY)` + `setDeadzone(w, h)`.
- **Add look-ahead** in the direction of movement so the player sees where they're going.
- **Clamp the camera to level bounds** (`setBounds`) so you never see past the edges.
- **Round the camera position to whole pixels** for pixel-art games to avoid shimmer; pair with `pixelArt: true` / `roundPixels: true`.
- Smooth vertical follow separately (or snap on landing) so jumps don't make the camera bounce.

---

## 6. Common bugs to avoid (checklist)

- **Floaty jump** → add asymmetric gravity (heavier falling), apex hang, variable height, terminal velocity.
- **"Jump didn't register" at edges / just before landing** → missing coyote time / jump buffering.
- **Sticking to walls / snagging on flat floors** → resolving both axes at once, or not ignoring seams between adjacent solid tiles. Resolve axes separately.
- **Falling through floor at high speed** → tunneling; sub-step or swept collision + fixed timestep + clamp fall speed.
- **Can't drop through / falls through one-ways when jumping up** → one-way check must gate on downward velocity + coming-from-above.
- **Double/infinite jump** → grounded flag not reset correctly; only allow jump when grounded OR coyote timer > 0, and consume the buffer.
- **Camera jitter** → no deadzone / not pixel-rounded / following unsmoothed physics position.
- **Movement frame-rate dependent** → scale by delta (Phaser gives you `delta`; Arcade integrates for you, but custom code must use it).
- **Getting stuck when spawning inside a tile** → validate spawn/respawn points are in empty space.

---

## Defaults to apply

1. **Always implement the "forgiveness trio": coyote time (~0.1s), jump buffering (~0.12s), and variable jump height (cut velocity on early release).** These are cheap and are the difference between a good and bad platformer. Bake them into any generated platformer by default.
2. **Use asymmetric gravity (heavier when falling) + terminal velocity + apex hang.** Never ship a symmetric-parabola jump.
3. **Resolve collision one axis at a time (X then Y) against an AABB/tile grid; sub-step fast movement to prevent tunneling.** Prefer Phaser Arcade's collider over hand-rolled math when possible.
4. **One-way platforms: collide only when falling and coming from above; Down = drop through.**
5. **Camera: follow with a deadzone + look-ahead, clamp to bounds, round to pixels for pixel art.**
6. **Minimal scope:** tight jump + solid collision + one-ways + hazards/checkpoint + goal. Polish the jump before adding enemies.

---

## Sources
- GMTK — "Why Does Celeste Feel So Good to Play" / Platformer Toolkit (coyote time, buffering, variable jump): https://www.youtube.com/watch?v=yorTG9at90g
- GMTK Platformer Toolkit (interactive): https://gmtk.itch.io/platformer-toolkit
- "Coyote Time and Jump Buffering" (Nathan Hoad / common write-ups): https://nathanhoad.net/coyote-time-and-jump-buffering
- MDN — 2D breakout/AABB collision detection basics: https://developer.mozilla.org/en-US/docs/Games/Techniques/2D_collision_detection
- Phaser — Tilemaps & Arcade Physics collision: https://docs.phaser.io/phaser/concepts/tilemaps and https://docs.phaser.io/phaser/concepts/physics/arcade
- "Rendering Tile-Based Worlds" / one-way platforms (MaddyThorson / Celeste dev notes): https://maddythorson.medium.com/
- Red Blob Games — grid/tile math reference: https://www.redblobgames.com/
