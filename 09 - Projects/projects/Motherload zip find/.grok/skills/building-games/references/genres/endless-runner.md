# Genre Playbook — Endless Runner

Auto-forward games where the player dodges obstacles and collects pickups at ever-increasing speed (Temple Run, Subway Surfers, Canabalt, Flappy-adjacent, Chrome Dino). Works in 2D (Phaser) or 3D (Three.js). The whole genre rests on **procedural spawning + object pooling + a difficulty ramp**. Read `../threejs-foundational.md` first (delta time, pooling, mobile input).

---

## 1. Core mechanics (minimal-but-good scope for a demo)

1. **Auto-run:** the player moves forward automatically (or the world scrolls toward a fixed player) at a speed that **increases over time**.
2. **Simple avoidance input:** jump / slide, or lane-switch left/right (or both).
3. **Endless procedural spawning** of obstacles + collectibles as pooled objects.
4. **Collision → death** (one hit or a few lives), with **distance/coins as score**.
5. **Instant restart** and a persisted high score.

One lane setup + jump/slide + one obstacle type + coins + a speed ramp + hi-score is a complete, replayable game. Add power-ups, multiple biomes, and characters later. **The tuning of the speed/difficulty curve is the game.**

---

## 2. The "endless" trick: scroll the world, recycle the pieces

- **Keep the player near a fixed position; move the world toward them** (or move the player and follow with the camera — mathematically equivalent). Fixed-player + moving-world is usually simpler for spawning and scoring by distance.
- **Chunk/segment spawning:** build the track from prefab segments (ground tiles, obstacle patterns). Spawn a new segment ahead when the last one enters view; **despawn/recycle** segments once they pass behind the player/camera.
- **Object pooling is mandatory.** Runners spawn thousands of obstacles/coins over a run — preallocate pools and recycle (`active/visible=false` in Phaser, or a free-list). Never create/destroy per spawn (GC stutter kills the feel).
- **Parallax background** (2D): multiple layers scrolling at different speeds for depth; use `tileSprite`/texture offset scrolling rather than spawning background sprites.
- **Seamless ground:** loop a tiling ground texture by scrolling UV/tilePosition, or ping-pong two ground pieces, so there's no visible seam.

---

## 3. Controls & feel

- **Jump:** use platformer forgiveness where relevant — **jump buffering** (register a press slightly before landing) and a short **coyote window** make it feel fair. Variable jump height optional. Clamp fall speed.
- **Slide/duck:** a timed crouch that shrinks the hitbox; must end even if the key is held-then-released oddly (timer-driven, not just key-held).
- **Lane switch (3D runners):** discrete lanes; **tween** the player smoothly between lane X positions rather than teleporting, and lock input during the transition or allow queueing. Support swipe (mobile) + arrows/A-D (desktop).
- **One-button variants (Flappy/Canabalt):** tap = flap/jump; keep the single input crisp and buffered.
- **Coyote/buffer + snappy, immediate response** — runners punish input lag hard because timing is everything.
- **Fairness rule:** never spawn an **impossible/unavoidable** obstacle combination for the current speed. Constrain the spawner so the player always has a reachable gap/lane (see §5).

---

## 4. Difficulty & scoring

- **Speed ramps up with distance/time** (linear or gently curved), which naturally raises difficulty. Tie obstacle density and pattern complexity to the same progression.
- **Cap the max speed** or the game becomes unplayable / physics/collision get unstable — and above some speed the player literally can't react.
- **Reaction-time budget:** obstacles must appear far enough ahead that the player has time to react at the current speed (spawn distance should scale with speed). This is the core fairness constraint.
- **Score = distance** (+ coins). Persist the high score in `localStorage`. Show near-misses / combos for extra juice.
- **Pickups:** coins in patterns (arcs, lines) that sometimes lure the player into risk; occasional power-ups (magnet, shield, jetpack) on a timer.

---

## 5. Common bugs to avoid (checklist)

- **Creating/destroying objects per spawn** → GC hitches; use pools and recycle off-screen objects.
- **Impossible obstacle combos** → spawner must always leave a reachable gap/lane for the current speed; validate patterns.
- **Obstacles appear too late to dodge** → spawn distance / reaction budget must scale with speed.
- **Unbounded speed** → cap max speed; otherwise collision tunneling and unfair, unplayable pace.
- **Collision tunneling at high speed** → swept/segment collision or sub-stepping, not per-frame overlap only.
- **Visible seams in ground/background** → scroll UV/tilePosition or ping-pong two pieces; don't spawn gap-prone tiles.
- **Objects never despawn** → recycle everything behind the camera; otherwise memory/entity count grows until it crashes.
- **Movement/scroll not delta-scaled** → speed differs per frame rate.
- **Slide/jump state gets stuck** → drive crouch/jump with timers and reset on death/restart.
- **No fair restart** → reset speed, pools, score, and player state fully on restart (leftover pooled objects reappearing is a classic bug).

---

## Defaults to apply

1. **Fixed player + scrolling/recycled world built from pooled prefab segments** is the canonical, bug-resistant structure. Pool obstacles, coins, and segments; recycle behind the camera.
2. **Difficulty = increasing speed tied to distance, with a hard speed cap.** Scale obstacle spawn distance to the current speed so there's always time to react.
3. **Never spawn unavoidable obstacles** — the spawner must guarantee a reachable gap/lane for the current speed.
4. **Snappy input with jump buffering + coyote time**; tween lane switches smoothly; support swipe + keyboard.
5. **Score by distance + coins, persisted in localStorage; instant full-reset restart.**
6. **Use swept collision + a speed cap** to avoid high-speed tunneling. Add parallax + seamless scrolling ground for polish.

---

## Sources
- MDN — 2D collision detection & game techniques: https://developer.mozilla.org/en-US/docs/Games/Techniques
- Phaser — Groups / object pooling (recycle sprites): https://docs.phaser.io/phaser/concepts/gameobjects/group
- Phaser — TileSprite (scrolling ground/parallax): https://docs.phaser.io/api-documentation/class/gameobjects-tilesprite
- GMTK / general platformer feel (jump buffering, coyote time apply to runners too): https://gmtk.itch.io/platformer-toolkit
- "How Canabalt / procedural runner level generation works" (chunk-based spawning): https://www.gamedeveloper.com/design/the-power-of-procedural-generation-in-canabalt
- Chrome Dino (T-Rex Runner) open-source reference: https://github.com/wayou/t-rex-runner
- Gaffer On Games — Fix Your Timestep (high-speed collision stability): https://gafferongames.com/post/fix_your_timestep/
