# Genre Playbook — Top-Down / Twin-Stick Shooter

Top-down action where you **move with one input and aim with another** (mouse+WASD on desktop, two virtual sticks on mobile, or left/right stick on gamepad). Think Enter the Gungeon, Nuclear Throne, Vampire Survivors, .io shooters. Default engine: **Phaser** (2D) or Three.js with an orthographic/top camera. Read `../threejs-foundational.md` first (delta time, pooling, mobile joysticks).

---

## 1. Core mechanics (minimal-but-good scope for a demo)

1. **Decoupled move + aim** (the defining feature — see §2).
2. **A shooting weapon** with pooled bullets and a fire-rate cooldown.
3. **Enemies that spawn and chase** the player (simple seek/steering), with hit + death.
4. **Collision:** player↔walls, bullets↔enemies, enemies↔player (damage), plus screen/arena bounds.
5. **HUD:** health + score/wave; a game-over + restart.

A single weapon + one enemy type spawning in waves + juicy hit feedback is already a fun loop (Vampire Survivors shipped on essentially this). Add pickups/upgrades only after the core loop is fun.

---

## 2. Controls & aiming (the twin-stick core)

- **Movement is independent of facing.** Move on a normalized vector; aim on a separate vector. This is *the* genre feature — never couple them (that's a tank/first-person control scheme, not twin-stick).
- **Desktop:** WASD/arrows → move vector; mouse position → aim. Aim angle = `Math.atan2(mouse.y - player.y, mouse.x - player.x)` **in world space** (convert screen→world with the camera scroll/zoom, not raw clientX/Y). Left click = fire.
- **Gamepad:** left stick = move, right stick = aim/fire (fire when the right stick is deflected past a deadzone). Apply a **radial deadzone** (`if (len < 0.2) zero it`) and optionally re-scale so the edge maps to full speed.
- **Mobile:** two virtual joysticks (e.g. nipplejs) — left = move, right = aim/fire. Auto-fire while the aim stick is held is friendlier than a separate button.
- **Normalize the movement vector** before applying speed so diagonal movement isn't ~1.41× faster (the classic "faster on diagonals" bug). `vec.normalize().scale(speed * delta)`.
- **Rotate the sprite/mesh to the aim angle**, not the move angle. In Phaser remember sprites face "right" (0 rad) by default; offset art accordingly. In Three.js top-down, rotate around the up axis and mind the +Z/−Z forward convention.
- **Optional auto-aim / aim-assist** (snap to nearest enemy) is great on mobile and for accessibility.

---

## 3. Camera

- **Follow the player centered**, ideally with a small deadzone and **look-ahead toward the aim/mouse** (offset the camera partway toward the cursor) so you see more of where you're shooting.
- **Clamp to level bounds**; for arena games, a fixed camera showing the whole arena is also fine.
- Keep it smooth (lerp) but not laggy; snap on teleport/respawn.
- Phaser: `cameras.main.startFollow(player, true, 0.1, 0.1)` + `setBounds` + `setDeadzone`.

---

## 4. Bullets, enemies & performance

- **Pool everything.** Twin-stick games spawn hundreds of bullets and enemies. Use object pools (Phaser Groups with `maxSize`, or a preallocated array) — never create/destroy per shot. Recycle by toggling `active/visible`.
- **Bullets:** move by velocity·delta; despawn on wall hit, enemy hit, or when off-screen/out of range/TTL. Prefer circle-vs-circle overlap for bullet↔enemy (cheap and forgiving).
- **Enemy steering:** simplest good-enough is **seek** — velocity toward `(player - enemy).normalize() * speed`. Add **separation** (push apart from nearby enemies) so they don't stack into one blob. For walls, either use physics colliders or simple flow-field/grid pathfinding for larger maps.
- **Broad-phase collision** when counts are high: spatial hash / uniform grid so you only test nearby pairs, not O(n²). Phaser Arcade has a grid/tree internally; hand-rolled Canvas needs its own.
- **Spawn waves** with a timer + difficulty curve (rate and enemy count rising over time). Spawn off-screen at the arena edge, not on top of the player.
- **Cap concurrent entities** to protect frame rate; degrade gracefully (fewer particles) on mobile.

---

## 5. Genre-specific "feel" (the juice)

- **Screen shake** on shooting and explosions (small, decaying).
- **Hit flash** (tint enemy white for 1–2 frames) + **knockback** + **hit-stop** (freeze the game for a few ms on a big hit) — hit-stop is a huge, cheap feel win.
- **Muzzle flash, bullet tracers, and impact particles.**
- **Enemy death**: pop into particles/gibs + a sound + score popup + occasional slow-mo on a big kill.
- **Weapon feel:** spread/recoil for shotguns, fast tiny bullets for SMGs; controller rumble if available.
- **Readable bullets:** player bullets and enemy bullets must be visually distinct (color/size) so the player can dodge — critical in bullet-hell density.
- **Sound** for fire, hit, death, pickup; unlock audio on first gesture.

---

## 6. Common bugs to avoid (checklist)

- **Faster movement on diagonals** → normalize the move vector before scaling by speed.
- **Coupling aim to movement** → they must be independent (that's the whole genre).
- **Aiming off by camera offset/zoom** → convert screen coords to world coords using the camera; don't use raw clientX/Y.
- **Sprite faces wrong direction** → account for the engine's default facing (Phaser sprites face +X/0 rad) and 3D forward-axis convention.
- **No radial deadzone on gamepad** → stick drift makes the player spin/creep.
- **Creating/destroying bullets & enemies each frame** → GC stutter; pool them.
- **O(n²) collision with many entities** → use a spatial grid/hash for broad-phase.
- **Enemies stacking into one sprite** → add separation steering.
- **Enemies spawning on top of the player** → spawn at arena edges/off-screen.
- **Bullets living forever** → give them a TTL/range and despawn off-screen.
- **Movement not delta-scaled** → frame-rate dependent speed.

---

## Defaults to apply

1. **Decouple move and aim by default** — WASD/left-stick move + mouse/right-stick aim; auto-fire on mobile. This one choice defines the genre.
2. **Always normalize the movement vector** so diagonals aren't faster; apply a **radial deadzone** for sticks; convert mouse→world coords for aim angle.
3. **Pool bullets and enemies; use a spatial grid for collision** when counts get high; cap entities on mobile.
4. **Enemy AI = seek + separation** (+ simple wave spawner at edges). Good enough to be fun without pathfinding.
5. **Feel = hit flash + hit-stop + screen shake + score popups + distinct player/enemy bullet colors.** Cheap, huge payoff.
6. **Minimal scope:** one weapon, one enemy type, wave spawner, health/score HUD, game-over/restart. That's a complete loop.

---

## Sources
- MDN — 2D collision detection (circle/AABB): https://developer.mozilla.org/en-US/docs/Games/Techniques/2D_collision_detection
- Red Blob Games — 2D visibility, grids, pathfinding, and vector math: https://www.redblobgames.com/
- Steering Behaviors (Reynolds — seek/flee/separation), classic reference: https://www.red3d.com/cwr/steer/
- Phaser — cameras (follow, deadzone, bounds): https://docs.phaser.io/phaser/concepts/cameras
- Phaser — Groups & object pooling: https://docs.phaser.io/phaser/concepts/gameobjects/group
- nipplejs (mobile virtual joysticks): https://github.com/yoannmoinet/nipplejs
- MDN — Gamepad API (deadzones, axes): https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API
