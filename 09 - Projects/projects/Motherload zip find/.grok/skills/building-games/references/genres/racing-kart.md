# Genre Playbook — Arcade Racing / Kart

Arcade racing (Mario Kart, not a sim) is about **forgiving, exciting handling**, satisfying **drift**, and correct **lap/checkpoint** logic. Default engine: **Three.js/Babylon.js** (3D) with a simple arcade car model — do **not** reach for a full rigid-body vehicle sim for a demo. Read `../threejs-foundational.md` first (delta time, follow camera, forward-axis convention, disposal).

---

## 1. Core mechanics (minimal-but-good scope for a demo)

1. **Arcade car handling** — accelerate, brake/reverse, steer; grippy but slidey (see §2).
2. **Drift** with a boost payoff (the fun core of kart racers).
3. **A looping track** with walls/boundaries and ground collision.
4. **Checkpoints + lap counting** that can't be cheated by cutting the course.
5. **A follow (chase) camera** and a HUD (speed, lap X/N, lap time).

One car, one track, 3 laps, a lap timer, and drift-boost is a complete game. Add opponents (even simple waypoint-followers), items, and multiple tracks later. **Handling feel + drift are the whole genre — tune them first.**

---

## 2. Arcade handling (pseudo-physics, not a sim)

Model the car as a point/box with a **heading (yaw)**, a **forward speed**, and a small amount of lateral slide. Avoid full tire-slip simulation.

- **Acceleration:** `speed += throttle * accel * delta`, clamp to `maxSpeed`. Reverse/brake applies negative accel with a lower cap.
- **Drag + rolling resistance:** each frame `speed *= (1 - drag*delta)` (and stronger when off throttle) so the car coasts to a stop. Add higher drag off-track (grass/sand) to punish going wide.
- **Steering scales with speed, not constant:** turn rate should be low at very low speed (can't turn a parked kart much) and taper at very high speed so it isn't twitchy. A common trick: `turn = steerInput * baseTurn * clamp(speed / someSpeed, 0, 1)`. Reverse the steering sign when moving backward.
- **Steer sign (must not invert):** full rules live in the **`controls` skill**
  (`.grok/skills/controls/SKILL.md`) — open it for **any** vehicle, not only
  karts. Summary: while `speed > 0` and the chase camera is behind the car,
  **A / ← turns left**, **D / → turns right**. With
  `forward = (-sin(yaw), 0, -cos(yaw))`, **KeyA must yield +yaw**
  (`steer = +1` on A, then `yaw += steer * turnRate * …`). The classic bug is
  `KeyA → steer = -1` with `yaw += steer * +rate` (A turns right). Run the
  `controls` self-test before shipping; do **not** use FPS strafe as a steer test.
- **Apply yaw, then move along heading:** rotate the car by `turn * delta`, then translate along its forward vector by `speed * delta`. Keep a little **lateral velocity** that decays (grip) so the car feels weighty rather than on rails.
- **Grip model (simple):** split velocity into forward and sideways components; kill most of the sideways component each frame (high grip) — reducing how much you kill it is exactly what creates drift.
- **Keep the car on the ground:** raycast down to the track to set height/normal (handles hills/banking) rather than full suspension. Align the car's up to the surface normal for looks.

---

## 3. Drift (the payoff mechanic)

- **Drift = temporarily reduce lateral grip** (let more sideways velocity survive) while the player holds a drift/handbrake button and steers, so the car slides through the corner with the nose pointing inward.
- **Hop-then-drift (kart style):** a small hop initiates the drift; holding it while turning builds a **drift charge** over time.
- **Mini-turbo / boost payoff:** the longer/tighter the drift, the bigger the speed boost on release (stage it: blue → orange sparks). This risk/reward loop *is* kart racing.
- **Visual/audio feedback:** tire-skid marks (decals), drift particles/sparks that change color with charge, tire-screech sound, and a slight camera FOV kick on boost. Feedback sells the drift.
- **Countersteer feel:** while drifting, let the player modulate the slide angle with steering; snap back to grip smoothly on release (don't instantly zero the lateral velocity — lerp it).

---

## 4. Checkpoints & laps (get this right or laps break)

- **Place ordered checkpoints around the track** (invisible trigger volumes/gates), including a start/finish line. Store the count `N`.
- **Require sequential passing:** track `nextCheckpoint`. A checkpoint only counts if it's the expected next one; passing them out of order (or driving backward) does nothing. This **prevents lap-skipping / reverse-cheesing** — the classic bug where crossing the finish line repeatedly racks up laps.
- **Count a lap** only when the player crosses the finish line *after* hitting all checkpoints for that lap; then reset `nextCheckpoint` to 0.
- **Detect crossing with trigger overlap** (AABB/sphere against the car), and for fast cars use a **swept/segment test** (did the car's path this frame cross the gate plane?) so you don't tunnel through a thin checkpoint at high speed.
- **Respawn / rescue:** if the car flips, leaves the track, or stalls, respawn it at the **last passed checkpoint** facing forward. Also use checkpoints for "wrong way" detection.
- **HUD:** current lap / total, current + best lap time, position. Freeze the timer at finish.

---

## 5. Camera

- **Chase camera behind & slightly above the car**, following with **lerp/spring** (position and look-at both smoothed) so it lags a touch and swings on turns — this conveys speed. `camera.position.lerp(targetBehindCar, k)`; `camera.lookAt(carPosition + lookAhead)`.
- **Speed FOV:** widen FOV slightly at high speed / on boost and narrow when slow — a strong, cheap sense of speed.
- **Don't rigidly parent the camera to the car** or it feels stiff and induces motion sickness; smoothing is essential. Add a tiny bit of positional damping, not rotational snapping.
- Offer a couple of views (chase / hood) if easy. Add motion lines / ground blur / roadside object density for speed sensation.
- Mind the forward-axis convention (camera looks down −Z; car "front" should be +Z or handled via `lookAt`).

---

## 6. Common bugs to avoid (checklist)

- **Lap counter increments on any finish-line crossing** → require all checkpoints in order before counting a lap.
- **Player skips/cuts the track to cheat laps** → sequential-checkpoint gating + wrong-way detection.
- **Car tunnels through walls/checkpoints at speed** → swept/segment collision, not just per-frame overlap; consider fixed timestep.
- **Steering identical at all speeds** → twitchy when fast, unturnable when slow; scale turn rate by speed and flip sign in reverse.
- **A/D reversed while driving forward** → `steerInput` sign disagrees with the
  `forward`/`yaw` basis; re-check the vehicle self-test (A turns left at speed > 0),
  not the FPS D→+X strafe test.
- **Car feels on rails or uncontrollably slidey** → tune the grip (how much lateral velocity you kill per frame); drift = temporarily reduce that.
- **Camera stiff / nausea-inducing** → smooth (lerp/spring) both position and look-at; don't hard-parent.
- **Movement/steer not delta-scaled** → frame-rate dependent handling.
- **Car floats above/sinks into hills** → raycast to ground for height + surface normal each frame.
- **No off-track penalty** → add drag/slowdown on grass so cutting corners costs speed.

---

## Defaults to apply

1. **Use arcade pseudo-physics, not a rigid-body sim:** heading + forward speed + decaying lateral velocity; drag for coasting; steering that scales with speed. Grippy-but-slidey.
2. **Ship drift with a boost payoff** (reduce lateral grip while held + charge → mini-turbo on release, with sparks/skid feedback). This is the genre's hook.
3. **Lap logic MUST use ordered checkpoints:** a lap counts only after all checkpoints are passed in sequence, then the finish line. Prevents skip/reverse cheating. Respawn at last checkpoint.
4. **Use swept collision for walls and checkpoint gates** so fast cars don't tunnel.
5. **Chase camera with smoothing + speed-based FOV** for the sense of speed; never hard-parent.
6. **Minimal scope:** one car, one looping track with checkpoints/laps/timer, drift-boost, HUD. Opponents/items come later.
7. **Steer self-test before done:** `speed > 0`, hold A → turns left; hold D → turns right (chase cam). Fail = not done.

---

## Sources
- "How to make an arcade car / kart handling model" (community write-ups, e.g. Kenney/CodinGame arcade car): https://github.com/spacejack/carphysics2d and demo https://spacejack.github.io/carphysics2d/
- Marco Monster — "Car Physics for Games" (classic arcade/sim reference): https://asawicki.info/Mirror/Car%20Physics%20for%20Games/Car%20Physics%20for%20Games.html
- Three.js — chase/follow camera & `Object3D.lookAt`: https://threejs.org/docs/#api/en/core/Object3D.lookAt
- Three.js — `Raycaster` (ground/height sampling): https://threejs.org/docs/#api/en/core/Raycaster
- Gaffer On Games — Fix Your Timestep (stable vehicle integration / no tunneling): https://gafferongames.com/post/fix_your_timestep/
- Red Blob Games — line-segment intersection (checkpoint crossing tests): https://www.redblobgames.com/
