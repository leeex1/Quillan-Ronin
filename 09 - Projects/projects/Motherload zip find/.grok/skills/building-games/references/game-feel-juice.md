# Game Feel & Juice (screenshake, hitstop, tweening/easing, particles, squash & stretch, camera, SFX layering)

The canon: Steve Swink's *Game Feel*, Vlambeer/"Juice it or lose it" (Martin Jonasson & Petri Purho), and Jan Willem Nijman's "The art of screenshake" (see Sources). "Juice" = generous, layered, non-gameplay-affecting feedback that makes every input feel physical and satisfying. Focus: what an AI builder should reflexively add so games feel **alive instead of static**, without hurting readability or perf.

---

## 1. The core principle

**Every meaningful action should produce disproportionate, multi-sensory feedback.** A single "hit" should simultaneously trigger several of: sound, particles, screenshake, hitstop, flash, knockback, squash/stretch, and a number/pop. The gameplay math can stay simple — the *feel* comes from the feedback layer stacked on top. The classic demo takes the same identical Breakout and makes it feel amazing purely by adding these layers.

**Rule:** Separate "simulation" from "presentation." Feedback (shake, tween, particles, flash) must **never change gameplay outcomes** — it's cosmetic. This keeps physics deterministic (important for netcode/replays) while letting you pile on juice freely.

---

## 2. Screenshake (the highest impact-per-effort effect)

- Offset the **camera** (or a camera container), not the world objects. Apply a random offset each frame from a **trauma value** that **decays over time**.
- **Use trauma², not linear** (Squirrel Eiserloh's "Juicing Your Cameras"): store `trauma` in [0,1], set `shake = trauma * trauma` (or `trauma³`). Add trauma on events (small hit +0.3, explosion +0.8), decay `trauma -= decay * dt` each frame. Squaring makes small hits subtle and big hits explosive.
- Offset with **noise, not pure random**, for smoother shake: `offsetX = maxOffset * shake * (noise(t) )`; also shake **rotation** slightly (`maxAngle * shake * noise`) — rotational shake reads as more violent.
- **Directional shake** for directional hits (kick the camera opposite the impact) reads better than omnidirectional for melee/recoil.
- **Cap it & scale it down for accessibility.** Excessive shake causes motion sickness — expose a "screen shake" slider/toggle. Never shake so hard UI/targets become unreadable.
- Scale offsets by `dt`-independent amount but decay by `dt` so it's frame-rate independent.

Nijman's rules also include: **more bullets, bigger bullets, muzzle flash, impact effects, permanent decals, camera lerp/lookahead, camera kick, sleep/hitstop, gun delay, knockback, screen shake, and even a little randomness in everything.**

---

## 3. Hitstop / hitstun / "sleep" (freeze frames on impact)

- On a big hit, **freeze the game (or just the two involved entities) for a few frames** (~30–120ms), then resume. This sells impact enormously — the brain reads the pause as force.
- Implement as a global **time scale** or a short "freeze frames" counter: while frozen, skip the gameplay update (or set `timeScale=0`) but **keep rendering** and keep the flash/particles visible. Common: freeze ~2–6 frames for normal hits, longer for finishers.
- Keep hitstop **short**; too long feels laggy/unresponsive. Often paired with a brief hit **flash** (tint the sprite white for 1–2 frames) and knockback that starts *after* the freeze.
- Distinguish **hitstop** (brief freeze for feel) from **hitstun** (gameplay state where a hit character can't act) — the latter is real gameplay, the former is pure juice.

---

## 4. Tweening & easing (motion that isn't linear)

- **Almost nothing should move linearly.** Real, satisfying motion accelerates/decelerates. Use **easing functions** for UI transitions, pickups flying to the HUD, menus, popups, camera moves, damage numbers.
- Common curves (see easings.net for exact formulas/graphs):
  - **easeOutQuad/Cubic** — arrive gently (great for things settling into place).
  - **easeInQuad/Cubic** — start slow (things launching).
  - **easeOutBack** — slight overshoot then settle (pop-in for UI/pickups — very juicy).
  - **easeOutElastic / easeOutBounce** — springy/bouncy (buttons, coin pickups). Use sparingly.
  - **easeInOutQuad** — smooth both ends (camera, generic).
- **Use a tween library** rather than hand-rolling: **GSAP** (best-in-class, huge easing set, timelines), **@tweenjs/tween.js** (lightweight, common in Three.js), or the engine's built-in (Phaser `this.tweens.add`). Tweens compose (chains, delays, yoyo, repeat) and auto-handle dt.
- **Kill/clean up tweens** when the target is destroyed or the scene shuts down (leaked tweens mutating dead objects = crashes/bugs). Phaser: kill on `SHUTDOWN`; GSAP: `.kill()`.
- A **spring/damped-lerp** (`x += (target - x) * (1 - exp(-k*dt))`) is a great procedural alternative for continuous following (cameras, cursors, UI) — always frame-rate correct if you use the `exp` form rather than `x += (target-x)*0.1` (the naive lerp is frame-rate dependent — see §6).

---

## 5. Particles & squash-and-stretch

- **Particles everywhere:** dust on landing/footsteps, sparks on hit, debris on death, trails on projectiles, confetti on win, ambient motes. They cost little and add enormous life. Use the engine's particle system (Phaser `add.particles`, Three.js points/instanced sprites) and **pool** them (no per-emit allocation — see `threejs-foundational.md`).
- **Squash & stretch** (Disney's foundational animation principle): deform on acceleration to convey weight/speed while preserving apparent volume.
  - Jump: stretch vertically (taller/thinner) on takeoff, squash (shorter/wider) on landing, then spring back via `easeOutBack`/elastic.
  - Preserve volume: if you scale `y` by `s`, scale `x` by ~`1/s`, so it doesn't look like it's growing.
  - Drive it from velocity (stretch along the movement direction) or trigger on discrete events (land, hit, shoot recoil).
- **Anticipation + follow-through:** wind up before a big action (tiny reverse/squash) and overshoot after — even a couple frames sells it.
- **Pop numbers & flashes:** floating damage numbers that rise + fade + scale-pop, combo counters, "+10" pickups. Cheap, huge readability + satisfaction win.

---

## 6. Camera feel (lerp, lookahead, deadzone, punch)

- **Never hard-snap the camera to the player.** Smoothly follow (`lerp`/spring) toward the target so motion is fluid.
  - **Frame-rate-correct smoothing:** `pos += (target - pos) * (1 - Math.exp(-k * dt))`. The naive `pos += (target - pos) * 0.1` is **frame-rate dependent** (faster on 144Hz than 60Hz) — a common bug; use the exp form or a fixed-timestep camera update.
- **Lookahead:** offset the camera slightly in the direction of movement / where the player is aiming, so the player sees more of what's ahead. Ease the offset in/out.
- **Deadzone / soft zone:** don't move the camera until the player leaves a central box — prevents nauseating micro-jitter from small movements.
- **Camera punch/kick** on impacts (a quick zoom-in or positional kick that eases back) complements screenshake.
- Clamp the camera to level bounds; round the final camera position to whole pixels for pixel-art games to avoid shimmer.

---

## 7. SFX layering & audio feel

- **Layer sounds** for richness: a single "shoot" = a low thump + a mid body + a high click/transient. Impacts = hit + debris + a tail.
- **Pitch-randomize** repeated sounds (±5–15%) so rapid fire / footsteps don't sound robotic (`playbackRate` / Howler `rate`). Slightly randomize volume too.
- Sound is feedback: **every** action should have audio, and it should be tight (low latency) and satisfy on first frame of the action (see the audio skill for Web Audio/Howler, mobile unlock, buses).
- Duck music briefly under big events; add a subtle low-frequency "boom" on explosions.

---

## 8. Don't over-juice (readability & perf guardrails)
- **Juice must not obscure gameplay.** If shake/flash/particles hide the player, enemies, or projectiles, dial back. Readability > spectacle.
- **Accessibility:** provide toggles/sliders for screen shake, flashing, and reduced motion (respect `prefers-reduced-motion`). Heavy flashing risks photosensitivity — avoid rapid full-screen strobing.
- **Perf:** pool particles/tweens, cap concurrent particles, don't allocate in the hit path. Juice is cheap but "thousands of unpooled particles per explosion" is not.
- Keep hitstop short and prediction-safe in multiplayer (apply juice on the client only; never let it alter the authoritative sim).

---

## Defaults to apply
- **Auto-juice defaults:** whenever the builder generates a hit/pickup/death/land/win event, reflexively attach the stack: **sound + particles + short hitstop + white flash + screenshake(trauma²) + easeOutBack pop / squash-stretch + floating number.** Same simple gameplay, dramatically better feel — this is the single biggest perceived-quality lever.
- **Ship a tiny reusable "juice" toolkit** in generated games: a `trauma`-based shake (squared, noise-driven, decaying), a `hitstop(frames)` helper (timeScale/freeze-count), a `flash(sprite)` helper, easing/tween helpers (or GSAP/tween.js), and frame-rate-correct camera follow with lerp + lookahead + deadzone.
- **Always use the `exp`-based lerp** for cameras/followers so smoothing is frame-rate independent — bake this in to avoid the classic 144Hz-vs-60Hz bug.
- **Separate presentation from simulation** so juice never affects gameplay/netcode. Provide screenshake/flash/reduced-motion toggles by default for accessibility.

---

## Sources
- Martin Jonasson & Petri Purho — "Juice it or lose it" (GDC talk): https://www.youtube.com/watch?v=Fy0aCDmgnxg ; companion "Game feel" write-up on grapefrukt.
- Jan Willem Nijman (Vlambeer) — "The art of screenshake": https://www.youtube.com/watch?v=AJdEqssNZ-U
- Steve Swink — *Game Feel: A Game Designer's Guide to Virtual Sensation* (book), gamefeelbook.com.
- Squirrel Eiserloh — "Juicing Your Cameras With Math" (GDC, trauma² screenshake): https://www.youtube.com/watch?v=tu-Qe66AvtY
- easings.net — easing function reference with formulas & graphs: https://easings.net/
- Robert Penner's easing equations (origin of the standard easings): http://robertpenner.com/easing/
- GSAP docs (tweening/easing/timelines): https://gsap.com/docs/v3/ ; tween.js: https://github.com/tweenjs/tween.js
- Disney's 12 principles of animation (squash & stretch, anticipation, follow-through): https://en.wikipedia.org/wiki/Twelve_basic_principles_of_animation
- MDN — `prefers-reduced-motion`: https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-reduced-motion
- "The Juice Factor" / GDC Vault talks on feedback & game feel.
