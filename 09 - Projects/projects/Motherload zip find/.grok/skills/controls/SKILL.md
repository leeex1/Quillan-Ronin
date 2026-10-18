---
name: controls
description: >
  Player-facing input signs for browser games: WASD, vehicles, flight, FPS
  mouse-look, and the #1 failure mode (inverted A/D). Mandatory control
  self-tests and a tiny test interface so you can verify A turns left before
  shipping. Load for ANY game with movement, steering, flying, driving, tanks,
  boats, mechs, drones, planes, karts, third-person follow cams — not only
  racing. Triggers on "controls", "WASD", "inverted", "steer", "flight",
  "airplane", "kart", "vehicle", "yaw", "roll", "pitch", "mouse look", "A/D".
metadata:
  short-description: "Control signs, inverted A/D fix, vehicle/flight maps, mandatory self-test"
user-invocable: false
---

# Controls (player-visible signs — do not ship inverted)

**Read this end-to-end before writing movement/steer/flight code** for any game
that uses WASD, arrows, a chase camera, or a flying craft. Do **not** skip this
and only open `racing-kart` or `fps` — those genre files assume you already
know these signs. Inverted A/D is the most common ship-blocker in vehicle and
flight demos.

Pair with **`building-games`** (loop, camera, 3D orientation) and
**`building-games/references/input.md`** (keydown state, gamepad, touch). This
skill owns **what left/right/up mean to the player** and **how you prove it**.

---

## 0. Hard rules (fail the build if broken)

1. **Player-visible left/right is law.** From a **chase / behind** camera while
   the craft moves **forward**:
   - **A / ←** → nose (or bank) turns **left on screen**
   - **D / →** → nose (or bank) turns **right on screen**
2. **Never reuse FPS strafe as vehicle steer.** FPS “D → +right vector” is
   **position** on the ground plane. Vehicle A/D is **yaw (or roll) rate**, not
   a strafe offset. Mixing them is the #1 cause of inverted A/D.
3. **You must run a control self-test (§5) before saying done.** Screenshot-only
   QA is not enough. If A turns right, **flip the steer/roll sign**, retest,
   then ship — do not invent a new coordinate story.

---

## 1. Shared 3D basis (use this everywhere)

three.js: right-handed, **+Y up**, meshes face **+Z**, cameras look **−Z**.

**Yaw-only heading on XZ** (ground vehicles, walkers, most arcade craft):

```
// yaw = 0 faces world −Z; +yaw is CCW about +Y (nose moves toward −X)
forward = (-sin(yaw), 0, -cos(yaw))
right   = ( cos(yaw), 0, -sin(yaw))   // = normalize(cross(forward, worldUp))
```

With a chase cam **behind** the craft (camera near `position - forward * dist`):

| Player sees | World (this basis) | Input |
|-------------|--------------------|--------|
| Nose left   | **+yaw**           | **A / ←** must produce **+yaw** (or equivalent bank-left for planes) |
| Nose right  | **−yaw**           | **D / →** must produce **−yaw** |

If your basis differs, keep **one** consistent pair — but the **player-visible**
row above is mandatory.

---

## 2. Genre maps

### 2a. FPS / on-foot (strafe, not steer)

```
W = +forward, S = −forward, D = +right, A = −right   // position, not yaw
mouse: yaw -= movementX * sens; pitch -= movementY * sens; clamp pitch
```

Body yaw for look; **movement uses yaw only** (do not apply pitch to walk).

### 2b. Ground / water vehicle (kart, bike, jetski, boat, tank, rover, snowmobile)

Arcade body with `heading`/`yaw` and forward `speed`:

```js
// Input (held keys → actions once per frame)
let steer = 0; // -1..+1, player-visible
if (keys.has('KeyA') || keys.has('ArrowLeft'))  steer += 1;  // LEFT
if (keys.has('KeyD') || keys.has('ArrowRight')) steer -= 1;  // RIGHT
// Optional: steer = clamp(steer + gamepadX, -1, 1) with same sign convention

// Integrate (speedFactor ~ 0..1 from |speed|)
const reverse = speed >= 0 ? 1 : -1; // wheel-left still feels left in reverse
yaw += steer * turnRate * speedFactor * reverse * dt;

// Move along heading
const fx = -Math.sin(yaw), fz = -Math.cos(yaw);
position.x += fx * speed * dt;
position.z += fz * speed * dt;
```

**Canonical bug (do not copy):**

```js
// WRONG — this is what ships inverted A/D in the wild
if (KeyA) steer -= 1;
if (KeyD) steer += 1;
yaw += steer * turnRate * dt;  // A → −yaw → nose RIGHT on chase cam
```

If you already wrote `KeyA → steer--`, either **swap the key mapping** or
**negate once at integrate** (`yaw += -steer * …`) — then run §5. Do not flip
twice (keys + integrate + bank mesh).

### 2c. Fixed-wing flight (airplane, glider, RC plane)

| Input | Action | Player expectation |
|-------|--------|--------------------|
| **A / ←** | **Roll left** (aileron) | Left wing down / bank left |
| **D / →** | **Roll right** | Right wing down / bank right |
| **W / ↑** | Pitch (pick one scheme and label HUD) | Usually nose down *or* pull-up — be consistent |
| **S / ↓** | Opposite pitch | |
| **Q / E** | Yaw / rudder (optional) | Q left, E right |

- **A/D are not strafe** and not “ground steer with FPS signs.”
- Apply roll in the craft’s **local forward axis** with a sign that matches
  **bank left on A**. If the mesh banks the wrong way, flip **one** sign on the
  roll apply (or on the A/D → roll mapping), not the whole basis.
- Coordinated turn: positive bank should produce a turn that matches the bank
  direction under the chase/external cam.

### 2d. Heli / drone / 6DOF

Document the scheme on a start overlay. Minimum:

- Throttle/altitude on discrete keys must **not** stick “always up” after one
  press (use held state or explicit up/down).
- Strafe/yaw: **A left, D right** in the craft’s horizontal frame (player-visible).

### 2e. 2D side-scroller / platformer

**D / →** moves **right on screen**; **A / ←** moves **left**. Gravity only
inverted if the genre is explicitly upside-down.

---

## 3. Camera must agree

- Chase cam: `desired = craftPos + up*height + forward*(-followDist)`; lerp;
  `lookAt(craft)`.
- Compute `forward` **once** for both movement and camera — do not rebuild with
  opposite yaw sign in the camera path.
- Debug order: (1) keys register → (2) signs correct (§5) → (3) camera agrees.

---

## 4. Input plumbing (brief)

- Track keys with a `Set` on `keydown`/`keyup` using **`event.code`**; clear on
  `blur` / `visibilitychange`. Move in the game loop with **dt**, not in the
  key handler.
- Unify keyboard + touch + gamepad into **actions** (`throttle`, `steer`,
  `pitch`, `roll`, …). See `building-games/references/input.md`.
- Touch: left stick = move/steer, right = actions; ≥44px targets.

---

## 5. Mandatory control self-test (before “done”)

Screenshot-only is **not** enough for any craft with A/D.

### 5a. Player-visible checklist (every vehicle / flight build)

While **moving forward** (speed > small epsilon), chase cam behind:

| Hold | Must observe within ~0.5s |
|------|---------------------------|
| **A** | Nose or bank moves **left** on screen |
| **D** | Nose or bank moves **right** on screen |
| **W** (ground) | Speed increases / moves along facing |
| **S** (ground) | Brakes or reverse (as designed) |

If A fails: flip steer/roll sign **once**, retest both A and D.

### 5b. Minimal test interface (implement this)

Expose a tiny hook so you (and automated QA) can prove signs without guessing
private closures:

```js
// e.g. src/game/controlsTest.ts — dev/QA only is fine
export type ControlsProbe = {
  getYaw: () => number;       // radians; or getHeading()
  getSpeed: () => number;
  /** Inject held actions instead of real keys; both stay applied until you
   *  change them, so §5c can hold a key across frames and clear at the end. */
  setSteer?: (v: number) => void; // -1..1, same sign as production
  setKeys?: (codes: string[]) => void; // held until the next call; `[]` clears
};

declare global {
  interface Window {
    __controlsTest?: ControlsProbe;
  }
}
```

Wire `window.__controlsTest` from the game loop when `import.meta.env.DEV` or a
`?qa=1` flag is set.

### 5c. Automated smoke (run it)

Drive the §5b probe with the preinstalled **`agent-browser`** CLI — that is the
first move, not a hand-written script. **A thrown `eval` exits non-zero; a
merely falsy one does not**, so assert by throwing. Run it as one `batch` — one
CLI call, not one per verb — taking JSON on stdin; `--bail` stops at the first
failing step and exits non-zero.

```bash
agent-browser batch --bail <<'JSON'   # find's label is case-sensitive: copy it from `snapshot -i`
[["open","http://127.0.0.1:8080/"],
 ["find","text","Start","click"],
 ["eval","if (!window.__controlsTest?.setKeys) throw Error('no §5b probe: add setKeys')"],
 ["eval","__controlsTest.setKeys(['KeyW'])"],
 ["wait","600"],
 ["eval","if (__controlsTest.getSpeed() <= 0.1) throw Error('W: no move')"],
 ["eval","(async () => { const t = __controlsTest, y0 = t.getYaw(); t.setKeys(['KeyW','KeyA']); await new Promise(r => setTimeout(r, 300)); t.setKeys(['KeyW']); const d = t.getYaw() - y0, w = Math.atan2(Math.sin(d), Math.cos(d)); if (w < -0.05) throw Error('A turns RIGHT — inverted, got ' + w.toFixed(2)); if (w <= 0.05) throw Error('A barely turned (' + w.toFixed(2) + ') — hold longer or check the sim is running'); return 'A ok' })()"],
 ["eval","__controlsTest.setKeys([])"],
 ["screenshot","/workspace/screenshots/controls.png"],
 ["close"]]
JSON
```

**Hold keys through the probe, not with `keydown`** — `agent-browser keydown`
does not reach §4's `Set`, so a correct game reads as broken. No `setKeys` on
your probe? Add it (§5b), or dispatch the key event yourself — the browser-QA
reference's **Keys** note has the mechanism and the form.

The hold sits **inside** one `eval`: spread across commands it lasts however
long they take, and the ±π wrap in `Math.atan2(Math.sin(d), Math.cos(d))`
reads a craft that turned more than π as one turning the other way. The IIFE is
what keeps the step re-runnable — a bare `const d = …` fails the second time
with "already declared" — and `async` is what lets the hold run in page time.
Repeat for **D** with `'KeyD'` and both comparisons mirrored (`w > 0.05` is
inverted, `w >= -0.05` is barely), single-quoted so the JSON needs no escapes;
for planes assert **roll**: A ⇒ bank left.

**`A turns RIGHT`** is the sign error: flip **one** sign and re-run.
**`A barely turned`** is not — the craft moved the right way, just less than
0.05 rad in 300 ms, which a boat or a heavy rover will do; raise the hold, or
check the sim is running, and flip nothing. Any other non-zero exit — no probe,
a wedged daemon — means the check never ran: read the message first. Read the
browser-QA reference `AGENTS.md` links before your first flow — verbs, argument
shapes and the fallback are there.

### 5d. What not to do

- Do **not** only test “D increases some internal variable.”
- Do **not** use FPS “D → +X when facing −Z” as the vehicle pass condition.
- Do **not** flip mesh bank, camera, and steer all at once when fixing — change
  **one** sign, retest.

---

## 6. Finish criteria (controls)

- [ ] Opened **this** skill before writing movement/steer/flight.
- [ ] Genre map chosen (§2) and start-screen / HUD labels match it.
- [ ] Chase-cam A/D player-visible test **passed** (§5a).
- [ ] `window.__controlsTest` (or equivalent) available in dev/QA and used once.
- [ ] No inverted bank mesh relative to roll/steer input.
- [ ] Keys are held-state + dt-scaled; no sticky thrust from a single Space tap
      unless intentional and labeled.

If any box is unchecked, the game is **not** done.
