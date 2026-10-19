# Unified Input for Browser Games (keyboard + mouse + touch + Gamepad, action mapping, deadzones, buffering)

Consolidated from MDN (Gamepad API, Pointer Events, KeyboardEvent) and fighting/platformer input-design canon (see Sources). Focus: what an AI builder needs so games are **playable on every device with one code path**, controls are rebindable, and input feels responsive (buffering) rather than dropped.

---

## 1. Architecture: one action layer over all devices

**Never scatter `if (keys.KeyW)` gameplay checks through your code.** Build two layers:
1. **Raw device state** — updated from events (keyboard/pointer) and polling (gamepad).
2. **Abstract actions** — a fixed set like `moveX`, `moveY`, `aimX`, `jump`, `attack`, `pause`. Gameplay reads only actions. Any device can drive any action → keyboard, touch, and gamepad all "just work," and rebinding is trivial.

```js
const actions = { moveX:0, moveY:0, jump:false, attack:false /* ... */ };
```

Compute actions once per frame in an `updateInput()` called at the top of the loop, then gameplay consumes `actions` (and derived `justPressed` edges).

---

## 2. Keyboard: use `event.code`, track state, don't act on the event

- **Track pressed keys in a `Set`** on `keydown`/`keyup`; read the set in the loop. **Do NOT run movement/gameplay directly in the event handler** (event repeat rate ≠ frame rate → frame-dependent, laggy movement).
- **Use `event.code`** (physical key, e.g. `'KeyW'`, `'Space'`, `'ArrowLeft'`) not `event.key` (layout/locale-dependent) and not the **deprecated `keyCode`**. `event.code` keeps WASD in the same physical spot on AZERTY/QWERTZ.
- **`preventDefault()` for game keys** (Space, arrows) so the page doesn't scroll; but don't blanket-block everything (leave F5, devtools, tab-out).
- **Clear the key set on `blur`/`visibilitychange`** — otherwise a key held while tabbing out gets "stuck down" forever (very common bug).
- For text-entry vs gameplay, gate input by focus/mode.

```js
const keys = new Set();
addEventListener('keydown', e => { keys.add(e.code); if (GAME_KEYS.has(e.code)) e.preventDefault(); });
addEventListener('keyup',   e => keys.delete(e.code));
addEventListener('blur',    () => keys.clear());
```

---

## 3. Pointer Events: unify mouse + touch + pen

- **Use Pointer Events (`pointerdown/move/up/cancel`) instead of separate mouse + touch listeners.** One API covers mouse, touch, and stylus, with `pointerId` for multitouch and `pressure`/`pointerType`.
- Handle **`pointercancel`** (OS steals the touch) as a release — forgetting it strands buttons/joysticks "held."
- Set CSS **`touch-action: none`** on the canvas and `preventDefault()` to stop scroll/zoom/pull-to-refresh eating input.
- Convert client coords to canvas/world coords using `getBoundingClientRect()` and `devicePixelRatio`; don't assume clientX == canvas pixel.
- Mouse-look for FPS uses the **Pointer Lock API** (`requestPointerLock` on a gesture; read `movementX/Y`) — see the Three.js controls skill.

---

## 4. Touch controls (mobile)

- **Virtual joystick** for movement (e.g. **nipplejs**, static or dynamic) mapped to `moveX/moveY` normalized −1..1; **on-screen buttons** or a right-side aim/tap zone for actions. Split-screen: left = move, right = act/aim.
- **Minimum 44px hit targets** (Apple HIG). Give visual feedback on press.
- Support multitouch: track pointers by `pointerId` so moving the joystick doesn't cancel a jump button.
- Normalize touch → the same **actions** as keyboard/gamepad; apply with delta time.

---

## 5. Gamepad API: poll every frame, use standard mapping + deadzones

- **Poll `navigator.getGamepads()` every frame** in the loop. The `gamepadconnected`/`gamepaddisconnected` events only tell you a pad exists — **do not cache the `Gamepad` object**; it's a snapshot. Get fresh state each frame.
- A pad often only appears **after the user presses a button** ("waking"). Handle connect/disconnect gracefully mid-game.
- Prefer **`gamepad.mapping === 'standard'`** — browsers remap Xbox/PS/etc. to a consistent layout:
  - Buttons: 0=A/Cross, 1=B/Circle, 2=X/Square, 3=Y/Triangle, 4/5=bumpers, 6/7=triggers (**use `.value` 0–1**, analog), 8=Back/Select, 9=Start, 10/11=stick presses, 12–15=D-pad U/D/L/R.
  - Axes (−1..1): 0=LX, 1=LY (**−1 is up**), 2=RX, 3=RY.
- Read digital with `buttons[i].pressed`, analog triggers with `buttons[i].value > threshold`.
- Non-standard pads (`mapping !== 'standard'`) have unpredictable indices — offer a rebinding screen as fallback.
- Haptics via `gamepad.vibrationActuator?.playEffect('dual-rumble', {...})` where supported (progressive enhancement).

### Deadzones (required for analog sticks)
Sticks never rest at exactly 0. Apply a deadzone or you get drift ("character walks by itself").
- **Radial deadzone** (correct for sticks — treat X/Y together, not per-axis) with re-normalization so full tilt still reaches magnitude 1:

```js
function radialDeadzone(x, y, dz = 0.15) {
  const m = Math.hypot(x, y);
  if (m < dz) return { x: 0, y: 0 };
  const scale = ((m - dz) / (1 - dz)) / m; // re-normalize
  return { x: x * scale, y: y * scale };
}
```
- Typical deadzone 0.1–0.25. Also deadzone trigger `.value`. Per-axis deadzones make diagonals feel wrong — prefer radial.

---

## 6. Action mapping & rebinding

- Store bindings as **data** (action → list of physical inputs): `{ jump: ['Space','KeyW', {pad:0}], attack: ['Mouse0', {pad:2}] }`. `updateInput()` resolves bindings into `actions`.
- **Rebinding UI:** capture the next input event, write it into the binding table, persist to localStorage (see `save-persistence.md`). Detect and warn on conflicts.
- Support **multiple simultaneous devices** and per-action multiple bindings (WASD *and* arrows *and* stick all drive `moveX`).
- Keep a **`justPressed` / `justReleased` edge set**: compare this frame's action booleans to last frame's for one-shot actions (jump, shoot, menu confirm). Reading level-triggered state for these causes repeat-fire bugs.

---

## 7. Input buffering (responsiveness / fairness)

- **Buffer discrete actions for a short window** (~100–150ms / ~6–8 frames) so an input pressed slightly before it's actionable still fires. Essential for platformers and fighting games — makes controls feel "tight" instead of "eating" inputs.
  - E.g. jump pressed 4 frames before landing → still jump on landing.
- **Coyote time:** allow jump for a few frames *after* walking off a ledge — pairs with jump buffering; both dramatically improve platformer feel.
- Store buffered inputs with a timestamp/frame stamp; consume when the action becomes valid, and expire after the window.
- **Poll gamepad and sample input at a fixed rate tied to your fixed-timestep** update for deterministic gameplay/netcode; buffering + fixed step is what makes combos/prediction reproducible.

---

## 8. Bug-prevention checklist
- **Movement in the keydown handler** → frame-rate-dependent, jerky; set flags, move in the loop with `dt`.
- **Using `event.key`/`keyCode`** → breaks on non-US layouts / deprecated; use `event.code`.
- **Stuck keys after tab-out** → clear key set on `blur`/`visibilitychange`.
- **Caching the `Gamepad` object** → stale input; call `getGamepads()` every frame.
- **No stick deadzone** → character drifts; apply radial deadzone with re-normalization.
- **Per-axis deadzone** → mushy/wrong diagonals; use radial.
- **Ignoring `pointercancel`** → stuck touch buttons/joysticks on mobile.
- **Page scrolls/zooms during play** → missing `touch-action:none` / `preventDefault`.
- **Repeat-firing one-shot actions** → reading level state instead of `justPressed` edges.
- **"Ate my jump" feel** → no input buffering / coyote time.
- **Gamepad never detected** → user hasn't pressed a button to wake it; handle connect event + prompt.

---

## Defaults to apply
- **Always generate the two-layer input system**: raw devices → normalized `actions`, with `justPressed`/`justReleased` edges. Gameplay reads only actions, so keyboard+mouse, touch (nipplejs + buttons), and gamepad all work from one code path.
- **Bake in the safety defaults**: `event.code`, clear keys on blur, poll gamepad each frame, radial deadzone (~0.15) with re-normalization, `touch-action:none`, handle `pointercancel`.
- **Include jump/input buffering (~120ms) + coyote time** in platformers/action templates by default — biggest feel win for cheap.
- **Make bindings data-driven and persisted** so a rebinding screen is a small add, and auto-detect the active device to show correct button prompts.
- **Semantic signs for move/steer/flight** are **not** defined here — open the
  **`controls` skill** (`.grok/skills/controls/SKILL.md`) so `steer`/`roll`
  mean player-left correctly. This file is plumbing; that skill is the sign
  convention + mandatory A/D self-test.

---

## Sources
- MDN — Gamepad API: https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API ; Using the Gamepad API: https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API/Using_the_Gamepad_API ; Implementing controls: https://developer.mozilla.org/en-US/docs/Games/Techniques/Controls_Gamepad_API
- MDN — `Navigator.getGamepads()`: https://developer.mozilla.org/en-US/docs/Web/API/Navigator/getGamepads ; `Gamepad` / `GamepadButton`: https://developer.mozilla.org/en-US/docs/Web/API/Gamepad
- MDN — Pointer Events: https://developer.mozilla.org/en-US/docs/Web/API/Pointer_events ; `touch-action`: https://developer.mozilla.org/en-US/docs/Web/CSS/touch-action
- MDN — `KeyboardEvent.code` + code values: https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/code ; https://developer.mozilla.org/en-US/docs/Web/API/UI_Events/Keyboard_event_code_values
- MDN — Pointer Lock API: https://developer.mozilla.org/en-US/docs/Web/API/Pointer_Lock_API
- W3C — Gamepad spec (standard mapping): https://w3c.github.io/gamepad/#remapping
- Josh Sunshine / gamepad mapping deep dive: https://adamjones.me/blog/gamepad-mapping/
- nipplejs (virtual joystick): https://github.com/yoannmoinet/nipplejs
- Input buffering & coyote time (game feel): GDC/《Celeste》 & fighting-game input design write-ups, e.g. https://www.gamedeveloper.com/design/how-to-add-input-buffering-to-your-game and Celeste physics notes.
