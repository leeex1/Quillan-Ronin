// touch.js — virtual joystick + action buttons for touch devices.
// Feeds the SAME `input` object main.js uses for keyboard/mouse, so all game
// logic is shared. Auto-activates only on touch devices; desktop is untouched.
// Dialogue choices are native HTML buttons (already tappable), so no extra
// choice buttons are needed — TALK advances linear dialogue like E does.

export class TouchControls {
  constructor(input, hooks) {
    this.input = input;
    this.hooks = hooks; // { interact(), dodge(), pause() }
    this.active =
      (typeof window !== 'undefined' && 'ontouchstart' in window) ||
      (typeof navigator !== 'undefined' && navigator.maxTouchPoints > 0);
    if (!this.active) return;
    if (document.body && document.body.classList) document.body.classList.add('touch');
    this.joyId = null;
    this.joyOX = 0; this.joyOY = 0;
    this.bindJoystick();
    this.bindButtons();
    const hint = document.getElementById('hud-hint');
    if (hint) hint.textContent = 'Stick: move (push far to sprint) · ATK light · HVY heavy · BLK hold / tap = parry · ROLL dodge · TALK interact';
    const keyEl = document.querySelector('#interact-prompt .key');
    if (keyEl) keyEl.textContent = 'TAP';
  }

  $(id) { return document.getElementById(id); }

  // All touch handlers preventDefault (no scroll / pinch-zoom / double-tap).
  on(el, type, fn) {
    if (!el) return;
    el.addEventListener(type, (e) => { e.preventDefault(); fn(e); }, { passive: false });
  }

  bindJoystick() {
    const zone = this.$('joy-zone'), base = this.$('joy-base'), knob = this.$('joy-knob');
    if (!zone || !base || !knob) return;
    const R = 60, DEAD = 10;
    this.on(zone, 'touchstart', (e) => {
      const t = e.changedTouches[0];
      if (this.joyId !== null) return; // single stick
      this.joyId = t.identifier;
      this.joyOX = t.clientX; this.joyOY = t.clientY;
      base.style.left = t.clientX + 'px';
      base.style.top = t.clientY + 'px';
      base.classList.remove('hidden');
      knob.style.transform = 'translate(-50%,-50%)';
    });
    const move = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier !== this.joyId) continue;
        let dx = t.clientX - this.joyOX, dy = t.clientY - this.joyOY;
        const m = Math.hypot(dx, dy);
        if (m > R) { dx = (dx / m) * R; dy = (dy / m) * R; }
        knob.style.transform = `translate(calc(-50% + ${dx.toFixed(1)}px), calc(-50% + ${dy.toFixed(1)}px))`;
        const inp = this.input;
        inp.f = dy < -DEAD;
        inp.b = dy > DEAD;
        inp.l = dx < -DEAD;
        inp.r = dx > DEAD;
        inp.sprint = m > R * 0.82; // push to the rim to sprint
      }
    };
    const end = (e) => {
      for (const t of e.changedTouches) {
        if (t.identifier !== this.joyId) continue;
        this.joyId = null;
        const inp = this.input;
        inp.f = inp.b = inp.l = inp.r = inp.sprint = false;
        base.classList.add('hidden');
      }
    };
    this.on(zone, 'touchmove', move);
    this.on(zone, 'touchend', end);
    this.on(zone, 'touchcancel', end);
  }

  bindButtons() {
    const inp = this.input, h = this.hooks;
    const btn = (id, down, up) => {
      const el = this.$(id);
      if (!el) return;
      this.on(el, 'touchstart', () => down());
      if (up) {
        this.on(el, 'touchend', () => up());
        this.on(el, 'touchcancel', () => up());
      }
    };
    btn('tb-atk', () => { inp.lightQueued = true; });
    btn('tb-hvy', () => { inp.heavyQueued = true; });
    btn('tb-roll', () => { h.dodge(); });
    btn('tb-talk', () => { h.interact(); });
    btn('tb-pause', () => { h.pause(); });
    // block mirrors the F key exactly: edge on press (parry timing), held while down
    btn('tb-blk',
      () => { if (!inp.blockHeld) inp.blockEdge = true; inp.blockHeld = true; },
      () => { inp.blockHeld = false; });
  }
}
