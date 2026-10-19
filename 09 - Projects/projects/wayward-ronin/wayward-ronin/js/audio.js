// audio.js — all sound is synthesized with WebAudio. No external files.
// Ambient: crickets + wind. Combat: whooshes, clangs, parries. Music: sparse
// koto-like plucks on a minor pentatonic scale.

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.master = null;
    this.ambientNodes = [];
  }

  // Must be called from a user gesture (title screen button).
  init() {
    if (this.ctx) { if (this.ctx.state === 'suspended') this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.9;
    this.master.connect(this.ctx.destination);
    this.startAmbient();
  }

  setMuted(m) {
    this.muted = m;
    if (this.master) this.master.gain.value = m ? 0 : 0.9;
  }

  now() { return this.ctx ? this.ctx.currentTime : 0; }

  // ---- primitive voices ----
  noiseBurst({ dur = 0.15, freq = 2000, q = 1, gain = 0.5, type = 'bandpass', sweepTo = null, delay = 0 }) {
    if (!this.ctx || this.muted) return;
    const t = this.now() + delay;
    const len = Math.max(1, (dur * this.ctx.sampleRate) | 0);
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const f = this.ctx.createBiquadFilter();
    f.type = type; f.frequency.setValueAtTime(freq, t); f.Q.value = q;
    if (sweepTo) f.frequency.exponentialRampToValueAtTime(sweepTo, t + dur);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(gain, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    src.connect(f); f.connect(g); g.connect(this.master);
    src.start(t); src.stop(t + dur + 0.02);
  }

  tone({ freq = 440, dur = 0.2, gain = 0.3, type = 'sine', sweepTo = null, delay = 0 }) {
    if (!this.ctx || this.muted) return;
    const t = this.now() + delay;
    const o = this.ctx.createOscillator();
    o.type = type; o.frequency.setValueAtTime(freq, t);
    if (sweepTo) o.frequency.exponentialRampToValueAtTime(Math.max(20, sweepTo), t + dur);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g); g.connect(this.master);
    o.start(t); o.stop(t + dur + 0.02);
  }

  // ---- game sounds ----
  swing(heavy = false) {
    this.noiseBurst({ dur: heavy ? 0.28 : 0.16, freq: heavy ? 900 : 2400, sweepTo: heavy ? 200 : 500, gain: heavy ? 0.35 : 0.22, q: 2 });
  }
  hitFlesh(heavy = false) {
    this.noiseBurst({ dur: 0.12, freq: 300, type: 'lowpass', gain: 0.5 });
    this.tone({ freq: heavy ? 90 : 140, sweepTo: 45, dur: 0.16, gain: 0.4, type: 'triangle' });
  }
  clash(parry = false) {
    // metallic clang: inharmonic partials + noise
    const base = parry ? 1240 : 880;
    [1, 1.51, 2.09, 2.94].forEach((m, i) =>
      this.tone({ freq: base * m, dur: parry ? 0.5 : 0.3, gain: 0.16 / (i + 1), type: 'square' }));
    this.noiseBurst({ dur: 0.08, freq: 5200, gain: parry ? 0.3 : 0.18, q: 3 });
  }
  footstep() {
    this.noiseBurst({ dur: 0.07, freq: 220, type: 'lowpass', gain: 0.10 });
  }
  dodge() {
    this.noiseBurst({ dur: 0.22, freq: 1400, sweepTo: 300, gain: 0.18, q: 1.5 });
  }
  uiClick() { this.tone({ freq: 660, dur: 0.06, gain: 0.12, type: 'triangle' }); }
  dialogueBlip() { this.tone({ freq: 320 + Math.random() * 60, dur: 0.045, gain: 0.07, type: 'square' }); }
  choice() { this.tone({ freq: 520, sweepTo: 780, dur: 0.1, gain: 0.14, type: 'triangle' }); }
  repUp() { this.tone({ freq: 392, dur: 0.12, gain: 0.12 }); this.tone({ freq: 523, dur: 0.18, gain: 0.12, delay: 0.09 }); }
  repDown() { this.tone({ freq: 330, dur: 0.12, gain: 0.12 }); this.tone({ freq: 233, dur: 0.2, gain: 0.12, delay: 0.09 }); }
  deathSting() {
    this.tone({ freq: 220, sweepTo: 55, dur: 1.4, gain: 0.35, type: 'sawtooth' });
    this.noiseBurst({ dur: 0.8, freq: 400, type: 'lowpass', gain: 0.25 });
  }
  // sparse koto-like pluck, A minor pentatonic: A C D E G
  pluck(degree = 0, delay = 0) {
    const scale = [220, 261.6, 293.7, 329.6, 392, 440, 523.3];
    const f = scale[((degree % scale.length) + scale.length) % scale.length];
    this.tone({ freq: f, dur: 1.6, gain: 0.10, type: 'triangle', delay });
    this.tone({ freq: f * 2.01, dur: 0.9, gain: 0.035, type: 'sine', delay });
  }
  endingChord() {
    [0, 2, 4, 6].forEach((d, i) => this.pluck(d, i * 0.55));
  }

  // ---- ambient bed: wind + crickets, looping via scheduled re-trigger ----
  startAmbient() {
    if (!this.ctx || this.ambientNodes.length) return;
    // wind: looping filtered noise with slow LFO on the filter
    const len = 4 * this.ctx.sampleRate;
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const src = this.ctx.createBufferSource();
    src.buffer = buf; src.loop = true;
    const f = this.ctx.createBiquadFilter();
    f.type = 'lowpass'; f.frequency.value = 320; f.Q.value = 0.6;
    const lfo = this.ctx.createOscillator();
    lfo.frequency.value = 0.07;
    const lfoGain = this.ctx.createGain();
    lfoGain.gain.value = 170;
    lfo.connect(lfoGain); lfoGain.connect(f.frequency);
    const g = this.ctx.createGain(); g.gain.value = 0.05;
    src.connect(f); f.connect(g); g.connect(this.master);
    src.start(); lfo.start();
    this.ambientNodes.push(src, lfo);
    // crickets: schedule chirp clusters
    const chirp = () => {
      if (!this.muted && this.ctx) {
        const n = 3 + (Math.random() * 4 | 0);
        for (let i = 0; i < n; i++) {
          this.tone({ freq: 4200 + Math.random() * 600, dur: 0.03, gain: 0.018, type: 'sine', delay: i * 0.07 });
        }
      }
      this._cricketTimer = setTimeout(chirp, 900 + Math.random() * 2600);
    };
    chirp();
    // distant sparse plucks for mood
    const mood = () => {
      if (!this.muted && this.ctx && Math.random() < 0.6) this.pluck((Math.random() * 7) | 0);
      this._moodTimer = setTimeout(mood, 9000 + Math.random() * 14000);
    };
    mood();
  }
}
