import React, { useRef, useState } from 'react';

// Browser-native audio graph: AudioContext -> per-voice gain -> master gain ->
// destination. Scheduling uses a look-ahead clock for sample-accurate timing.
type TrackName = 'Kick' | 'Snare' | 'Hat' | 'Bass';
const TRACKS: TrackName[] = ['Kick', 'Snare', 'Hat', 'Bass'];
const STEPS = 16;

const emptyPattern = (): boolean[][] => TRACKS.map(() => Array(STEPS).fill(false));

const DAWStudio: React.FC = () => {
  const [pattern, setPattern] = useState<boolean[][]>(() => {
    const initial = emptyPattern();
    initial[0][0] = true; initial[0][8] = true;         // kick
    initial[1][4] = true; initial[1][12] = true;        // snare
    for (let i = 0; i < STEPS; i += 2) initial[2][i] = true; // hat
    initial[3][0] = true; initial[3][6] = true; initial[3][10] = true; // bass
    return initial;
  });
  const [bpm, setBpm] = useState(120);
  const [playing, setPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const contextRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<number | null>(null);
  const nextNoteRef = useRef(0);
  const stepRef = useRef(0);
  const patternRef = useRef(pattern);
  const bpmRef = useRef(bpm);
  patternRef.current = pattern;
  bpmRef.current = bpm;

  const ensureContext = (): AudioContext => {
    if (!contextRef.current) {
      const Ctor = (window as any).AudioContext || (window as any).webkitAudioContext;
      const ctx: AudioContext = new Ctor();
      const master = ctx.createGain();
      master.gain.value = 0.25;
      master.connect(ctx.destination);
      (ctx as any)._master = master;
      contextRef.current = ctx;
    }
    return contextRef.current;
  };

  const voice = (ctx: AudioContext, track: number, time: number) => {
    const master = (ctx as any)._master as GainNode;
    if (track === 3) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.value = 110;
      gain.gain.setValueAtTime(0.18, time);
      gain.gain.exponentialRampToValueAtTime(0.001, time + 0.18);
      osc.connect(gain); gain.connect(master);
      osc.start(time); osc.stop(time + 0.2);
      return;
    }
    const length = track === 0 ? 0.18 : track === 1 ? 0.12 : 0.04;
    const buffer = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * length), ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) {
      const t = i / ctx.sampleRate;
      if (track === 0) {
        data[i] = Math.sin(2 * Math.PI * (120 - 90 * t / length) * t) * Math.exp(-t * 18);
      } else {
        data[i] = (Math.random() * 2 - 1) * Math.exp(-t * (track === 1 ? 24 : 55));
      }
    }
    const source = ctx.createBufferSource();
    const gain = ctx.createGain();
    gain.gain.value = track === 2 ? 0.12 : 0.3;
    source.buffer = buffer;
    source.connect(gain); gain.connect(master);
    source.start(time);
  };

  const scheduler = () => {
    const ctx = contextRef.current;
    if (!ctx) return;
    const secondsPerStep = 60 / bpmRef.current / 4;
    while (nextNoteRef.current < ctx.currentTime + 0.1) {
      const step = stepRef.current;
      patternRef.current.forEach((row, track) => {
        if (row[step]) voice(ctx, track, nextNoteRef.current);
      });
      setCurrentStep(step);
      nextNoteRef.current += secondsPerStep;
      stepRef.current = (step + 1) % STEPS;
    }
  };

  const toggle = () => {
    if (playing) {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      timerRef.current = null;
      contextRef.current?.suspend();
      setPlaying(false);
      return;
    }
    const ctx = ensureContext();
    void ctx.resume();
    nextNoteRef.current = ctx.currentTime + 0.05;
    stepRef.current = 0;
    timerRef.current = window.setInterval(scheduler, 25);
    setPlaying(true);
  };

  const flip = (track: number, step: number) => {
    setPattern(prev => prev.map((row, index) =>
      index === track ? row.map((value, column) => column === step ? !value : value) : row
    ));
  };

  return (
    <div style={{ padding: 12 }}>
      <h3 style={{ marginTop: 0 }}>Sonic Surge — Step Sequencer</h3>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <button type="button" onClick={toggle}>{playing ? 'Stop' : 'Play'}</button>
        <label>
          BPM
          <input type="number" min={60} max={200} value={bpm}
                 onChange={event => setBpm(Number(event.target.value) || 120)} />
        </label>
        <span style={{ fontSize: 12, opacity: 0.75 }}>step {currentStep + 1}/{STEPS}</span>
      </div>
      <div style={{ display: 'grid', gap: 4 }}>
        {pattern.map((row, track) => (
          <div key={TRACKS[track]} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 48 }}>{TRACKS[track]}</span>
            {row.map((on, step) => (
              <button key={step} type="button" onClick={() => flip(track, step)}
                style={{
                  width: 24, height: 24,
                  background: on ? '#00d5ff' : (step === currentStep ? '#37474f' : '#1e1e1e'),
                  color: on ? '#000' : '#fff', border: '1px solid #455',
                }}>
                {step + 1}
              </button>
            ))}
          </div>
        ))}
      </div>
      <p style={{ fontSize: 12, opacity: 0.7 }}>
        Native Web Audio graph · look-ahead scheduling · no external audio SDK
      </p>
    </div>
  );
};

export default DAWStudio;
