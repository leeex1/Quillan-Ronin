import numpy as np, wave, struct, math, os

SR = 44100
OUT = r"C:\02_QUILLAN\09 - Projects\thronglets-ronin\game\assets\audio"
os.makedirs(OUT, exist_ok=True)

def write_wav(name, samples, sr=SR):
    samples = np.clip(samples, -1.0, 1.0)
    data = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(os.path.join(OUT, name), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data)
    print("wrote", name)


def mix(*layers):
    n = max(len(l) for l in layers)
    out = np.zeros(n)
    for l in layers:
        out[:len(l)] += l
    return out
def env(n, attack=0.02, release=0.3):
    e = np.ones(n)
    a = max(1, int(attack * SR))
    r = max(1, int(release * SR))
    e[:a] = np.linspace(0, 1, a)
    e[-r:] = np.linspace(1, 0, r)
    return e

def tone(freq, dur, harmonics=((1,1.0),), attack=0.01, release=0.25, sr=SR):
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    s = np.zeros_like(t)
    for mult, amp in harmonics:
        s += amp * np.sin(2 * np.pi * freq * mult * t)
    return s * env(len(t), attack, release) / (sum(a for _, a in harmonics) or 1)

def noise(dur, sr=SR):
    return np.random.normal(0, 1, int(dur * sr))

# --- wind ambience: filtered noise, slow LFO ---
n = noise(8.0)
from scipy.signal import lfilter
# simple one-pole lowpass approximation
alpha = 0.955
y = lfilter([1-alpha], [1, -alpha], n)
lfo = 0.7 + 0.3 * np.sin(2 * np.pi * 0.08 * np.arange(len(y)) / SR)
y = y * lfo
y /= np.max(np.abs(y)) * 1.2
write_wav("wind.wav", y * 0.55)

# --- chirp: quick two-tone blip, thrall voice ---
sr = SR
t = np.linspace(0, 0.28, int(0.28 * sr), endpoint=False)
s = np.sin(2*np.pi*880*t) * np.exp(-t*9) + 0.6*np.sin(2*np.pi*1320*t + 0.4) * np.exp(-t*12)
write_wav("chirp.wav", s * env(len(t), 0.005, 0.15) * 0.9)

# --- click: UI tick ---
t = np.linspace(0, 0.07, int(0.07 * sr), endpoint=False)
s = np.sin(2*np.pi*1900*t) * np.exp(-t*60)
write_wav("click.wav", s * 0.7)

# --- feed: warm chime when mouths fed ---
s = mix(tone(523.25, 0.5, ((1,1.0),(2,0.35),(3,0.18)), 0.01, 0.4), tone(783.99, 0.7, ((1,0.9),(2,0.25)), 0.02, 0.5))
write_wav("feed.wav", s * 0.75)

# --- mitosis: deep bell + shimmer ---
s = mix(tone(196.0, 2.2, ((1,1.0),(2,0.5),(3,0.33),(4,0.18)), 0.005, 1.6), tone(1174.66, 1.6, ((1,0.35),(2,0.15)), 0.01, 1.2))
write_wav("mitosis.wav", s * 0.9)

# --- cleanse: purification shimmer ---
s = mix(tone(659.25, 1.1, ((1,0.9),(3,0.3),(5,0.15)), 0.01, 0.9), tone(1318.5, 1.3, ((1,0.4),), 0.02, 1.1))
write_wav("cleanse.wav", s * 0.7)

# --- blight rumble: low dread ---
n2 = noise(2.4)
y2 = lfilter([1-0.97], [1, -0.97], n2)
y2 *= (0.5 + 0.5*np.sin(2*np.pi*0.6*np.arange(len(y2))/SR))**2
y2 /= np.max(np.abs(y2)) * 1.6
lf = tone(55.0, 2.2, ((1,1.0),(2,0.6)), 0.01, 1.8)
write_wav("blight.wav", mix(np.pad(y2*0.9, (0, max(0, len(lf)-len(y2)))), np.pad(lf*0.6, (0, max(0, len(y2)-len(lf))))) * 0.8)

# --- torii hum: sacred gate drone ---
t = np.linspace(0, 6.0, int(6.0 * SR), endpoint=False)
s = np.sin(2*np.pi*73.42*t) * 0.6 + np.sin(2*np.pi*146.83*t) * 0.25 + np.sin(2*np.pi*220.0*t) * 0.1
s *= (0.75 + 0.25*np.sin(2*np.pi*0.12*t))
write_wav("torii_hum.wav", s * 0.5)

# --- game over / ascend: ascension chord ---
notes = [261.63, 329.63, 392.0, 523.25, 659.25]
s = sum(np.pad(tone(f, 3.2, ((1,1.0),(2,0.4),(3,0.2)), 0.01, 2.6), (0, i*int(0.09*SR))) for i, f in enumerate(notes))
write_wav("ascend.wav", s * 0.4)

# --- ui error / refuse: dull thud ---
s = tone(110.0, 0.4, ((1,1.0),(2,0.4)), 0.005, 0.3)
write_wav("refuse.wav", s * 0.8)
