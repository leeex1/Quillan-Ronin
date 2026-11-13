import wave, os
import numpy as np
from scipy.signal import lfilter

SR = 44100
OUT = r"C:\02_QUILLAN\09 - Projects\thronglets-ronin\game\assets\audio"

def write_wav(name, samples, sr=SR):
    peak = np.max(np.abs(samples)) or 1.0
    samples = samples / peak * 0.72
    data = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(os.path.join(OUT, name), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(data)
    print("wrote", name)

# --- distant wind: 3 cascaded one-pole lowpass (rumble), gust envelope, air band ---
dur = 9.0
white = np.random.normal(0, 1, int(dur * SR))
alpha = 0.985
y = lfilter([1-alpha], [1, -alpha], white)
y = lfilter([1-alpha], [1, -alpha], y)
y = lfilter([1-alpha], [1, -alpha], y)
# gust LFO: slow rounded swells ~0.06 Hz
x = np.arange(len(y)) / SR
lfo = 0.45 + 0.55 * (np.sin(2 * np.pi * 0.06 * x - np.pi / 2) + 1) / 2
lfo = lfo ** 1.4
# add faint airy whistle high-passed component
air = lfilter([1-0.92], [1, -0.92], np.random.normal(0, 1, len(y)))
air = lfilter([1-0.6], [1, -0.6], air)
air = air - lfilter([1-0.6], [1, -0.6], lfilter([1-0.92], [1, -0.92], air))
y2 = y * lfo
mix = y2 + 0.06 * air * lfo
write_wav("wind.wav", mix)

# --- normalize the others to a sane peak ---
for f in ["chirp.wav", "cleanse.wav", "feed.wav", "mitosis.wav", "ascend.wav", "blight.wav", "click.wav", "torii_hum.wav"]:
    with wave.open(os.path.join(OUT, f), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    write_wav(f, x)
