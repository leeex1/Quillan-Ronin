import numpy as np, wave, os
SR = 44100
OUT = r"C:\02_QUILLAN\09 - Projects\thronglets-ronin\game\assets\audio\ascend.wav"
notes = [261.63, 329.63, 392.0, 523.25, 659.25]
maxlen = int((3.2 + len(notes)*0.09) * SR)
mixbuf = np.zeros(maxlen)
def env(n, attack, release):
    e = np.ones(n)
    a = max(1, int(attack*SR)); r = max(1, int(release*SR))
    e[:a] = np.linspace(0,1,a); e[-r:] = np.linspace(1,0,r)
    return e
def tone(f, dur, harmonics, attack, release):
    t = np.linspace(0, dur, int(dur*SR), endpoint=False)
    s = np.zeros_like(t)
    for m, a in harmonics:
        s += a * np.sin(2*np.pi*f*m*t)
    return s * env(len(t), attack, release) / (sum(a for _, a in harmonics) or 1)
for i, f in enumerate(notes):
    tt = tone(f, 3.0, ((1,1.0),(2,0.4),(3,0.2)), 0.01, 2.4)
    start = i * int(0.09*SR)
    mixbuf[start:start+len(tt)] += tt
mixbuf /= max(1.0, np.max(np.abs(mixbuf))) + 0.2
data = (mixbuf * 0.8 * 32767).astype(np.int16).tobytes()
with wave.open(OUT, "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(data)
print("wrote ascend.wav", len(mixbuf)/SR, "s")
