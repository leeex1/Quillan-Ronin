import wave, numpy as np
import os

D = r"C:\02_QUILLAN\09 - Projects\thronglets-ronin\game\assets\audio"
for f in sorted(os.listdir(D)):
    if not f.endswith(".wav"):
        continue
    with wave.open(os.path.join(D, f), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        continue
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    peak = np.max(np.abs(x))
    rms = np.sqrt(np.mean(x**2))
    dc = np.mean(x)
    # crude spectral centroid
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    centroid = (freqs * X).sum() / (X.sum() + 1e-9)
    clipping = np.mean(np.abs(x) > 0.99) * 100
    print(f"{f:14s} dur={n/sr:5.2f}s ch={ch} peak={peak:.3f} rms={rms:.4f} dc={dc:+.4f} centroid={centroid:7.1f}Hz clip%={clipping:.2f}")
