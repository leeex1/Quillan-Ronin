"""Import wrapper for the space-named `LLM Ears.py` module.

Acoustics (BPM / texture / centroid) run on CPU via librosa — no Whisper.
Whisper is loaded lazily, forced to CPU, only when transcription is requested.
"""

from __future__ import annotations

import os
import numpy as np
from scipy import signal as scipy_signal


class SynesthesiaEngine:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self._whisper = None

    def acoustics(self, path: str):
        import librosa

        y, sr = librosa.load(path, sr=None, mono=True, duration=60.0)
        if len(y) == 0:
            return 120.0, "unknown", 0.0
        raw_tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.asarray(raw_tempo).reshape(-1)[0])
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid = float(np.mean(cent))
        if centroid < 1500:
            texture = "dark / bass-dominant (trap, lofi, nu-metal)"
        elif centroid <= 2500:
            texture = "balanced / mid-focused (rock, boom-bap, acoustic)"
        else:
            texture = "bright / treble-forward (pop, synthwave)"
        return round(bpm, 2), texture, round(centroid, 1)

    def transcribe(self, path: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            if self._whisper is None:
                import whisper

                self._whisper = whisper.load_model(self.model_size, device="cpu")
            result = self._whisper.transcribe(path, fp16=False, device="cpu")
            return [
                {
                    "start": round(s.get("start", 0), 2),
                    "end": round(s.get("end", 0), 2),
                    "text": s.get("text", "").strip(),
                }
                for s in result.get("segments", [])
            ]
        except Exception as exc:  # noqa: BLE001
            return [
                {"start": 0, "end": 0, "text": "transcribe failed: " + str(exc)[:200]}
            ]
