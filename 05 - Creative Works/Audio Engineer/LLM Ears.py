#!/usr/bin/env python3
import os
import glob
import shutil
import tempfile
import warnings
import numpy as np

# External libs
import yt_dlp
import whisper
import librosa

warnings.filterwarnings("ignore")


class SynesthesiaEngine:
    def __init__(self, model_size="base", temp_dir=None):
        """
        model_size: 'tiny', 'base', 'small', 'medium', 'large' (if you have RAM/GPU)
        temp_dir: optional directory to store temporary downloads
        """
        print("[*] Booting Synesthesia Engine...")
        print(f"[*] Loading Whisper model: {model_size} (this may take a moment)...")
        self.whisper_model = whisper.load_model(model_size)
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="synesthesia_")
        # Ensure temp_dir exists
        os.makedirs(self.temp_dir, exist_ok=True)

    def _is_url(self, path_or_url):
        return str(path_or_url).lower().startswith(("http://", "https://"))

    def download_youtube_audio(self, url, output_basename="current_track"):
        """
        Downloads audio from a YouTube URL to temp_dir and returns the path to the mp3 file.
        """
        print(f"[*] Extracting audio from URL: {url}")
        outtmpl = os.path.join(self.temp_dir, f"{output_basename}.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # find produced mp3 file in temp_dir with that basename
        pattern = os.path.join(self.temp_dir, f"{output_basename}.*")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError("yt-dlp did not produce an output file.")
        # prefer mp3 if present
        mp3_files = [f for f in files if f.lower().endswith(".mp3")]
        chosen = mp3_files[0] if mp3_files else files[0]
        print(f"[+] Audio extracted and saved as: {chosen}")
        return chosen

    def analyze_acoustics(self, file_path):
        """
        Returns: tempo (float), texture (string)
        """
        print("[*] Running acoustic analysis (librosa)...")
        # librosa can read many formats (requires ffmpeg for mp3)
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # BPM / tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)  # ensure numeric

        # Spectral centroid (how 'bright' the signal is)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_cent = float(np.mean(cent))

        # Heuristic thresholds — note these are approximate and depend on sr
        if avg_cent < 1500:
            texture = "Heavy, bass-dominant, dark (e.g., Trap, Nu-Metal, Lo-Fi)"
        elif 1500 <= avg_cent <= 2500:
            texture = "Mid-range focused, balanced (e.g., Rock, Boom-Bap, Acoustic)"
        else:
            texture = "Bright, treble-dominant, piercing (e.g., Pop-Punk, Synthwave)"

        return round(tempo, 2), texture

    def transcribe_and_timestamp(self, file_path):
        """
        Uses Whisper to transcribe and returns list of dicts:
            [{"start": float, "end": float, "text": str}, ...]
        """
        print("[*] Running vocal transcription (Whisper)...")
        result = self.whisper_model.transcribe(file_path)
        segments = result.get("segments", [])
        timestamps = []
        for seg in segments:
            timestamps.append(
                {
                    "start": round(seg.get("start", 0.0), 2),
                    "end": round(seg.get("end", 0.0), 2),
                    "text": seg.get("text", "").strip(),
                }
            )
        return timestamps

    def generate_llm_report(self, source, keep_first_n_timestamps=20):
        """
        Main pipeline. 'source' may be a YouTube URL or a local file path.
        Returns the text report (string).
        """
        audio_file = None
        temp_created = False
        try:
            if self._is_url(source):
                audio_file = self.download_youtube_audio(source, output_basename="current_track")
                temp_created = True
            else:
                # local file path; validate exists
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Local file not found: {source}")
                audio_file = source

            tempo, texture = self.analyze_acoustics(audio_file)
            timestamps = self.transcribe_and_timestamp(audio_file)

            # Build report
            lines = []
            lines.append("=" * 60)
            lines.append("🎵 SYNESTHESIA REPORT GENERATED")
            lines.append("=" * 60)
            lines.append(f"Source: {source}")
            lines.append("\n[1] ACOUSTIC PROFILE")
            lines.append(f"- Detected BPM: {tempo}")
            lines.append(f"- Sonic Texture: {texture}")
            lines.append("\n[2] VOCAL & RHYTHMIC TIMELINE")
            # keep first N segments only for brevity
            for seg in timestamps[:keep_first_n_timestamps]:
                lines.append(f"[{seg['start']}s - {seg['end']}s] {seg['text']}")

            report = "\n".join(lines)
            print(report)
            return report

        finally:
            # cleanup temporary files created by this engine
            if temp_created and audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    print(f"[*] Removed temp audio file: {audio_file}")
                except Exception:
                    pass

    def close(self):
        """Remove temp dir if it's empty/created by us."""
        try:
            if os.path.isdir(self.temp_dir):
                # be conservative: only remove if dir is empty
                if not os.listdir(self.temp_dir):
                    os.rmdir(self.temp_dir)
        except Exception:
            pass


if __name__ == "__main__":
    engine = SynesthesiaEngine(model_size="base")
    try:
        target = input("\nEnter YouTube URL or local audio path: ").strip()
        engine.generate_llm_report(target)
    finally:
        engine.close()