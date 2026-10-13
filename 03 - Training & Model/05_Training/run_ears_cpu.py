import pathlib, sys
sys.path.insert(0, r"C:\02_QUILLAN\Audio Engineer")
import importlib.util
spec = importlib.util.spec_from_file_location("llm_ears", r"C:\02_QUILLAN\Audio Engineer\LLM Ears.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Monkey patch whisper to use CPU
import whisper
orig_load = whisper.load_model
def load_cpu(*a, **kw):
    kw["device"] = "cpu"
    return orig_load(*a, **kw)
whisper.load_model = load_cpu
engine = mod.SynesthesiaEngine(model_size="tiny")  # tiny for CPU speed
from pathlib import Path
base = Path(r"C:\02_QUILLAN\audio\Elemental_Avionics")
files = sorted(base.glob("*.mp3"))
print(f"found {len(files)}")
for f in files[:3]:
    print(f"\n--- {f.name} ---")
    try:
        tempo, texture = engine.analyze_acoustics(str(f))
        segs = engine.transcribe_and_timestamp(str(f))
        vw = sum(len(s["text"].split()) for s in segs)
        print(f"BPM {tempo} {texture} vocal_words {vw} segs {len(segs)}")
        for s in segs[:3]:
            print(s)
    except Exception as e:
        print(f"err {e}")
