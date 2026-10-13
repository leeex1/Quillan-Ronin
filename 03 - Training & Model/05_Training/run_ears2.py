import pathlib, sys
sys.path.insert(0, r"C:\02_QUILLAN\Audio Engineer")
import importlib.util
spec = importlib.util.spec_from_file_location("llm_ears", r"C:\02_QUILLAN\Audio Engineer\LLM Ears.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import whisper
orig_load = whisper.load_model
whisper.load_model = lambda *a, **kw: orig_load(*a, **kw, device="cpu")
engine = mod.SynesthesiaEngine(model_size="tiny")
from pathlib import Path
base = Path(r"C:\02_QUILLAN\audio\Elemental Avionics")
files = sorted(base.glob("*.mp3"))
print(f"found {len(files)}")
# Prioritize locked
locked_names = ["helios", "elemental", "evolution", "bushido", "ronin"]
results=[]
for f in files:
    print(f"\n--- {f.name} ---")
    try:
        tempo, texture = engine.analyze_acoustics(str(f))
        segs = engine.transcribe_and_timestamp(str(f))
        vw = sum(len(s["text"].split()) for s in segs)
        print(f"BPM {tempo} {texture[:25]} vocal {vw} segs {len(segs)}")
        results.append((f.name, tempo, texture, vw))
    except Exception as e:
        print(f"err {e}")
        results.append((f.name, 0, "error", 999))
# Sort: locked first, then vocal asc
def key(r):
    name=r[0].lower()
    is_locked=any(k in name for k in locked_names)
    return (0 if is_locked else 1, r[3])
results_sorted=sorted(results, key=key)
print("\n=== RANKED ===")
for i,r in enumerate(results_sorted):
    print(f"{i+1:2d}. {r[0]:45} BPM{r[1]:6} {r[2][:20]} vocal:{r[3]:3}")
top15=[r[0] for r in results_sorted[:15]]
print("\nTOP15:")
for n in top15: print(n)
Path(r"C:\02_QUILLAN\06_Media\routenote_ready").mkdir(parents=True, exist_ok=True)
Path(r"C:\02_QUILLAN\06_Media\routenote_ready\Elemental_Avionics_15.txt").write_text("\n".join(top15), encoding="utf-8")
print("saved")
