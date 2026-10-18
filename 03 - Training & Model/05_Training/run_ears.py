import pathlib, os, sys
sys.path.insert(0, r"C:\02_QUILLAN\Audio Engineer")
from pathlib import Path
# Import engine from file
import importlib.util
spec = importlib.util.spec_from_file_location("llm_ears", r"C:\02_QUILLAN\Audio Engineer\LLM Ears.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
engine = mod.SynesthesiaEngine(model_size="base")
base = Path(r"C:\02_QUILLAN\audio\Elemental_Avionics")
files = sorted(base.glob("*.mp3"))
print(f"found {len(files)} files")
locked = {"Helios_Obsidian", "Elemental_Avionics", "EVOlutions", "Bushido_Medly", "EVOlutions", "EEvolution_Avenue", "_Elemental_Avionics_", "_Bushido_Medly_"}
# also add evolutions variants
results=[]
for f in files:
    print(f"\n--- {f.name} ---")
    try:
        tempo, texture = engine.analyze_acoustics(str(f))
        segs = engine.transcribe_and_timestamp(str(f))
        vocal_len = sum(len(s["text"]) for s in segs)
        vocal_words = sum(len(s["text"].split()) for s in segs)
        print(f"BPM {tempo} texture {texture} vocal_words {vocal_words} vocal_len {vocal_len}")
        results.append((f.name, tempo, texture, vocal_words, vocal_len, segs[:2]))
    except Exception as e:
        print(f"err {f.name}: {e}")
        results.append((f.name, 0, "error", 999, 999, []))
# Sort by vocal_words asc, then texture bright > balanced > heavy, then keep locked first
def score(r):
    name, tempo, texture, vw, vl, segs = r
    is_locked = any(k.lower() in name.lower() for k in ["helios", "elemental", "evolution", "bushido", "ronin_med", "eevolution"])
    # lower vocal is better, locked gets -1000
    return ( -1000 if is_locked else 0, vw, vl)

results_sorted = sorted(results, key=lambda x: (0 if any(k.lower() in x[0].lower() for k in ["helios","elemental","evolution","bushido","ronin_med","eevolution"]) else 1, x[3], x[4]))
print("\n=== RANKED 19 -> 15 ===")
for i,r in enumerate(results_sorted):
    print(f"{i+1:2d}. {r[0]:45} BPM{r[1]:6} {r[2][:25]} vocal:{r[3]:3} locked:{any(k.lower() in r[0].lower() for k in ['helios','elemental','evolution','bushido','ronin'])}")
# Pick top 15
top15 = [r[0] for r in results_sorted[:15]]
print("\nTOP15:")
for n in top15: print(n)
# Save
Path(r"C:\02_QUILLAN\06_Media\routenote_ready").mkdir(parents=True, exist_ok=True)
Path(r"C:\02_QUILLAN\06_Media\routenote_ready\Elemental_Avionics_15.txt").write_text("\n".join(top15), encoding="utf-8")
print("saved list")
