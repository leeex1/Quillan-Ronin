import sys
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
                   check=True)
    from huggingface_hub import snapshot_download

DST = r"C:\02_QUILLAN\training_data\hf_samurai"
Path(DST).mkdir(parents=True, exist_ok=True)
for f in ["quillan_corpus_CLEAN_V7.jsonl", "instruct_train.jsonl",
          "code_train.jsonl", "tokenizer.json", "tokenizer_config.json"]:
    print(f"fetch {f} ...", flush=True)
    snapshot_download(repo_id="CrashOverrideX/Quillan_Samurai_sets",
                      repo_type="dataset", local_dir=DST,
                      allow_patterns=[f])
    print(f"done {f}", flush=True)
print("ALL DONE", flush=True)
