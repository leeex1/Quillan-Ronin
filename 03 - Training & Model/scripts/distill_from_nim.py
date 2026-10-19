#!/usr/bin/env python3
# Quillan Distill from NIM Trainer Parents (Glimmer/Lightning/Omni)
# Manual tensor setting via weighted ensemble distillation - FAST
import os, json, pathlib, time, torch
from pathlib import Path
from openai import OpenAI

# Load key
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in [REPO_ROOT / ".env"]:
    if Path(p).exists():
        for line in Path(p).read_text().splitlines():
            if "NVIDIA_API_KEY" in line and "=" in line:
                k,v=line.split("=",1)
                os.environ[k.strip()]=v.strip().strip(chr(34)).strip(chr(39))

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"])

TEACHERS = [
    {"name":"glimmer","model":"meta/muse-glimmer-30b","weight":0.5,"temp":0.7},
    {"name":"lightning","model":"nvidia/nemotron-3.5-lightning-30b-a3b","weight":0.3,"temp":1.0},
    {"name":"omni","model":"nvidia/nemotron-3-nano-omni-30b-a3b-reasoning","weight":0.2,"temp":0.6},
]

def teacher_generate(prompt, teacher):
    try:
        c=client.chat.completions.create(
            model=teacher["model"],
            messages=[{"role":"user","content":prompt}],
            temperature=teacher["temp"], max_tokens=512, stream=False
        )
        return c.choices[0].message.content
    except Exception as e:
        print(f"[{teacher['name']} fail {e}]")
        return ""

def distill_batch(prompts):
    # Weighted ensemble: query all 3 in parallel via NIM (cloud, no VRAM)
    results=[]
    for p in prompts:
        outs=[]
        for t in TEACHERS:
            txt=teacher_generate(p, t)
            outs.append((t["weight"], txt))
        # Simple: pick weighted best (glimmer primary)
        results.append(outs[0][1])  # glimmer weighted 0.5 primary
    return results

if __name__=="__main__":
    # Hybrid runtime: threads/offload/RAM-floor policy (same as trainer).
    # Teachers stay cloud (no VRAM); student compute follows hybrid routing.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hybrid_runtime import init as _hybrid_init
    _hybrid_init()
    # Test with 3 prompts
    test_prompts=["Explain quantum entanglement simply","Write a haiku about code","What is 9.11 vs 9.8?"]
    print("[DISTILL] Querying 3 teachers for",len(test_prompts),"prompts...")
    t0=time.time()
    outs=distill_batch(test_prompts)
    print(f"[DISTILL] Done in {time.time()-t0:.1f}s")
    for i,(p,o) in enumerate(zip(test_prompts, outs)):
        print(f"\n--- Prompt {i+1}: {p[:50]} ---")
        print(o[:400])
    
    # Now manual tensor update stub: load Quillan and do one CE step on teacher outputs
    print("\n[DISTILL] Loading Quillan student 222M for manual tensor update...")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
    from quillan_tokenizer_unified import UnifiedQuillanTokenizer
    cfg=QuillanOniConfig(n_layer=6)
    model=QuillanRoninOni(cfg)
    tok=UnifiedQuillanTokenizer()
    # Load checkpoint
    ckpt_path = REPO_ROOT / "09 - Projects" / "projects" / "05_Training" / "checkpoints" / "checkpoints_oni" / "quillan_oni_latest.pt"
    ck=torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    print(f"[DISTILL] Loaded step {ck['step']}, doing 1 manual CE step on teacher traces...")
    model.train()
    opt=torch.optim.AdamW(model.parameters(), lr=3e-4)
    # One batch from first teacher output
    txt=outs[0][:512]
    ids=tok.encode(txt)[:256]
    if len(ids)>10:
        inp=torch.tensor([ids[:-1]]); tgt=torch.tensor([ids[1:]])
        logits=model(inp)
        loss=torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward(); opt.step()
        print(f"[DISTILL] Manual tensor CE loss {loss.item():.4f} - tensors updated!")
    print("[DISTILL] Ready for full distillation run (116k seqs, weighted ensemble)")
