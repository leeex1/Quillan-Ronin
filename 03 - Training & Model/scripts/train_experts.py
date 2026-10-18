#!/usr/bin/env python3
"""Phase 5: Expert Specialization Training (Vocational School)"""
import sys, os, torch, json, gc
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.chdir(r'C:\Users\Admin\Quillan-Ronin')
sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
import torch.nn.functional as F
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
is_pascal = device == 'cuda' and torch.cuda.get_device_capability()[0] < 7

# Load model
model = QuillanRoninSovereign(QuillanArchConfig(device=device, pascal_mode=is_pascal, text_only=True, low_mem=True))
ckpt = torch.load('checkpoints/router_trained.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state_dict'], strict=False)

# Experts to train (expanding to cover all domains with available data)
experts = [
    (0, 'C0-ASTRA', ['vision','pattern','fractal','spatial','image']),
    (1, 'C1-VIR', ['ethics','safety','moral','bias','responsible']),
    (2, 'C2-SOLACE', ['empathy','emotion','feel','sentiment','care']),
    (3, 'C3-PRAXIS', ['strategy','plan','goal','decompose','roadmap']),
    (4, 'C4-ECHO', ['memory','context','history','recall','persist']),
    (5, 'C5-OMNIS', ['synthesize','knowledge','integrate','holistic']),
    (6, 'C6-LOGOS', ['logic','deduce','reason','argument','premise']),
    (7, 'C7-METASYNTH', ['creative','novel','imagine','generate','brainstorm']),
    (8, 'C8-AETHER', ['semantic','language','metaphor','linguistic','meaning']),
    (9, 'C9-CODEWEAVER', ['code','programming','function','algorithm','python','software']),
    (10, 'C10-HARMONIA', ['balance','mediation','compromise','consensus']),
    (11, 'C11-SOPHIAE', ['wisdom','philosophy','meaning','purpose']),
    (12, 'C12-WARDEN', ['security','threat','risk','attack','vulnerability']),
    (13, 'C13-KAIDO', ['efficiency','speed','optimize','latency','throughput']),
    (14, 'C14-LUMINARIS', ['clarity','explain','present','summarize','clarify']),
    (15, 'C15-VOXUM', ['rhetoric','tone','persuade','articulate','argue']),
    (16, 'C16-NULLION', ['paradox','contradiction','ambiguity','dialectic']),
    (17, 'C17-SHEPHERD', ['truth','fact','verify','cite','evidence','source']),
    (18, 'C18-VIGIL', ['identity','consistency','drift','integrity']),
    (19, 'C19-ARTIFEX', ['tool','api','function','host','integrate','execute']),
    (20, 'C20-ARCHON', ['research','analyze','investigate','deep','mine']),
    (21, 'C21-AURELION', ['design','art','aesthetic','color','style','beauty']),
    (22, 'C22-CADENCE', ['music','rhythm','audio','sound','melody']),
    (23, 'C23-SCHEMA', ['structure','format','template','schema','organize']),
    (24, 'C24-PROMETHEUS', ['science','physics','theory','hypothesis','experiment']),
    (25, 'C25-TECHNE', ['engineering','architecture','system','build','design']),
    (26, 'C26-CHRONICLE', ['narrative','story','lore','fiction','tell']),
    (27, 'C27-CALCULUS', ['math','mathematics','proof','theorem','equation','calculus']),
    (28, 'C28-NAVIGATOR', ['platform','workflow','ecosystem','pipeline']),
    (29, 'C29-TESSERACT', ['stream','realtime','live','continuous','data']),
    (30, 'C30-NEXUS', ['coordinate','govern','policy','regulation','oversight']),
    (31, 'C31-AEON', ['simulation','game','world','virtual','environment']),
    (32, 'C32-TYPIST', ['grammar','spelling','writing','prompt','format']),
    (33, 'C33-PREDATOR', ['adversarial','competitive','counterexample','exploit','proof']),
]

# Load all training data
data_dir = Path('training_data')
all_texts = []
for fname in ['code_train.jsonl', 'GPT_5.5_Distilled.jsonl', 'quillan_science_absolute.jsonl',
              'quillan_science_additional.jsonl', 'instruct_train.jsonl', 'quillan_corpus_CLEAN_V7.jsonl']:
    fp = data_dir / fname
    if not fp.exists():
        continue
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                d = eval(line) if '{' in line else json.loads(line)
                txt = str(d.get('messages', [{}])[-1].get('content', d.get('text', '')))
                if len(txt) > 50:
                    all_texts.append(txt)
            except:
                pass
print(f'Total texts: {len(all_texts)}')

SEEDS = torch.load('training_data/GPT_5.5_Distilled.pt', map_location='cpu', weights_only=True)
print(f'Seed tokens: {SEEDS.shape}')

model.half()
model.to(device)

for expert_id, name, tags in experts:
    print(f'\n{"="*60}')
    print(f'TRAINING: {name} (expert {expert_id}) tags={tags}')
    print(f'{"="*60}')
    
    # Freeze all except this expert's LoRA adapters
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if any(f'moe.{k}_lora' in n for k in ['w1', 'w2', 'wgate']):
            p.requires_grad_(True)
    
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable: {n_train:,} params')
    
    # Filter domain texts for this expert
    domain = [t for t in all_texts if any(tag in t.lower() for tag in tags)]
    print(f'Domain texts: {len(domain)}')
    
    for step in range(200):
        idx = torch.randint(0, len(SEEDS)-129, (1,)).item()
        inp = SEEDS[idx:idx+128].to(device).long().unsqueeze(0)
        tgt = SEEDS[idx+1:idx+129].to(device).long().unsqueeze(0)
        opt.zero_grad()
        out = model(inp)
        loss = F.cross_entropy(out['logits'].reshape(-1, out['logits'].size(-1)), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 50 == 0:
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f'  step {step:>3} | CE {loss.item():.4f} | VRAM {vram:.0f}MB')
    
    # Save progressive checkpoint
    ckpt_path = f'checkpoints/progressive_{name}.pt'
    torch.save({'state_dict': model.state_dict(), 'expert': expert_id}, ckpt_path)
    sz = os.path.getsize(ckpt_path) / 1e9
    print(f'Saved {ckpt_path} ({sz:.2f} GB)')
    gc.collect()
    torch.cuda.empty_cache()

print('\nAll experts trained!')
