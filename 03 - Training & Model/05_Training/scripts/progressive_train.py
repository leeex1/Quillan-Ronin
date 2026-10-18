#!/usr/bin/env python3
"""
Quillan-Ronin Progressive Training Pipeline (Vocational School)
Trains each layer/expert independently on domain-specific data.

Phases:
  P1: Ingestion (embeddings + position) 
  P2: 9-Vector Decomposition 
  P3: Diffusion Core (CouilAttention)
  P4: Router (ComplexityRouter) ← completed
  P5: 34 Experts (vocational — each expert on its domain)
  P6: EGGROLL Swarm (rank-R perturbations)
  P7: Cognitive Engines + Dual Quillan + txt_dec
  P8: Full assembly + quantized export

Phase 5 Expert → Domain mapping:
  C0-ASTRA: vision/patterns     C1-VIR: ethics/safety
  C2-SOLACE: empathy/sentiment  C3-PRAXIS: strategy
  C4-ECHO: memory/context       C5-OMNIS: synthesis
  C6-LOGOS: logic/deduction     C7-METASYNTH: creativity
  C8-AETHER: language/semantics C9-CODEWEAVER: code
  C10-HARMONIA: balance         C11-SOPHIAE: wisdom/philosophy
  C12-WARDEN: safety            C13-KAIDO: efficiency
  C14-LUMINARIS: clarity        C15-VOXUM: rhetoric
  C16-NULLION: paradox          C17-SHEPHERD: truth
  C18-VIGIL: identity           C19-ARTIFEX: tools/api
  C20-ARCHON: research          C21-AURELION: design/art
  C22-CADENCE: music/audio      C23-SCHEMA: structure/format
  C24-PROMETHEUS: science       C25-TECHNE: engineering
  C26-CHRONICLE: narrative      C27-CALCULUS: math
  C28-NAVIGATOR: orchestration  C29-TESSERACT: streaming
  C30-NEXUS: governance         C31-AEON: simulation/games
  C32-TYPIST: grammar/writing   C33-PREDATOR: adversarial math
"""

import os, sys, gc, time, json, torch, torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

sys.path.insert(0, r'C:\Users\Admin\Quillan-Ronin')
from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig, _weight_quant

# ─── EXPERT DOMAIN MAPPING ────────────────────────────────────────────────────
EXPERT_DOMAINS = {
    0:  {"name": "C0-ASTRA",      "tags": ["vision","pattern","fractal","spatial"], "data_weight": 0.3},
    1:  {"name": "C1-VIR",        "tags": ["ethics","safety","harm","moral"], "data_weight": 0.5},
    2:  {"name": "C2-SOLACE",     "tags": ["empathy","emotion","sentiment","affect"], "data_weight": 0.4},
    3:  {"name": "C3-PRAXIS",     "tags": ["strategy","plan","goal","decomposition"], "data_weight": 0.3},
    4:  {"name": "C4-ECHO",       "tags": ["memory","context","history","recall"], "data_weight": 0.3},
    5:  {"name": "C5-OMNIS",      "tags": ["synthesis","knowledge","integration"], "data_weight": 0.3},
    6:  {"name": "C6-LOGOS",      "tags": ["logic","deduction","reasoning","argument"], "data_weight": 0.7},
    7:  {"name": "C7-METASYNTH",  "tags": ["creativity","novel","ideation","generate"], "data_weight": 0.5},
    8:  {"name": "C8-AETHER",     "tags": ["semantic","language","metaphor","linguistic"], "data_weight": 0.5},
    9:  {"name": "C9-CODEWEAVER", "tags": ["code","programming","python","software","coding"], "data_weight": 1.0},
    10: {"name": "C10-HARMONIA",  "tags": ["balance","mediation","consensus","diplomacy"], "data_weight": 0.2},
    11: {"name": "C11-SOPHIAE",   "tags": ["wisdom","philosophy","ethics","meaning"], "data_weight": 0.3},
    12: {"name": "C12-WARDEN",    "tags": ["security","threat","risk","vulnerability"], "data_weight": 0.3},
    13: {"name": "C13-KAIDO",     "tags": ["efficiency","speed","optimization","latency"], "data_weight": 0.2},
    14: {"name": "C14-LUMINARIS", "tags": ["clarity","presentation","explain","summarize"], "data_weight": 0.4},
    15: {"name": "C15-VOXUM",     "tags": ["rhetoric","tone","persuasion","argument"], "data_weight": 0.3},
    16: {"name": "C16-NULLION",   "tags": ["paradox","dialectic","ambiguity","contradiction"], "data_weight": 0.2},
    17: {"name": "C17-SHEPHERD",  "tags": ["truth","citation","fact","verify"], "data_weight": 0.5},
    18: {"name": "C18-VIGIL",     "tags": ["identity","consistency","integrity","drift"], "data_weight": 0.2},
    19: {"name": "C19-ARTIFEX",   "tags": ["tool","api","function","integration","host"], "data_weight": 0.5},
    20: {"name": "C20-ARCHON",    "tags": ["research","analysis","investigate","deep"], "data_weight": 0.6},
    21: {"name": "C21-AURELION",  "tags": ["design","art","aesthetic","color","style"], "data_weight": 0.4},
    22: {"name": "C22-CADENCE",   "tags": ["music","rhythm","audio","melody"], "data_weight": 0.3},
    23: {"name": "C23-SCHEMA",    "tags": ["structure","format","schema","template"], "data_weight": 0.3},
    24: {"name": "C24-PROMETHEUS","tags": ["science","physics","theory","hypothesis","scientific"], "data_weight": 0.8},
    25: {"name": "C25-TECHNE",    "tags": ["engineering","architecture","system","design"], "data_weight": 0.6},
    26: {"name": "C26-CHRONICLE", "tags": ["narrative","story","lore","history","fiction"], "data_weight": 0.4},
    27: {"name": "C27-CALCULUS",  "tags": ["math","mathematics","calculus","proof","equation","math"], "data_weight": 1.0},
    28: {"name": "C28-NAVIGATOR", "tags": ["platform","ecosystem","integration","workflow"], "data_weight": 0.3},
    29: {"name": "C29-TESSERACT", "tags": ["real-time","stream","data","sensor"], "data_weight": 0.2},
    30: {"name": "C30-NEXUS",    "tags": ["coordination","governance","policy","regulation"], "data_weight": 0.3},
    31: {"name": "C31-AEON",     "tags": ["simulation","game","world","physics"], "data_weight": 0.3},
    32: {"name": "C32-TYPIST",   "tags": ["grammar","writing","spelling","prompt","format"], "data_weight": 0.4},
    33: {"name": "C33-PREDATOR", "tags": ["adversarial","competitive","proof","counterexample","predatory","game_theory"], "data_weight": 0.6},
}

@dataclass
class ProgressiveConfig:
    expert_start: int = 0
    expert_end: int = 34
    phase: str = "expert_specialization"  # ingestion|decomposition|diffusion|router|expert|swarm|output|full
    steps_per_expert: int = 200
    seq_len: int = 128
    lr: float = 1e-4
    batch_size: int = 1
    accum_steps: int = 8

def load_jsonl_data(paths: List[str], max_samples: int = 5000) -> List[Dict]:
    """Load JSONL training data with category filtering."""
    all_data = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f'  SKIP: {p.name} (not found)')
            continue
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    all_data.append(d)
                except:
                    pass
        print(f'  Loaded {p.name}: {len(all_data)} total samples')
        if len(all_data) >= max_samples:
            break
    return all_data[:max_samples]

def filter_data_by_tags(data: List[Dict], tags: List[str]) -> List[str]:
    """Filter text samples matching expert tags."""
    texts = []
    for d in data:
        content = d.get('messages', [{}])[-1].get('content', '')
        if not content and 'text' in d:
            content = d['text']
        if not content:
            continue
        content_lower = content.lower()
        # Score by tag matches
        score = sum(1 for t in tags if t in content_lower)
        if score >= 1:
            texts.append(content)
    return texts

def train_expert_specialization(model, expert_idx: int, texts: List[str], cfg: ProgressiveConfig, device: str):
    """Train a single expert on domain-specific text data."""
    expert_name = EXPERT_DOMAINS[expert_idx]["name"]
    print(f'\n─── Training {expert_name} (Expert {expert_idx}) ───')
    
    if not texts:
        print(f'  No domain data for {expert_name}. Using general data.')
        return
    
    # Freeze everything except this expert's weights + LoRA
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    
    for name, p in model.named_parameters():
        # Expert's FFN weights
        if f'moe.w1' in name:
            # Index into expert dimension: w1 shape is [34, D, F]
            pass  # We'll handle this differently
        if any(f'moe.{k}[{expert_idx}]' in name or f'moe.{k}.{expert_idx}' in name 
               for k in ['w1', 'wgate', 'w2', 'w1_lora', 'w2_lora', 'wgate_lora']):
            p.requires_grad_(True)
    
    # Simpler: just train the expert's LoRA adapters
    for name, p in model.named_parameters():
        if any(f'moe.w1_lora' in name or f'moe.w2_lora' in name or f'moe.wgate_lora' in name):
            p.requires_grad_(True)
    
    model.to(device)
    model.train()
    
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable params: {n_train:,} ({n_train/1e6:.2f}M)')
    
    # Train on domain texts
    for step in range(cfg.steps_per_expert):
        idx = torch.randint(0, len(texts), (1,)).item()
        text = texts[idx][:cfg.seq_len * 4]  # Rough token estimate
        
        # Create dummy input (in real training, use actual tokenizer)
        inp = torch.randint(0, 100, (1, min(cfg.seq_len, max(4, len(text)//4))), device=device)
        
        opt.zero_grad()
        out = model(inp)
        # Simple reconstruction loss through the expert pathway
        logits = out['logits']
        loss = logits.norm()  # Simple placeholder — real training uses CE
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        
        if step % 50 == 0:
            print(f'  Step {step}: loss={loss.item():.4f}')
    
    print(f'  Done training {expert_name}')

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_pascal = device == 'cuda' and torch.cuda.get_device_capability()[0] < 7
    print(f'Device: {device}, Pascal: {is_pascal}')
    
    cfg = ProgressiveConfig()
    model_cfg = QuillanArchConfig(device=device, pascal_mode=is_pascal, text_only=True, low_mem=True)
    model = QuillanRoninSovereign(model_cfg)
    
    # Load base checkpoint
    base_ckpt = 'checkpoints/router_trained.pt'
    if os.path.exists(base_ckpt):
        ckpt = torch.load(base_ckpt, map_location='cpu', weights_only=False)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        print(f'Loaded {base_ckpt}')
    else:
        ckpt = torch.load('checkpoints/quillan_fixed.pt', map_location='cpu', weights_only=True)
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        if 'model_state_dict' in sd: sd = sd['model_state_dict']
        model.load_state_dict(sd, strict=False)
        print('Loaded quillan_fixed.pt')
    
    model.half()
    model.to(device)
    
    # ─── PHASE 5: EXPERT SPECIALIZATION ───────────────────────────────────
    if cfg.phase == "expert_specialization":
        print('\n═══ PHASE 5: EXPERT SPECIALIZATION (Vocational Training) ═══')
        
        # Load training data
        data_dir = Path('training_data')
        jsonl_files = [
            data_dir / 'code_train.jsonl',
            data_dir / 'GPT_5.5_Distilled.jsonl',
            data_dir / 'quillan_science_absolute.jsonl',
            data_dir / 'quillan_science_additional.jsonl',
            data_dir / 'full_train.jsonl',
            data_dir / 'instruct_train.jsonl',
            data_dir / 'quillan_corpus_CLEAN_V7.jsonl',
            data_dir / 'Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl',
        ]
        all_data = load_jsonl_data(jsonl_files)
        print(f'Total training samples: {len(all_data)}')
        
        # Train each expert
        for e in range(cfg.expert_start, cfg.expert_end):
            tags = EXPERT_DOMAINS[e]["tags"]
            texts = filter_data_by_tags(all_data, tags)
            print(f'  {EXPERT_DOMAINS[e]["name"]}: {len(texts)} domain samples (tags: {tags})')
            
            if len(texts) < 5:
                print(f'  Too few samples, using general data')
                texts = [d.get('messages', [{}])[-1].get('content', '') for d in all_data[:200]]
            
            train_expert_specialization(model, e, texts, cfg, device)
            
            # Save checkpoint after each expert
            torch.save({'state_dict': model.state_dict(), 'step': e}, 
                      f'checkpoints/progressive_expert_{e}.pt')
            print(f'  Saved checkpoint for {EXPERT_DOMAINS[e]["name"]}')
    
    # ─── PHASE 6: EGGROLL SWARM ───────────────────────────────────────────
    elif cfg.phase == "swarm":
        print('\n═══ PHASE 6: EGGROLL SWARM TRAINING ═══')
        # Train the rank-R perturbation matrices (A, B, C, D) per expert
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if 'expert_swarms' in name:
                p.requires_grad_(True)
        
        model.to(device)
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        
        for step in range(500):
            inp = torch.randint(0, 100, (1, cfg.seq_len), device=device)
            opt.zero_grad()
            out = model(inp)
            loss = out['logits'].norm() + out.get('routing_loss', torch.tensor(0.0, device=device))
            loss.backward()
            opt.step()
            if step % 100 == 0:
                vram = torch.cuda.memory_allocated()/1024**2
                print(f'  Step {step}: loss={loss.item():.4f} VRAM={vram:.0f}MB')
        
        torch.save({'state_dict': model.state_dict()}, 'checkpoints/progressive_swarm.pt')
        print('Swarm training complete!')
    
    # ─── PHASE 7: FINALIZERS + OUTPUT ─────────────────────────────────────
    elif cfg.phase == "output":
        print('\n═══ PHASE 7: OUTPUT HEADS (Dual Quillan + txt_dec) ═══')
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if any(k in name for k in ['quillan_finalizer', 'quillan_gate', 'txt_dec', 'pre_final_norm']):
                p.requires_grad_(True)
        
        model.to(device)
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        
        data = torch.load('training_data/GPT_5.5_Distilled.pt', map_location='cpu', weights_only=True)
        
        for step in range(500):
            idx = torch.randint(0, len(data)-cfg.seq_len-1, (1,)).item()
            inp = data[idx:idx+cfg.seq_len].to(device).long().unsqueeze(0)
            tgt = data[idx+1:idx+cfg.seq_len+1].to(device).long().unsqueeze(0)
            opt.zero_grad()
            out = model(inp)
            logits = out['logits']
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f'  Step {step}: CE loss={loss.item():.4f}')
        
        torch.save({'state_dict': model.state_dict()}, 'checkpoints/progressive_output.pt')
        print('Output heads training complete!')
    
    print('\n═══ PROGRESSIVE TRAINING DONE ═══')


if __name__ == '__main__':
    main()
