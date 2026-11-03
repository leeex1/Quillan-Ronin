import os, sys, io, gc, inspect, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
tok = Tokenizer.from_file(tok_path)

prompts = [
    "Quillan is a sovereign cognitive agent. Its mission is to",
    "The quick brown fox jumps over the lazy",
    "Artificial intelligence and machine",
    "In the bushido code, honor and rectitude are"
]

def generate_text(model, prompt: str, max_new_tokens: int = 12, temperature: float = 0.7, top_k: int = 40):
    enc = tok.encode(prompt).ids
    generated = list(enc)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_x = torch.tensor([generated[-128:]], dtype=torch.long)
            out = model(curr_x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            last_logits = logits[0, -1].float()
            
            if temperature > 0:
                last_logits = last_logits / temperature
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[..., [-1]]] = -float('Inf')
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(last_logits, dim=-1).item()
                
            generated.append(next_token)
            if next_token == 0 or next_token == tok.token_to_id('<|endoftext|>'):
                break
                
    return tok.decode(generated)

def test_checkpoint(tag: str, ckpt_path: str):
    print("=" * 80, flush=True)
    print(f"VERIFYING CHECKPOINT: {tag}", flush=True)
    print(f"Path: {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.1f} MB)", flush=True)
    print("=" * 80, flush=True)
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = QuillanOniConfig(**cfg_dict)
    
    model = QuillanRoninOni(cfg).cpu().eval()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    del ckpt
    gc.collect()
    
    for p in prompts:
        enc = tok.encode(p).ids
        x = torch.tensor([enc], dtype=torch.long)
        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            last_logits = logits[0, -1].float()
            probs = F.softmax(last_logits, dim=-1)
            top5 = torch.topk(probs, 5)
            print(f"\nPrompt: '{p}'", flush=True)
            print("  Top 5 Next Tokens:", flush=True)
            for prob, idx in zip(top5.values, top5.indices):
                tok_str = tok.decode([idx.item()])
                print(f"    {prob.item()*100:5.2f}% | ID {idx.item():5d} | '{tok_str}'", flush=True)
                
        gen_sample = generate_text(model, p, max_new_tokens=10, temperature=0.7)
        print(f"  Generated Sample: \"{gen_sample}\"", flush=True)
        
    del model
    gc.collect()
    print("=" * 80, flush=True)

if __name__ == "__main__":
    out_dir = r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs"
    p_mini = os.path.join(out_dir, "quillan_mini_0.6b_tripartite.pt")
    
    if os.path.exists(p_mini):
        test_checkpoint("Quillan Mini 0.6B Tripartite", p_mini)
