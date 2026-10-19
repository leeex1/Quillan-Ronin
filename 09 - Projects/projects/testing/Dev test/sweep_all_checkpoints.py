import os, sys, torch, torch.nn.functional as F, tiktoken, gc
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

# Test ALL viable checkpoints to find one that actually generates text
ckpt_dir = REPO_ROOT / "checkpoints" / "checkpoints_sft"
targets = [
    "quillan_sft_v3_best.pt",
    "quillan_dialogue_best.pt",
    "quillan_hyper_tuned_v531.pt",
    "quillan_final_best.pt",
    "quillan_final_explanatory_master.pt",
    "quillan_1000000x_master_aligned.pt",
    "quillan_fluent_aligned.pt",
]

prompt = "<|system|>\n# Quillan System Start\n<|user|>\nHello! Who are you?\n<|assistant|>\n"
prompt_toks = enc.encode(prompt)

for name in targets:
    path = ckpt_dir / name
    if not path.exists():
        continue
    
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    step = ckpt.get('step', 'N/A')
    loss = ckpt.get('loss', 'N/A')
    
    # Load with strict=False to handle missing keys
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    # Generate 30 tokens quickly
    generated = list(prompt_toks)
    with torch.no_grad():
        for _ in range(30):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr = logits[:, -1, :].clone()
            
            # Block newline tokens and previous token
            curr[0, generated[-1]] -= 50.0
            # Block pure whitespace tokens (newlines, spaces)
            for ws_tok in [198, 220, 628, 50256]:  # \n, space, \n\n, <|endoftext|>
                curr[0, ws_tok] -= 30.0
            
            next_tok = torch.argmax(curr, dim=-1).item()
            generated.append(next_tok)
    
    output = enc.decode(generated[len(prompt_toks):])
    
    # Check top-5 logits for first token to understand distribution
    inp = torch.tensor([prompt_toks[-128:]], dtype=torch.long)
    with torch.no_grad():
        logits = model(inp)
        if isinstance(logits, tuple):
            logits = logits[0]
        first_logits = logits[:, -1, :].squeeze()
        top5_vals, top5_ids = torch.topk(first_logits, 5)
        top5_decoded = [enc.decode([tid.item()]) for tid in top5_ids]
    
    print(f"\n{'='*60}")
    print(f"  {name} (Step={step}, Loss={loss})")
    print(f"  Top-5 first tokens: {list(zip(top5_decoded, top5_vals.tolist()))}")
    print(f"  Output (30 tok): {output[:200]}")

print(f"\n{'='*60}")
print("[+] Multi-checkpoint sweep complete.")
