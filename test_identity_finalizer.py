import torch, inspect, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from tokenizers import Tokenizer
sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok = Tokenizer.from_file(r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json')
ckpt = torch.load(r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt', map_location='cpu')
valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
cfg_dict = {k: v for k, v in ckpt['config'].items() if k in valid_keys}
cfg = QuillanOniConfig(**cfg_dict)
model = QuillanRoninOni(cfg).cpu().eval()
model.load_state_dict(ckpt['model_state_dict'], strict=False)

# Convert all parameters to bfloat16
model = model.to(torch.bfloat16)

# Test with identity finalizers
H = cfg.hidden_dim
model.quillan_finalizer_q1.weight.data = torch.eye(H, dtype=torch.bfloat16) / 1.1
model.quillan_finalizer_q2.weight.data = torch.eye(H, dtype=torch.bfloat16) / 1.1

prompts = [
    "Quillan is a sovereign cognitive agent. Its mission is to",
    "The quick brown fox jumps over the lazy",
    "Artificial intelligence and machine"
]

for p in prompts:
    enc = tok.encode(p).ids
    x = torch.tensor([enc], dtype=torch.long)
    with torch.no_grad():
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        last_logits = logits[0, -1].float()
        probs = torch.softmax(last_logits, dim=-1)
        top5 = torch.topk(probs, 5)
        print(f"\nPrompt: '{p}'")
        for prob, idx in zip(top5.values, top5.indices):
            tok_text = tok.decode([idx.item()])
            print(f"   {prob.item()*100:5.2f}% | ID {idx.item():5d} | '{tok_text}'")
