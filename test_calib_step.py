import os, sys, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
sys.path.insert(0, r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

def test_forward_backward():
    tok_path = r'C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\oni\tokenizer.json'
    tok = Tokenizer.from_file(tok_path)
    
    ckpt_path = r'C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_tripartite.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    import inspect
    valid_keys = set(inspect.signature(QuillanOniConfig.__init__).parameters.keys())
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    cfg = QuillanOniConfig(**cfg_dict)
    
    model = QuillanRoninOni(cfg).cpu()
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    print("Model loaded successfully. Trainable parameter count:")
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Total: {total_p/1e6:.2f}M")
    
    # Filter trainable parameters: LayerNorms, bridges, finalizers, lm_head, router
    trainable_names = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["ln_", "bridge", "finalizer", "comm_gate", "ingest_gate", "router", "pull_gate", "w_gate", "lm_head"]):
            param.requires_grad = True
            trainable_names.append(name)
        else:
            param.requires_grad = False
            
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Calibratable Alignment Parameters: {trainable_p/1e6:.2f}M ({len(trainable_names)} tensors)")
    
    # Test forward and backward on a small sequence
    x = torch.randint(0, 1000, (1, 32))
    targets = torch.randint(0, 1000, (1, 32))
    
    t0 = time.time()
    out = model(x, deliberation=False)
    logits = out[0] if isinstance(out, tuple) else out
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    t1 = time.time()
    
    print(f"Forward+Backward Step Time: {t1 - t0:.3f}s | Initial Loss: {loss.item():.4f}")
    
if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory / (1024**3), "GB")
    test_forward_backward()

