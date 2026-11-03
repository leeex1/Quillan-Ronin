import os
import sys
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

tok_path = r'C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\tokenizer.json'
print("[1/3] Loading Llama 3.2 128k Tokenizer...", flush=True)
tok = Tokenizer.from_file(tok_path)

prompts = [
    "Write a short poetic metaphor describing a quantum particle in a superposition.",
    "Explain the core invariant of the Paxos consensus algorithm in one concise sentence."
]

def test_checkpoint(model_name: str, ckpt_path: str, n_layers: int):
    print("\n" + "=" * 80, flush=True)
    print(f"VERIFYING: {model_name} (Layers={n_layers}, D=2048)", flush=True)
    print(f"Path: {ckpt_path} ({os.path.getsize(ckpt_path)/(1024*1024):.1f} MB)", flush=True)
    print("=" * 80, flush=True)
    
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.pop('model_state_dict')
    del ckpt
    import gc
    gc.collect()
    print(f">>> Checkpoint loaded in {time.time()-t0:.1f}s", flush=True)
    
    cfg = QuillanOniConfig(
        vocab_size=128256,
        hidden_dim=2048,
        n_layer=n_layers,
        n_head=32,
        head_dim=64,
        ffn_dim=8192,
        num_experts=34,
        expert_rank=16,
        use_evo_moe=False
    )
    cfg.vectorized_experts = True
    
    t0 = time.time()
    orig_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = QuillanRoninOni(cfg)
    torch.set_default_dtype(orig_dtype)
    print(f">>> Model instantiated in {time.time()-t0:.1f}s", flush=True)
    
    t0 = time.time()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f">>> State dict loaded in {time.time()-t0:.1f}s (Missing={len(missing)}, Unexpected={len(unexpected)})", flush=True)
    del state_dict
    gc.collect()
    model.eval()
    
    for p_idx, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {p_idx}]: \"{prompt}\"", flush=True)
        print(">>> Model Output: ", end="", flush=True)
        
        enc = tok.encode(f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        cur_input = torch.tensor([enc.ids], dtype=torch.long)
        
        words = []
        with torch.no_grad():
            for step in range(25):
                logits = model(cur_input, compute_aux=False, return_aux=False)
                if isinstance(logits, tuple): logits = logits[0]
                next_logits = logits[0, -1, :].clone()
                
                # Temperature 0.7 + top-k 40
                top_k = 40
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = -float('Inf')
                probs = F.softmax(next_logits / 0.7, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                
                if nxt in [128001, 128009]:
                    break
                token_str = tok.decode([nxt])
                print(token_str, end="", flush=True)
                words.append(token_str)
                cur_input = torch.cat([cur_input, torch.tensor([[nxt]], dtype=torch.long)], dim=1)
        print("\n", flush=True)
        
    del model
    gc.collect()

# First test Mini 0.8B (8 layers = fast, ~4.7GB)
test_checkpoint(
    "Quillan-Ronin Mini 0.8B (25% Quad-Donor: 2L/2F/2B/2Q)",
    r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_mini_0.6b_sovereign_wsp128k_qwen35.pt",
    8
)

# Then test Flagship 1B (16 layers, 25% Quad-Donor: 4L/4F/4B/4Q)
test_checkpoint(
    "Quillan-Ronin Flagship 1B (25% Quad-Donor: 4L/4F/4B/4Q)",
    r"C:\02_QUILLAN\checkpoints\checkpoints_oni\sft_runs\quillan_1b_sovereign_wsp128k_qwen35.pt",
    16
)

print("\n" + "=" * 80, flush=True)
print("DUAL MODEL VERIFICATION COMPLETED", flush=True)
print("=" * 80, flush=True)
