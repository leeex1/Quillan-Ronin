import os
import sys
import time
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path
from transformers import GPT2LMHeadModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — 1,000,000X FLUENCY BACKBONE INGESTION")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

print("[*] Loading cached GPT-2 Medium English semantic embeddings...")
gpt2_base = GPT2LMHeadModel.from_pretrained("gpt2-medium")
base_wte = gpt2_base.transformer.wte.weight.detach() # shape (50257, 1024)
base_wpe = gpt2_base.transformer.wpe.weight.detach() # shape (1024, 1024)

print(f"[+] Loaded GPT-2 Medium Embedding Matrix: shape={base_wte.shape}, std={base_wte.std().item():.4f}")

# Inject pre-trained English semantic embeddings into Quillan
with torch.no_grad():
    model.ingestion.txt_emb.weight.copy_(base_wte)
    model.ingestion.pos_embed.data.squeeze(0).copy_(base_wpe)
    # Output vocabulary decoder layer weight tying
    model.txt_dec.weight.copy_(base_wte)

print("[+] Successfully initialized Quillan Ingestion & Head with GPT-2 English Semantic Manifold!")

# Save initialized backbone checkpoint
init_ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_gpt2_semantic_init.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'version': 'quillan-v5.3.1-gpt2-semantic-init'
}, init_ckpt_path)
print(f"[SAVE] Semantic Backbone Checkpoint saved to: {init_ckpt_path}")

# Run zero-shot English completion test
model.eval()

def generate_semantic_demo(prompt, max_tokens=60, temp=0.7, top_p=0.9):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    print(f"\nPROMPT: {prompt.strip()}")
    print("RESPONSE: ", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            for tid in set(generated[-32:]):
                curr_logits[0, tid] -= 3.0

            if generated[-1] == 220:
                curr_logits[0, 220] = float('-inf')

            if temp > 0:
                scaled_logits = curr_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scaled_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()
            else:
                next_tok = torch.argmax(curr_logits, dim=-1).item()

            generated.append(next_tok)
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
    print("\n" + "-" * 50)

print("\n=== INITIAL ENGLISH SEMANTIC EMBEDDING SPEECH TEST ===")
test_prompts = [
    "The capital of France is",
    "Artificial intelligence is defined as",
    "<|user|>\nHello! Who are you?\n<|assistant|>\n"
]

for p in test_prompts:
    generate_semantic_demo(p, max_tokens=50, temp=0.7, top_p=0.9)
