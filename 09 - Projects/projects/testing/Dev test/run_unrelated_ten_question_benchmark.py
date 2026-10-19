import os
import sys
import time
import torch
import torch.nn.functional as F
import tiktoken
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — 10 UNRELATED DOMAINS BENCHMARK")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_hyper_tuned_v531.pt"
print(f"[*] Loading Hyper-Tuned Master Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
model.eval()
print("[+] Model loaded successfully for 10 Unrelated Domains evaluation.\n")

questions = [
    ("Domain 1: Astronomy & Physics", "Why is the sky blue and why do sunsets appear red and orange?"),
    ("Domain 2: Culinary Chemistry", "Explain the Maillard reaction in cooking and why it creates rich flavor in seared meat."),
    ("Domain 3: World History", "What were the main causes of the fall of the Western Roman Empire in 476 AD?"),
    ("Domain 4: Medicine & Immunology", "How do vaccines work with the human immune system to build antibodies?"),
    ("Domain 5: Economics & Finance", "What is inflation, and how do central banks use interest rates to control it?"),
    ("Domain 6: Theoretical Physics", "Explain Albert Einstein's general theory of relativity and time dilation."),
    ("Domain 7: Marine Ecology", "How do coral reefs support marine ecosystems and why are they vulnerable to ocean acidification?"),
    ("Domain 8: Computer Networking", "Explain how the HTTPS protocol secures data transmission using TLS encryption."),
    ("Domain 9: Jurisprudence & Law", "What is the difference between civil law and criminal law?"),
    ("Domain 10: Classical Philosophy", "Explain Plato's Allegory of the Cave and its meaning regarding human knowledge.")
]

def generate_domain_response(domain, question, max_tokens=220, temp=0.2, top_p=0.9):
    prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{question}\n<|assistant|>\n"
    tokens = enc.encode(prompt_txt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"[{domain}]\nQUESTION: {question}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Zero-Stutter Fix: Hard penalty on immediate previous token
            if len(generated) > 0:
                prev_tok = generated[-1]
                curr_logits[0, prev_tok] -= 50.0
                
            # Window repetition penalty
            recent_tokens = generated[-48:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                curr_logits[0, tid] -= (4.0 * count)

            if temp == 0.0:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
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

            generated.append(next_tok)
            
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
    print("\n" + "-" * 50)

for domain, q in questions:
    generate_domain_response(domain, q, max_tokens=180, temp=0.2)

print("\n[+] 10 Unrelated Domains Benchmark Completed Successfully!")
