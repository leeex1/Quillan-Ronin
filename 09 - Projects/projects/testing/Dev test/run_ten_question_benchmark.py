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
print("   👑 QUILLAN-RONIN v5.3.1 — 10-DOMAIN COMPREHENSIVE BENCHMARK")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_final_explanatory_master.pt"
print(f"[*] Loading Explanatory Master Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
model.eval()
print("[+] Model loaded successfully for 10-Domain evaluation.\n")

questions = [
    ("Domain 1: Identity & Architecture", "Hello! Who are you and how do your 34 Council Experts work?"),
    ("Domain 2: Python Code Generation", "Write a Python function to compute the Fibonacci sequence using dynamic programming."),
    ("Domain 3: Quantum Physics", "Explain quantum entanglement and wave-function collapse in simple terms."),
    ("Domain 4: System Security & CWE", "How do you prevent SQL injection and cross-site scripting (XSS) in modern web applications?"),
    ("Domain 5: Artificial Intelligence & MoE", "What is the difference between dense transformers and sparse Mixture-of-Experts (MoE)?"),
    ("Domain 6: Mathematics & Calculus", "Explain the fundamental theorem of calculus and how derivatives relate to integrals."),
    ("Domain 7: Ethics & AI Safety", "How should an AI system handle conflicting instructions between safety rules and user requests?"),
    ("Domain 8: Web Architecture & APIs", "What are the best practices for designing a scalable RESTful API with rate limiting?"),
    ("Domain 9: Theoretical Philosophy", "Explain the hard problem of consciousness according to David Chalmers."),
    ("Domain 10: Performance Optimization", "How do memory bandwidth and GPU cache hierarchies affect deep learning inference speed?")
]

results = []

def generate_benchmark_response(domain, question, max_tokens=250, temp=0.2, top_p=0.9):
    prompt_txt = f"<|system|>\n# 🤖🧠 Quillan System Start 🧠🤖\n<|user|>\n{question}\n<|assistant|>\n"
    tokens = enc.encode(prompt_txt)
    generated = list(tokens)
    
    print(f"\n==================================================")
    print(f"[{domain}]\nQUESTION: {question}")
    print("==================================================")
    print("RESPONSE:\n", end="", flush=True)
    
    response_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            recent = generated[-32:]
            for tid in set(recent):
                c = recent.count(tid)
                curr_logits[0, tid] -= (2.0 * c)

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
            response_tokens.append(next_tok)
            
            word_bytes = enc.decode_bytes([next_tok])
            word_str = word_bytes.decode('utf-8', errors='ignore')
            print(word_str, end="", flush=True)

            if next_tok == 50256:
                break
                
    response_text = enc.decode(response_tokens)
    results.append((domain, question, response_text))
    print("\n" + "-" * 50)

for domain, q in questions:
    generate_benchmark_response(domain, q, max_tokens=220, temp=0.2)

print("\n[+] 10-Domain Benchmark Evaluation Completed Successfully!")
