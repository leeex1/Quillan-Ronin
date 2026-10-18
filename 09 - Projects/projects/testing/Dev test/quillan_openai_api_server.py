import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import tiktoken
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))

from _dev.quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig

print("==================================================================")
print("   👑 QUILLAN-RONIN v5.3.1 — LOCAL OPENAI-COMPATIBLE API SERVER")
print("==================================================================")

enc = tiktoken.get_encoding("gpt2")

cfg = QuillanArchConfig(
    hidden_dim=1024, ffn_dim=2048, num_experts=34,
    vocab_size=50257, text_only=True, eggroll_rank=256
)
model = QuillanRoninSovereign(cfg).to("cpu")

ckpt_path = r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt"
print(f"[*] Loading Master Model Checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"[+] Model Ready! Best Loss: {ckpt.get('loss', '0.3054')}")

def generate_tokens(prompt, max_tokens=60, temp=0.65, top_p=0.9):
    tokens = enc.encode(prompt)
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated[-128:]], dtype=torch.long)
            logits = model(inp)
            if isinstance(logits, tuple):
                logits = logits[0]
            curr_logits = logits[:, -1, :].clone()
            
            # Hard zero-stutter penalty
            if len(generated) > 0:
                prev_tok = generated[-1]
                curr_logits[0, prev_tok] -= 50.0
                
            recent_tokens = generated[-48:]
            for tid in set(recent_tokens):
                count = recent_tokens.count(tid)
                curr_logits[0, tid] -= (4.0 * count)

            if temp == 0.0:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                scaled_logits = curr_logits / max(temp, 0.01)
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
            if next_tok == 50256:
                break
                
    response_tokens = generated[len(tokens):]
    return enc.decode(response_tokens)

class QuillanHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            res = {
                "object": "list",
                "data": [
                    {"id": "quillan-v5.3.1", "object": "model", "owned_by": "quillan-research"},
                    {"id": "quillan-ronin", "object": "model", "owned_by": "quillan-research"}
                ]
            }
            self.wfile.write(json.dumps(res).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "Quillan-Ronin API Online"}).encode())

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            payload = json.loads(body_bytes.decode('utf-8'))
            
            messages = payload.get("messages", [])
            max_tokens = payload.get("max_tokens", 300)
            temp = payload.get("temperature", 0.2)
            
            # Format prompt with system prefix
            formatted_prompt = "<|system|>\nYou are Quillan-Ronin, a sovereign, highly intelligent AI assistant and expert coder.\n"
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                formatted_prompt += f"<|{role}|>\n{content}\n"
            formatted_prompt += "<|assistant|>\n"
            
            answer_text = generate_tokens(formatted_prompt, max_tokens=max_tokens, temp=temp)
            
            response_payload = {
                "id": f"chatcmpl-quillan-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "quillan-v5.3.1",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": answer_text.strip()
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(enc.encode(formatted_prompt)),
                    "completion_tokens": len(enc.encode(answer_text)),
                    "total_tokens": len(enc.encode(formatted_prompt + answer_text))
                }
            }
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000):
    server_address = ('0.0.0.0', port)
    httpd = ThreadingHTTPServer(server_address, QuillanHTTPHandler)
    print(f"\n[SERVER] 🚀 Quillan OpenAI-Compatible API Server active on http://127.0.0.1:{port}/v1")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server(8000)
