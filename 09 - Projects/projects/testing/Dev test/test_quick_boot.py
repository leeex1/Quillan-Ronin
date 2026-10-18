import sys, time, torch, tiktoken
from pathlib import Path

SCRATCH = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\47d0b748-384a-4cc6-bcb6-f13c52f1d9d9\scratch")
sys.path.insert(0, str(SCRATCH))

from quillan_v10_unrolled_sovereign import QuillanUnrolledSovereign, QuillanUnrolledConfig

t0 = time.time()
cfg = QuillanUnrolledConfig()
model = QuillanUnrolledSovereign(cfg).to('cpu')
ckpt = torch.load(r"C:\02_QUILLAN\checkpoints\checkpoints_sft\quillan_thinking_reasoning_master.pt", map_location='cpu', weights_only=False)
model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
model.eval()
print(f"Booted in {time.time() - t0:.2f}s!")

enc = tiktoken.get_encoding("gpt2")
prompt = "<|user|>\nWhat is photosynthesis?\n<|assistant|>\n"
tokens = enc.encode(prompt)
t_gen = time.time()
out = model.generate(tokens, max_tokens=30, temp=0.7)
text = enc.decode(out[len(tokens):])
print(f"Response ({time.time() - t_gen:.2f}s):\n{text.strip()}")
