"""GPU fit+speed test: Oni fp16 on cuda:0, greedy 30 tokens, vs CPU 1.44 tok/s."""
import sys
import time
import torch
from pathlib import Path
REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO / "scripts" / "hf_code_era"), str(REPO / "scripts"),
          str(REPO / "03 - Training & Model"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
print(f"vram free: {torch.cuda.mem_get_info(0)[0] / 1e9:.2f}GB", flush=True)
tok = QuillanBPETokenizer()
cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=1024, ffn_dim=2048,
                       n_layer=6, num_experts=34, top_k=4, max_seq_len=512)
try:
    m = QuillanRoninOni(cfg)
    bd = torch.load(REPO / "checkpoints" / "hf_restore" / "mini_sft_best.pt",
                    map_location="cpu", weights_only=True)
    missing, unexp = m.load_state_dict(bd.get("model_state_dict", bd), strict=False)
    print(f"bound missing={len(missing)} unexp={len(unexp)}", flush=True)
    m = m.half().cuda().eval()
    print(f"vram after load: {torch.cuda.mem_get_info(0)[0] / 1e9:.2f}GB free", flush=True)
    ids = tok.encode("Say hello in one short sentence.")
    gen = list(ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            pt = torch.tensor([gen[-256:]], dtype=torch.long, device="cuda")
            out = m(pt)
            logits = out[0] if isinstance(out, tuple) else out
            nxt = int(torch.argmax(logits[0, -1, :]).item())
            gen.append(nxt)
            if nxt in (0, 50256, 50261):
                break
    dt = time.perf_counter() - t0
    n = len(gen) - len(ids)
    print(f"GPU fp16: {n} tokens in {dt:.1f}s = {n / dt:.2f} tok/s", flush=True)
    print("TEXT:", repr(tok.decode([t for t in gen[len(ids):] if t < 50257]))[:300], flush=True)
except RuntimeError as e:
    print(f"GPU TEST FAILED: {str(e)[:300]}", flush=True)
