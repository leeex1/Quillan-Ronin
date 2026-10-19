"""
QUILLAN MASTER GOLD DATASET COMPILER (v2.1 - Council Personas + Knowledge Shards)
==================================================================================
Compiles a robust [N, 256] token tensor for high-density training:
  1. Ingests all 34 Council Agent personas from .github/agents/*.agent.md
  2. Ingests research papers & coding excellence from tokenizer_sample.jsonl
  3. Packages into [15631, 256] shape with prompt masking (-100) on prompts.
"""
import os
import sys
import json
import time
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(r"C:\02_QUILLAN")
for p in [str(REPO), str(REPO / "scripts"), str(REPO / "03 - Training & Model")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from quillan_bpe_tokenizer import QuillanBPETokenizer

def main():
    print("=" * 72, flush=True)
    print("  👑 COMPILING MASTER GOLD TRAINING DATASET (34 COUNCIL + PAPERS)", flush=True)
    print("=" * 72, flush=True)

    tok = QuillanBPETokenizer()
    print(f"Tokenizer loaded (vocab: {tok.vocab_size})", flush=True)

    out_dir = REPO / "training_data" / "canonical_standardized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "quillan_master_gold_training_v1.pt"

    target_samples = 15631
    seq_len = 256
    stride = 128

    all_pairs = []

    # 1. Ingest all 34 Council Agent Personas from .github/agents
    agents_dir = REPO / ".github" / "agents"
    if agents_dir.exists():
        for f in sorted(agents_dir.glob("*.agent.md")):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                agent_name = f.stem.replace(".agent", "").upper()
                prompt = f"User: Consult Council Member {agent_name}.\n\nAssistant:"
                response = f" [{agent_name} Deliberation]\n{content.strip()}"
                all_pairs.append((prompt, response))
            except Exception:
                pass
        print(f"Ingested {len(all_pairs)} Council Agent personas from .github/agents", flush=True)

    # 2. Ingest coding excellence & papers from tokenizer_sample.jsonl
    sample_src = REPO / "09 - Projects" / "projects" / "05_Training" / "scripts" / "tokenizer_sample.jsonl"
    print(f"Ingesting domain records from {sample_src.name}...", flush=True)

    input_ids_list = []
    labels_list = []

    # First, pack council agent pairs with prompt masking
    for prompt, response in all_pairs:
        p_ids = tok.encode(prompt)
        r_ids = tok.encode(response)
        combined = p_ids + r_ids
        if len(combined) > seq_len:
            combined = combined[:seq_len]
            p_len = min(len(p_ids), seq_len - 10)
        else:
            p_len = len(p_ids)

        # Pad to seq_len
        pad_len = seq_len - len(combined)
        padded = combined + [50256] * pad_len
        lbls = [-100] * p_len + combined[p_len:] + [-100] * pad_len

        input_ids_list.append(torch.tensor(padded, dtype=torch.long))
        labels_list.append(torch.tensor(lbls, dtype=torch.long))

    # Next, slice tokenizer_sample.jsonl into sliding window chunks
    print(f"Sliding window slicing of core technical corpus...", flush=True)
    with open(sample_src, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(input_ids_list) >= target_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                txt = obj.get("text", "")
                if len(txt) < 40:
                    continue
                tokens = tok.encode(txt)
                for i in range(0, len(tokens) - seq_len + 1, stride):
                    if len(input_ids_list) >= target_samples:
                        break
                    chunk = tokens[i : i + seq_len]
                    if len(chunk) == seq_len:
                        t_chunk = torch.tensor(chunk, dtype=torch.long)
                        input_ids_list.append(t_chunk)
                        labels_list.append(t_chunk.clone())
            except Exception:
                continue

    print(f"Total samples assembled: {len(input_ids_list)}", flush=True)

    input_ids_tensor = torch.stack(input_ids_list)
    labels_tensor = torch.stack(labels_list)

    print(f"Tensor Shape: {input_ids_tensor.shape}", flush=True)
    payload = {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "meta": {
            "source": "Council_34_and_Core_Papers",
            "samples": len(input_ids_list),
            "seq_len": seq_len,
            "vocab_size": tok.vocab_size
        }
    }

    torch.save(payload, out_file)
    sz_mb = out_file.stat().st_size / (1024**2)
    print(f"🌟 Saved master dataset: {out_file} ({sz_mb:.1f} MB)", flush=True)
    print("=" * 72, flush=True)

if __name__ == "__main__":
    main()
