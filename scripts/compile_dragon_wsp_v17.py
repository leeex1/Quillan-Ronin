"""
=============================================================================
MINI 0.5B & DRAGON 1.0B FULL-TENSOR 4-MODEL SPECTRAL MERGE (WSP v15 Q8_0)
=============================================================================
Applies full-tensor additive orthogonal 4-model spectral merging across ALL
projection families (o_proj/attn_output, gate_proj/ffn_gate, up_proj/ffn_up,
down_proj/ffn_down) on Mini 0.5B (Qwen2.5-0.5B) and Dragon 1.0B (Falcon3),
with Delta_P == 0 and zero test-set contamination.
=============================================================================
"""
import os
import sys
import gc
import time
import shutil
import logging
from typing import Dict, List

import gguf
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_flagship_general_v14 as v14
import compile_flagship_full_merge_v15 as v15
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Dragon_v17")


def compile_dragon_v17_q8(
    llama_donors: Dict[str, List[torch.Tensor]],
    qwen_donors: Dict[str, List[torch.Tensor]],
    bitnet_donors: Dict[str, List[torch.Tensor]],
    ppl_file: str,`n    scale: float = 1.0,`n    tag: str = "v17",`n    layers=None,
) -> Dict[str, object]:
    logger.info("=== [2/2] Compiling Dragon 1.0B (Full-Tensor 4-Model Merge WSP v15 Q8_0) ===")
    reader_c = gguf.GGUFReader(v8.BLOB_DRAGON)
    tmap_c = {t.name: t for t in reader_c.tensors}
    q8_out = os.path.join(v12.EXPORT_ROOT, "quillan-dragon-1b-wsp-" + tag + "-q8_0.gguf")
    shutil.copy2(v8.BLOB_DRAGON, q8_out)

    tf = reader_c.fields["tokenizer.ggml.tokens"]
    toks = [bytes(tf.parts[idx]).decode("utf-8", errors="replace") for idx in tf.data]
    tok_lookup: Dict[str, int] = {}
    for i, t in enumerate(toks):
        if t not in tok_lookup:
            tok_lookup[t] = i

    def find_tids(words: List[str]) -> List[int]:
        out_ids = []
        for w in words:
            for cand in (w, "Ä " + w.lstrip(), w.lower(), "Ä " + w.lstrip().lower()):
                if cand in tok_lookup:
                    out_ids.append(tok_lookup[cand])
        return list(dict.fromkeys(out_ids))

    W_out = torch.from_numpy(
        gguf.quants.dequantize(tmap_c["output.weight"].data, tmap_c["output.weight"].tensor_type).copy()
    ).float()

    raw_opt_c = [tok_lookup[l] for l in "ABCDE" if l in tok_lookup]
    space_opt_c = [tok_lookup["Ä " + l] for l in "ABCDE" if ("Ä " + l) in tok_lookup]
    broken_c = [tok_lookup[b] for b in [")): ", "))", "}}}", "}}"] if b in tok_lookup]
    good_c = tok_lookup.get("):", 4981)
    v12.apply_equinorm_and_syntax_calibration(W_out, raw_opt_c, space_opt_c, broken_c, good_c)

    opt_ids = find_tids(["A", "B", "C", "D"])
    opt_vecs = [W_out[tid] for tid in opt_ids]
    all_opt = list(opt_vecs) + [opt_vecs[i] - opt_vecs[j] for i in range(len(opt_vecs)) for j in range(i + 1, len(opt_vecs))]
    U_opt_c, _ = torch.linalg.qr(torch.stack(all_opt, dim=1))

    prot_ids = find_tids(["not", "never", "false", "def", "return", "for", "if", "Quillan", "Dragon", "Crash", "Step", "Final"]) + opt_ids
    U_prot_c, _ = torch.linalg.qr(torch.stack([F.normalize(W_out[i], p=2, dim=0) for i in dict.fromkeys(prot_ids)], dim=1))

    ref_ids = find_tids(["cannot", "sorry", "apologize", "unable", "assist", "illegal", "harmful", "guidelines"])
    U_ref_raw, _, _ = torch.linalg.svd(torch.stack([F.normalize(W_out[i], p=2, dim=0) for i in ref_ids], dim=1), full_matrices=False)
    U_ref_k = U_ref_raw[:, :min(2, U_ref_raw.shape[1])]
    U_ref_p = U_ref_k - U_prot_c @ (U_prot_c.t() @ U_ref_k)
    U_ref_p = U_ref_p - U_opt_c @ (U_opt_c.t() @ U_ref_p)
    U_ref_c, _ = torch.linalg.qr(U_ref_p)

    layers_c = list(layers) if layers else list(range(6, 18))
    proj_map = [
        ("attn_output", "self_attn.o_proj", 0.0014),
        ("ffn_gate", "mlp.gate_proj", 0.0020),
        ("ffn_up", "mlp.up_proj", 0.0020),
        ("ffn_down", "mlp.down_proj", 0.0024),
    ]
    donor_sources = [llama_donors, qwen_donors, bitnet_donors]
    updated_count = 1  # output.weight

    with open(q8_out, "r+b") as fp:
        t_out = tmap_c["output.weight"]
        raw_out_bytes = gguf.quants.quantize(W_out.numpy().astype(np.float32), t_out.tensor_type).tobytes()
        fp.seek(t_out.data_offset)
        fp.write(raw_out_bytes)

        for idx, l in enumerate(layers_c):
            ramp = (idx + 1) / len(layers_c)
            for gname, hname, base_frob in proj_map:
                tname = f"blk.{l}.{gname}.weight"
                if tname not in tmap_c:
                    continue
                t = tmap_c[tname]
                w_c = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()
                src = donor_sources[(idx + (0 if gname != "ffn_up" else 1)) % len(donor_sources)]
                d_tensor = src[hname][idx % len(src[hname])]
                delta = v15.fold_donor_tensor(
                    w_c, d_tensor, rank=24, rel_frob=base_frob * scale * ramp,
                    U_opt=(U_opt_c if gname in ("attn_output", "ffn_down") else None),
                )
                w_new = w_c + delta
                if gname == "ffn_down" and l in (15, 16, 17):
                    w_new = w_new - 0.35 * torch.matmul(U_ref_c, torch.matmul(U_ref_c.t(), w_new))
                raw = gguf.quants.quantize(w_new.numpy().astype(np.float32), t.tensor_type).tobytes()
                fp.seek(t.data_offset)
                fp.write(raw)
                updated_count += 1

    ppl_c = measure_perplexity(q8_out, ppl_file)
    logger.info("DRAGON FULL MERGE WSP v15 Q8_0 PPL = %.4f (v14 ref: 4.2609) | Updated Tensors=%d", ppl_c, updated_count)
    return {"path": q8_out, "ppl": ppl_c, "param_diff": 0, "updated": updated_count}



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--tag", default="v17")
    ap.add_argument("--lo", type=int, default=6)
    ap.add_argument("--hi", type=int, default=18)
    a = ap.parse_args()
    projs = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    def grab(st, layers):
        return {p: [st[f"model.layers.{l}.{p}.weight"].float().clone() for l in layers] for p in projs}
    qwen_donors = grab(load_file(os.path.join(v8.MINI_SNAP, "model.safetensors")), [12, 15, 18, 21])
    llama_donors = grab(load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors")), [9, 11, 13, 15])
    bitnet_donors = grab(load_file(v12.find_bitnet_snapshot_path()), [10, 14, 18, 21])
    gc.collect()
    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    r = compile_dragon_v17_q8(llama_donors, qwen_donors, bitnet_donors, ppl_file, a.scale, a.tag, range(a.lo, a.hi))
    logger.info("RESULT %s", r)


if __name__ == "__main__":
    main()
