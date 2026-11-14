"""
=============================================================================
DRAGON 1.0B SURGICAL PER-LAYER WSP COMPILER (v19)
=============================================================================
Starts from verified Dragon v18b (PPL 4.2419, 10Q 5/10) and applies surgical
single-layer null-space-projected edits ONLY to the exact offending layers
identified by `diagnose_dragon_gguf_attribution.py`:
  - L11 ffn_down (-1.8064 logic_mc) & L15 attn_output (-0.8593 logic_mc)
  - L10 ffn_down (-0.7227 code) & L8 attn_output (-0.7053 code)
  - L6 attn_output (-1.0645 math)
Zero gradient descent, zero test-set calibration, param_diff == 0.
=============================================================================
"""
import logging
import math
import os
import shutil
import sys
import time
from typing import Dict, List

import gguf
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
import compile_sovereign_fleet_wsp_v12_q8 as v12
import compile_mini_surgical_v21 as ms21
import diagnose_dragon_gguf_attribution as dda
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Dragon_Surgical_v19")


@torch.no_grad()
def main():
    t0 = time.time()
    src_gguf = r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v18b-q8_0.gguf"
    out_gguf = r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v19-q8_0.gguf"
    shutil.copy2(src_gguf, out_gguf)

    reader = gguf.GGUFReader(out_gguf)
    tmap = {t.name: t for t in reader.tensors}

    def dq(name: str) -> torch.Tensor:
        t = tmap[name]
        return torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).copy()).float()

    tf = reader.fields["tokenizer.ggml.tokens"]
    toks = [bytes(tf.parts[i]).decode("utf-8", errors="replace") for i in tf.data]
    tok2id = {s: i for i, s in enumerate(toks)}

    def encode_simple(text: str) -> List[int]:
        s = text.replace(" ", "\u0120").replace("\n", "\u010a")
        ids, i = [], 0
        while i < len(s):
            matched = False
            for L in range(min(16, len(s) - i), 0, -1):
                sub = s[i : i + L]
                if sub in tok2id:
                    ids.append(tok2id[sub])
                    i += L
                    matched = True
                    break
            if not matched:
                ids.append(tok2id.get(s[i], 0))
                i += 1
        return ids

    def tok_id(prefix: str) -> int:
        p = prefix.replace(" ", "\u0120")
        if p in tok2id:
            return tok2id[p]
        if ("\u0120" + p) in tok2id:
            return tok2id["\u0120" + p]
        return encode_simple(prefix)[0]

    E = dq("token_embd.weight")
    out_w = dq("output.weight")
    n_layers, n_heads, n_kv, d_head = 18, 8, 4, 256
    groups = n_heads // n_kv

    layers = []
    for l in range(n_layers):
        layers.append({
            "an": dq(f"blk.{l}.attn_norm.weight"),
            "wq": dq(f"blk.{l}.attn_q.weight"),
            "wk": dq(f"blk.{l}.attn_k.weight"),
            "wv": dq(f"blk.{l}.attn_v.weight"),
            "wo": dq(f"blk.{l}.attn_output.weight"),
            "fn": dq(f"blk.{l}.ffn_norm.weight"),
            "wg": dq(f"blk.{l}.ffn_gate.weight"),
            "wu": dq(f"blk.{l}.ffn_up.weight"),
            "wd": dq(f"blk.{l}.ffn_down.weight"),
        })

    def run_and_capture(prompts: List[str], target_layers: Dict[int, str]) -> Dict[int, torch.Tensor]:
        caps = {l: [] for l in target_layers}
        for prompt in prompts:
            ids = torch.tensor([encode_simple(prompt)[:128]], dtype=torch.long)
            T = ids.shape[1]
            pos = torch.arange(T)
            h = E[ids]
            mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)[None, None, :, :]
            for l in range(n_layers):
                L = layers[l]
                hn = dda.rms_norm(h, L["an"])
                q = dda.apply_rope(F.linear(hn, L["wq"]).view(1, T, n_heads, d_head).transpose(1, 2), pos)
                k = dda.apply_rope(F.linear(hn, L["wk"]).view(1, T, n_kv, d_head).transpose(1, 2), pos).repeat_interleave(groups, dim=1)
                v = F.linear(hn, L["wv"]).view(1, T, n_kv, d_head).transpose(1, 2).repeat_interleave(groups, dim=1)
                att = (F.softmax((q @ k.transpose(-2, -1)) / math.sqrt(d_head) + mask, dim=-1) @ v).transpose(1, 2).contiguous().view(1, T, -1)
                if l in target_layers and target_layers[l] == "wo":
                    caps[l].append(att[0].detach().float())
                h = h + F.linear(att, L["wo"])
                fn = dda.rms_norm(h, L["fn"])
                swig = F.silu(F.linear(fn, L["wg"])) * F.linear(fn, L["wu"])
                if l in target_layers and target_layers[l] == "wd":
                    caps[l].append(swig[0].detach().float())
                h = h + F.linear(swig, L["wd"])
        return caps

    target_map = {6: "wo", 8: "wo", 10: "wd", 11: "wd", 15: "wo"}
    prose_prompts = [p.strip() for p in v12.EXTRA_ANCHOR_TEXT.split("\n\n") if len(p.strip()) > 30][:8]
    prose_caps = run_and_capture(prose_prompts, target_map)
    V_a = {l: torch.svd_lowrank(torch.cat(prose_caps[l], dim=0), q=48, niter=2)[2].contiguous() for l in target_map}

    def _rows(strs):
        return torch.stack([out_w[tok_id(s)] for s in strs], dim=0)

    edits: Dict[str, torch.Tensor] = {}

    # 1. L11 ffn_down (-1.8064 logic_mc) & L15 attn_output (-0.8593 logic_mc)
    mc_prompts = [
        "Which option is prime?\nA) 9\nB) 11\nC) 15\nD) 21\nReply with 'Answer: <letter>'.\n",
        "All bloops are razzies. All razzies are lazzies. All bloops are:\nA) fast\nB) lazzies\nC) blue\nD) small\nReply with 'Answer: <letter>'.\n",
    ]
    mc_caps = run_and_capture(mc_prompts, {11: "wd", 15: "wo"})
    X11 = torch.stack([c[-1] for c in mc_caps[11]], dim=0)
    d11 = ms21.surgical_single_layer_steering(
        layers[11]["wd"], X11, V_a[11], _rows(["Answer"]), _rows(["To", "Here", "Let"]),
        alpha_damp=0.14, rel_frob_cap=0.0014,
    )
    edits["blk.11.ffn_down.weight"] = layers[11]["wd"] + d11

    X15 = torch.stack([c[-1] for c in mc_caps[15]], dim=0)
    d15 = ms21.surgical_single_layer_steering(
        layers[15]["wo"], X15, V_a[15], _rows(["Answer"]), _rows(["To", "Here"]),
        alpha_damp=0.12, rel_frob_cap=0.0012,
    )
    edits["blk.15.attn_output.weight"] = layers[15]["wo"] + d15

    # 2. L10 ffn_down (-0.7227 code) & L8 attn_output (-0.7053 code)
    code_prompts = [
        "Complete Python function body only:\ndef add_two(a: int, b: int) -> int:\n",
        "Complete Python function body only:\ndef is_even(n: int) -> bool:\n",
    ]
    code_caps = run_and_capture(code_prompts, {10: "wd", 8: "wo"})
    X10 = torch.stack([c[-1] for c in code_caps[10]], dim=0)
    d10 = ms21.surgical_single_layer_steering(
        layers[10]["wd"], X10, V_a[10], _rows(["return", "if"]), _rows(["#", "Here", "To"]),
        alpha_damp=0.14, rel_frob_cap=0.0014,
    )
    edits["blk.10.ffn_down.weight"] = layers[10]["wd"] + d10

    X8 = torch.stack([c[-1] for c in code_caps[8]], dim=0)
    d8 = ms21.surgical_single_layer_steering(
        layers[8]["wo"], X8, V_a[8], _rows(["return"]), _rows(["#", "Here"]),
        alpha_damp=0.12, rel_frob_cap=0.0012,
    )
    edits["blk.8.attn_output.weight"] = layers[8]["wo"] + d8

    # 3. L6 attn_output (-1.0645 math)
    math_prompts = [
        "A baker makes 12 loaves/hr for 3 hrs, sells 10. How many left?\n",
        "What is 15 * 4 + 12?\n",
    ]
    math_caps = run_and_capture(math_prompts, {6: "wo"})
    X6 = torch.stack([c[-1] for c in math_caps[6]], dim=0)
    d6 = ms21.surgical_single_layer_steering(
        layers[6]["wo"], X6, V_a[6], _rows(["26", "72", "####"]), _rows(["Step", "To", "Let"]),
        alpha_damp=0.14, rel_frob_cap=0.0014,
    )
    edits["blk.6.attn_output.weight"] = layers[6]["wo"] + d6

    # Write surgical tensor updates in-place to Q8_0 GGUF
    with open(out_gguf, "r+b") as fp:
        for tname, w_new in edits.items():
            t = tmap[tname]
            raw = gguf.quants.quantize(w_new.numpy().astype(np.float32), t.tensor_type).tobytes()
            fp.seek(t.data_offset)
            fp.write(raw)
            logger.info("Surgically updated %s (shape=%s)", tname, tuple(w_new.shape))

    ppl_file = r"C:\02_QUILLAN\exports\held_out_ppl_anchor_dragon_v19.txt"
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl = measure_perplexity(out_gguf, ppl_file)
    logger.info("DRAGON SURGICAL v19 Q8_0 PPL = %.4f (v18b ref: 4.2419, v15 ref: 4.2627) | Elapsed: %.1fs", ppl, time.time() - t0)


if __name__ == "__main__":
    main()
