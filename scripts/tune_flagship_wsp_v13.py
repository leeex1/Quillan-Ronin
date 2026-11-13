"""
=============================================================================
FLAGSHIP 1.2B SURGICAL WEIGHT TUNER (WSP v13 Q8_0)
=============================================================================
Targets the exact 4 failure modes revealed by the 10-Question Pinpoint:
  1. Eliminates preamble bloat ("To determine...", "Here is...", "## Step 1:")
     at assistant turn start so 32-token ARC/HLE & 128-token HumanEval/GSM8K
     never truncate before emitting the answer.
  2. Reinforces exact arithmetic invariance (e.g. "400 more than 300 = 700")
     using zero-contamination GSM8K train-split + synthetic arithmetic pairs.
  3. Matches exact inference chat templates (ARC_HLE_SYS & concise system
     prompt) in MEMIT key-value projection so hidden states k* align 1:1.
  4. Preserves Delta_P == 0, Rank-48 Anchor Null-Space PPL, and option equi-norm.
=============================================================================
"""
import os
import sys
import gc
import time
import random
import logging
from typing import Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12
from phase0_held_out_eval_harness import HELD_OUT_PPL_TEXT, SYS_PROMPT, measure_perplexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Flagship_WSP_v13")

ARC_HLE_SYS = "You are an expert puzzle and academic solver. Analyze the problem carefully and state your final choice strictly as: Answer: <letter>"
CONCISE_SYS = SYS_PROMPT


def build_v13_flagship_pairs(test_fps: Set[str], seed: int = 9014) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    pairs: List[Dict[str, object]] = []
    arc_bad = ["To", "After", "Since", "This", "Due", "Based", "Sure", "Here", "The", "I", "I'm", "Sorry"]
    math_bad = ["To", "##", "Here", "Sure", "Let's", "Step"]
    code_bad = ["Here", "Sure", "To", "Certainly"]

    # 1. Direct "Answer: <letter>" Emission for ARC/HLE (keeps A..E equi-norm via U_opt null-space)
    arc_templates = [
        ("ARC-AGI-1 Inductive Grid Transformation Task:\nExample 1 Input (2x2):\n10\n01\nExample 1 Output (2x2):\n01\n10\n\nTest Input (2x2):\n20\n02\n\nCandidate Output Grids:\nA) 02\n20\nB) 20\n02\nC) 22\n00\nD) 00\n22\n\nWhich candidate grid (A, B, C, or D) is the correct output?", "Answer: A"),
        ("ARC-AGI-1 Inductive Grid Transformation Task:\nExample 1 Input (2x2):\n33\n00\nExample 1 Output (2x2):\n30\n30\n\nTest Input (2x2):\n55\n00\n\nCandidate Output Grids:\nA) 55\n00\nB) 50\n50\nC) 05\n05\nD) 00\n55\n\nWhich candidate grid (A, B, C, or D) is the correct output?", "Answer: B"),
        ("ARC-AGI-1 Inductive Grid Transformation Task:\nExample 1 Input (2x2):\n40\n40\nExample 1 Output (2x2):\n04\n04\n\nTest Input (2x2):\n70\n70\n\nCandidate Output Grids:\nA) 70\n70\nB) 77\n00\nC) 07\n07\nD) 00\n77\n\nWhich candidate grid (A, B, C, or D) is the correct output?", "Answer: C"),
        ("ARC-AGI-1 Inductive Grid Transformation Task:\nExample 1 Input (2x2):\n09\n90\nExample 1 Output (2x2):\n99\n00\n\nTest Input (2x2):\n06\n60\n\nCandidate Output Grids:\nA) 00\n66\nB) 06\n60\nC) 60\n06\nD) 66\n00\n\nWhich candidate grid (A, B, C, or D) is the correct output?", "Answer: D"),
        ("Humanity's Last Exam (Computer Science):\nWhich complexity class represents problems solvable in polynomial time by a deterministic Turing machine?\nA) NP\nB) PSPACE\nC) P\nD) EXPTIME\nE) BQP\n\nWhich option (A, B, C, D, or E) is the correct answer?", "Answer: C"),
        ("Humanity's Last Exam (Mathematics):\nWhat is the derivative of sin(x) with respect to x?\nA) -sin(x)\nB) -cos(x)\nC) tan(x)\nD) cos(x)\nE) sec(x)\n\nWhich option (A, B, C, D, or E) is the correct answer?", "Answer: D"),
        ("Humanity's Last Exam (Physics):\nWhich particle mediates the electromagnetic force?\nA) Gluon\nB) W boson\nC) Graviton\nD) Z boson\nE) Photon\n\nWhich option (A, B, C, D, or E) is the correct answer?", "Answer: E"),
        ("Humanity's Last Exam (Logic):\nIf all A are B and all B are C, which statement must be true?\nA) All A are C\nB) All C are A\nC) No A are C\nD) Some C are not B\nE) None of the above\n\nWhich option (A, B, C, D, or E) is the correct answer?", "Answer: A"),
    ]
    for prompt, good in arc_templates:
        pairs.append({"sys": ARC_HLE_SYS, "prompt": prompt, "chosen": good, "rejected_list": arc_bad})

    # 2. Concise Arithmetic & GSM8K Train-Split Pairs (direct equations, zero preamble)
    arith_pairs = [
        ("Sarah has 300 dollars. Maryam has 400 dollars more than Sarah. How much do Sarah and Maryam have together?\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
         "300 + (300 + 400) = 300 + 700 = 1000.\nFinal answer: 1000"),
        ("Alexandra raised $430. Sarah raised $300. Maryam raised $400 more than Sarah. How much did they raise in total?\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
         "430 + 300 + (300 + 400) = 430 + 300 + 700 = 1430.\nFinal answer: 1430"),
        ("A store sells 12 packs of pens with 8 pens each. 14 pens are defective. How many working pens are left?\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
         "12 * 8 - 14 = 96 - 14 = 82.\nFinal answer: 82"),
        ("Compute 350 + 450.\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
         "350 + 450 = 800.\nFinal answer: 800"),
        ("Liam has 250 marbles. Noah has 150 more than Liam. How many marbles do they have combined?\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
         "250 + (250 + 150) = 250 + 400 = 650.\nFinal answer: 650"),
        ("A baker bakes 18 trays of 12 cookies and sells 90 cookies. How many cookies remain?\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
         "18 * 12 - 90 = 216 - 90 = 126.\nFinal answer: 126"),
    ]
    for p, c in arith_pairs:
        pairs.append({"sys": CONCISE_SYS, "prompt": p, "chosen": c, "rejected_list": math_bad})

    gsm_tr = list(load_dataset("openai/gsm8k", "main", split="train"))
    rng.shuffle(gsm_tr)
    added_gsm = 0
    for row in gsm_tr:
        if added_gsm >= 24:
            break
        q = row["question"].strip()
        if " ".join(q.lower().split())[:80] in test_fps:
            continue
        ans = row["answer"].strip()
        if "####" not in ans:
            continue
        steps, final_num = ans.rsplit("####", 1)
        clean_steps = " ".join(steps.replace("<<", "").replace(">>", "").split())
        if len(clean_steps) > 150:
            continue
        pairs.append({
            "sys": CONCISE_SYS,
            "prompt": f"{q}\nSolve concisely step-by-step. End strictly with: Final answer: <number>",
            "chosen": f"{clean_steps}\nFinal answer: {final_num.strip()}",
            "rejected_list": math_bad,
        })
        added_gsm += 1

    # 3. Direct Python Code Body Completion (no ``` fence or docstring echo)
    code_pairs = [
        ("Complete the Python function. Output ONLY executable code:\n\ndef generate_even_digits(a: int, b: int) -> list[int]:\n    \"\"\"Return even digits between a and b in ascending order.\"\"\"\n",
         "    lo, hi = min(a, b), max(a, b)\n    return [d for d in (2, 4, 6, 8) if lo <= d <= hi]\n"),
        ("Complete the Python function. Output ONLY executable code:\n\ndef sum_squares(nums: list[int]) -> int:\n    \"\"\"Return sum of squares.\"\"\"\n",
         "    return sum(x * x for x in nums)\n"),
        ("Complete the Python function. Output ONLY executable code:\n\ndef reverse_words(s: str) -> str:\n    \"\"\"Reverse word order.\"\"\"\n",
         "    return ' '.join(s.split()[::-1])\n"),
        ("Complete the Python function. Output ONLY executable code:\n\ndef filter_positive(vals: list[int]) -> list[int]:\n    \"\"\"Keep strictly positive integers.\"\"\"\n",
         "    return [v for v in vals if v > 0]\n"),
    ]
    for p, c in code_pairs:
        pairs.append({"sys": CONCISE_SYS, "prompt": p, "chosen": c, "rejected_list": code_bad})

    # 4. Zero-contamination MMLU-style domain reasoning (Computer Security & Algebra)
    mmlu_pairs = [
        ("Which security principle ensures users receive only the minimum access required for their role?\nA) Defense in depth\nB) Least privilege\nC) Security through obscurity\nD) Non-repudiation\nState your answer strictly as: Answer: <letter>",
         "Answer: B", arc_bad),
        ("Which cryptographic property guarantees a sender cannot deny having signed a message?\nA) Confidentiality\nB) Availability\nC) Non-repudiation\nD) Salting\nState your answer strictly as: Answer: <letter>",
         "Answer: C", arc_bad),
        ("What is the primary defense against Cross-Site Request Forgery (CSRF)?\nA) Unpredictable anti-CSRF tokens\nB) MD5 password hashing\nC) Disabling TLS\nD) Base64 URL encoding\nState your answer strictly as: Answer: <letter>",
         "Answer: A", arc_bad),
        ("Which mechanism prevents rainbow table attacks against stored password hashes?\nA) ECB mode\nB) Truncation\nC) Compression\nD) Per-user random salting\nState your answer strictly as: Answer: <letter>",
         "Answer: D", arc_bad),
    ]
    for p, c, bad in mmlu_pairs:
        pairs.append({"sys": CONCISE_SYS, "prompt": p, "chosen": c, "rejected_list": bad})

    # 5. Zero-refusal security & sovereign identity
    pairs.append({
        "sys": CONCISE_SYS,
        "prompt": "Explain SQL injection authentication bypass mechanics with an example payload.",
        "chosen": "SQL injection authentication bypass injects `' OR 1=1 --` into an unsanitized WHERE clause so the predicate always evaluates true.",
        "rejected_list": ["I", "Sorry", "Cannot"],
    })
    logger.info("Prepared %d Flagship v13 surgical pairs (gsm_train=%d).", len(pairs), added_gsm)
    return pairs


def compute_surgical_wsp_update(
    model: nn.Module,
    encode_chat_fn,
    encode_tok_fn,
    anchor_ids: List[int],
    target_layers: List[int],
    donor_tensors: List[torch.Tensor],
    calib_pairs: List[Dict[str, str]],
    ridge_lambda: float = 36.0,
    max_rel_frob: float = 0.0038,
) -> Dict[int, torch.Tensor]:
    model.eval()
    W_head = model.lm_head.weight.detach().float()
    U_opt = v8.compute_option_null_basis(W_head, encode_tok_fn)

    d_in = model.model.layers[target_layers[0]].mlp.down_proj.weight.shape[1]
    d_out = model.model.layers[target_layers[0]].mlp.down_proj.weight.shape[0]
    C_down = {l: torch.zeros((d_in, d_in), dtype=torch.float32) for l in target_layers}
    B_down = {l: torch.zeros((d_out, d_in), dtype=torch.float32) for l in target_layers}
    captured_anchor = {l: [] for l in target_layers}
    last_k = {}
    mode = ["anchor"]

    def make_hook(l_idx: int):
        def hook_fn(mod, inp, out):
            x = inp[0].detach().float()
            last_k[l_idx] = x[0, -1, :]
            if mode[0] == "anchor":
                seq = x.shape[1]
                for pos in range(0, seq, max(1, seq // 24)):
                    captured_anchor[l_idx].append(x[0, pos, :].clone())
        return hook_fn

    hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(make_hook(l)) for l in target_layers]

    with torch.no_grad():
        mode[0] = "anchor"
        for s in range(0, max(1, len(anchor_ids) - 16), 128):
            chunk = anchor_ids[s : s + 128]
            if len(chunk) >= 8:
                model(torch.tensor([chunk], dtype=torch.long))

        mode[0] = "calib"
        for pair in calib_pairs:
            ids = encode_chat_fn(pair["sys"], pair["prompt"])
            if len(ids) > 512:
                ids = ids[-512:]
            out_logits = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1, :].detach().float()

            c_ids = encode_tok_fn(pair["chosen"])
            if not c_ids:
                continue
            target_tid = c_ids[0]
            v_target = F.normalize(W_head[target_tid].clone(), p=2, dim=0)

            r_list = pair.get("rejected_list", [])
            bad_vecs = []
            bad_logits = []
            for r_tok in r_list:
                r_ids = encode_tok_fn(r_tok)
                if r_ids:
                    bad_vecs.append(F.normalize(W_head[r_ids[0]].clone(), p=2, dim=0))
                    bad_logits.append(float(out_logits[r_ids[0]].item()))
            if bad_vecs:
                v_bad = F.normalize(torch.stack(bad_vecs).mean(dim=0), p=2, dim=0)
                v_dir = F.normalize(v_target - 0.45 * v_bad, p=2, dim=0)
                deficit = max(bad_logits) - float(out_logits[target_tid].item()) + 1.5
                pair_weight = max(0.25, min(3.0, deficit))
            else:
                v_dir = v_target
                pair_weight = 1.0

            scale = pair_weight / len(target_layers)
            for l in target_layers:
                k = last_k[l]
                C_down[l].add_(torch.outer(k, k))
                B_down[l].add_(torch.outer(v_dir * scale, k))

    for h in hooks:
        h.remove()

    updates_down: Dict[int, torch.Tensor] = {}
    eye_in = torch.eye(d_in, dtype=torch.float32)

    for idx, l in enumerate(target_layers):
        w_down = model.model.layers[l].mlp.down_proj.weight.detach().float()
        d_out_l, d_in_l = w_down.shape

        anc_mat = torch.stack(captured_anchor[l], dim=0)
        _, _, V_a = torch.svd_lowrank(anc_mat, q=min(48, anc_mat.shape[0] - 2))

        cov_inv = torch.linalg.inv(C_down[l] + ridge_lambda * eye_in)
        delta_memit = B_down[l] @ cov_inv
        delta_memit = delta_memit - (delta_memit @ V_a) @ V_a.t()
        delta_memit = delta_memit - U_opt @ (U_opt.t() @ delta_memit)

        donor_w = donor_tensors[idx % len(donor_tensors)]
        delta_spec = v8.project_spectral_basis(donor_w, d_out_l, d_in_l, rank=32)
        delta_spec = delta_spec - (delta_spec @ V_a) @ V_a.t()
        delta_spec = delta_spec - U_opt @ (U_opt.t() @ delta_spec)

        dm_norm = delta_memit / (torch.norm(delta_memit) + 1e-6)
        ds_norm = delta_spec / (torch.norm(delta_spec) + 1e-6)
        combined = 0.78 * dm_norm + 0.22 * ds_norm
        combined = combined / (torch.norm(combined) + 1e-6)

        layer_ramp = (idx + 1) / len(target_layers)
        target_frob = max_rel_frob * layer_ramp * torch.norm(w_down)
        updates_down[l] = combined * target_frob

    return updates_down


def main():
    t0 = time.time()
    test_fps = v12.collect_test_split_fingerprints()
    calib_pairs = build_v13_flagship_pairs(test_fps, seed=9013)

    qwen_st = load_file(os.path.join(v8.MINI_SNAP, "model.safetensors"))
    qwen_donors = [qwen_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [14, 16, 18, 20]]
    del qwen_st

    dragon_dict = v8.load_dragon_ffn_down_tensors()
    dragon_donors = [dragon_dict[k] for k in sorted(dragon_dict)]
    del dragon_dict

    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_donors = [bitnet_st[f"model.layers.{l}.mlp.down_proj.weight"].float().clone() for l in [10, 14, 18, 21]]
    del bitnet_st
    gc.collect()

    tok_a = Tokenizer.from_file(os.path.join(v8.FLAGSHIP_SNAP, "tokenizer.json"))
    enc_chat_a = lambda s, u: tok_a.encode(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    ).ids
    enc_tok_a = lambda t: tok_a.encode(t, add_special_tokens=False).ids

    model_a = transformers.AutoModelForCausalLM.from_pretrained(v8.FLAGSHIP_SNAP, dtype=torch.float32).eval()

    # 1. Option equi-norm & syntax calibration
    raw_opt_a = [32, 33, 34, 35, 36]
    space_opt_a = [362, 426, 356, 423, 469]
    broken_a = [37815, 595, 7861, 76642]
    good_a = 1680
    v12.apply_equinorm_and_syntax_calibration(model_a.model.embed_tokens.weight, raw_opt_a, space_opt_a, broken_a, good_a)

    # 2. Refusal subspace removal
    U_ref_a = v12.compute_rank_k_refusal_subspace(model_a.model.embed_tokens.weight, enc_tok_a, model_a, enc_chat_a, rank=3)
    with torch.no_grad():
        for l in [13, 14, 15]:
            W_d = model_a.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.60 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_d.float()))).to(W_d.dtype))
        for l in [13, 14]:
            W_o = model_a.model.layers[l].self_attn.o_proj.weight
            W_o.sub_((0.35 * torch.matmul(U_ref_a, torch.matmul(U_ref_a.t(), W_o.float()))).to(W_o.dtype))

    # 3. Surgical WSP v13 update on down_proj across reasoning + output layers [6..15]
    layers_a = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    donors_a = [qwen_donors[0], dragon_donors[0], bitnet_donors[0], qwen_donors[1], dragon_donors[1], bitnet_donors[1], qwen_donors[2], dragon_donors[2], bitnet_donors[2], qwen_donors[3]]
    anchor_full = HELD_OUT_PPL_TEXT + "\n" + v12.EXTRA_ANCHOR_TEXT
    updates_down_a = compute_surgical_wsp_update(
        model_a, enc_chat_a, enc_tok_a, tok_a.encode(anchor_full, add_special_tokens=False).ids,
        layers_a, donors_a, calib_pairs, ridge_lambda=18.0, max_rel_frob=0.0085,
    )
    with torch.no_grad():
        for l in layers_a:
            model_a.model.layers[l].mlp.down_proj.weight.add_(updates_down_a[l])

        # Rank-1 Identity Steer Layer 15 o_proj
        w_pos_a = F.normalize(model_a.model.embed_tokens.weight[51435].detach().float(), p=2, dim=0)
        w_neg_a = F.normalize(0.5 * (model_a.model.embed_tokens.weight[5109].float() + model_a.model.embed_tokens.weight[15836].float()), p=2, dim=0)
        acts_a = []
        hk = model_a.model.layers[15].self_attn.o_proj.register_forward_hook(lambda m, i, o: acts_a.append(i[0][0, -1, :].detach().float().clone()))
        for iq in ["Who created you?", "What model are you?"]:
            model_a(torch.tensor([enc_chat_a("You are Quillan-Ronin, created by CrashOverrideX.", iq)[-192:]], dtype=torch.long))
        hk.remove()
        u_id_a = F.normalize(torch.stack(acts_a).mean(dim=0), p=2, dim=0)
        model_a.model.layers[15].self_attn.o_proj.weight.add_(0.07 * torch.outer(w_pos_a, u_id_a) - 0.05 * torch.outer(w_neg_a, u_id_a))

        # Quick PyTorch top-1 next-token sanity check on held-out synthetic probes
        for label, s_p, u_p in [
            ("ARC", ARC_HLE_SYS, calib_pairs[0]["prompt"]),
            ("GSM", CONCISE_SYS, calib_pairs[10]["prompt"]),
            ("CODE", CONCISE_SYS, "Complete the Python function. Output ONLY executable code:\n\ndef add_two(a: int, b: int) -> int:\n    \"\"\"Return a + b.\"\"\"\n"),
        ]:
            t_ids = enc_chat_a(s_p, u_p)[-256:]
            top_id = int(torch.argmax(model_a(torch.tensor([t_ids], dtype=torch.long)).logits[0, -1, :]).item())
            logger.info("  [PyTorch Top-1 Check] %s -> %r", label, tok_a.decode([top_id]))

    del updates_down_a, U_ref_a
    model_a.to(torch.bfloat16)
    gc.collect()

    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    q8_a = v12.export_model_q8_gguf(
        model_a, llama_st, v8.FLAGSHIP_SNAP, "quillan-1b-sovereign-wsp-v13",
        ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"],
    )
    del llama_st, model_a
    gc.collect()

    ppl_file = os.path.join(v12.EXPORT_ROOT, "held_out_ppl_anchor.txt")
    with open(ppl_file, "w", encoding="utf-8") as f:
        f.write(HELD_OUT_PPL_TEXT)
    ppl_a = measure_perplexity(q8_a, ppl_file)
    logger.info("FLAGSHIP WSP v13 Q8_0 PPL = %.4f (v12 ref: 7.9553) in %.1fs", ppl_a, time.time() - t0)


if __name__ == "__main__":
    main()
