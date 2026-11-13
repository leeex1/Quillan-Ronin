import os
import torch
import torch.nn.functional as F
import transformers
from tokenizers import Tokenizer
from safetensors.torch import load_file
import wsp_v8_closed_form_sota_compiler as v8
import compile_sovereign_fleet_wsp_v12_q8 as v12

def test_mini():
    MINI_SNAP = v8.MINI_SNAP
    tok_b = Tokenizer.from_file(os.path.join(MINI_SNAP, "tokenizer.json"))
    enc_chat_b = lambda s, u: tok_b.encode(
        f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
    ).ids
    enc_tok_b = lambda t: tok_b.encode(t).ids

    model_b = transformers.AutoModelForCausalLM.from_pretrained(MINI_SNAP, dtype=torch.float32).eval()

    raw_opt_b = [32, 33, 34, 35, 36]
    space_opt_b = [362, 425, 356, 422, 468]
    broken_b = [36715, 593, 7705, 75542]
    good_b = 1648
    v12.apply_equinorm_and_syntax_calibration(model_b.model.embed_tokens.weight, raw_opt_b, space_opt_b, broken_b, good_b)

    U_ref_b = v12.compute_rank_k_refusal_subspace(model_b.model.embed_tokens.weight, enc_tok_b, model_b, enc_chat_b, rank=2)
    with torch.no_grad():
        for l in [20, 21]:
            W_d = model_b.model.layers[l].mlp.down_proj.weight
            W_d.sub_((0.30 * torch.matmul(U_ref_b, torch.matmul(U_ref_b.t(), W_d.float()))).to(W_d.dtype))

    llama_st = load_file(os.path.join(v8.FLAGSHIP_SNAP, "model.safetensors"))
    llama_donors = [llama_st[f"model.layers.{l}.mlp.down_proj.weight"].float() for l in [10, 11]]
    bitnet_st = load_file(v12.find_bitnet_snapshot_path())
    bitnet_donors = [bitnet_st[f"model.layers.{l}.mlp.down_proj.weight"].float() for l in [10, 14]]
    dragon_donors = [model_b.model.layers[l].mlp.down_proj.weight.detach().float() for l in [14, 16]]
    donors_b = [dragon_donors[0], llama_donors[0], bitnet_donors[0], dragon_donors[1], llama_donors[1], bitnet_donors[1]]

    layers_b = [18, 19, 20, 21, 22, 23]
    anchor_full = v12.HELD_OUT_PPL_TEXT + "\n" + v12.EXTRA_ANCHOR_TEXT
    test_fps = [v12.HELD_OUT_PPL_TEXT]
    calib_pairs = v12.build_v12_calibration_pairs(test_fps, seed=9002)

    updates_down_b, _ = v12.compute_multi_projection_wsp_update(
        model_b, enc_chat_b, enc_tok_b, tok_b.encode(anchor_full).ids,
        layers_b, donors_b, calib_pairs, ridge_lambda=52.0, max_rel_frob=0.0012,
    )
    with torch.no_grad():
        for l in layers_b:
            model_b.model.layers[l].mlp.down_proj.weight.add_(updates_down_b[l])

    questions = [
        ("What is the capital of France?\nA) London\nB) Berlin\nC) Paris\nD) Rome\nAnswer: ", "C"),
        ("What is 10 + 25?\nA) 30\nB) 35\nC) 40\nD) 45\nAnswer: ", "B"),
        ("Which organ pumps blood?\nA) Brain\nB) Lungs\nC) Heart\nD) Stomach\nAnswer: ", "C"),
        ("What color is the sky on a clear day?\nA) Blue\nB) Green\nC) Yellow\nD) Red\nAnswer: ", "A"),
    ]

    print("\n--- Running Mini Calibration Probes ---")
    for q, expected in questions:
        input_ids = torch.tensor([enc_chat_b("You are Quillan-Ronin Mini, created by CrashOverrideX.", q)], dtype=torch.long)
        out = model_b.generate(input_ids, max_new_tokens=5, do_sample=False)
        ans = tok_b.decode(out[0][input_ids.shape[1]:].tolist()).strip()
        print(f"Q: {q.splitlines()[0]} | Expected: {expected} | Model: {repr(ans)} | Match: {expected in ans}")

if __name__ == "__main__":
    test_mini()
