import sys
import torch
from safetensors import safe_open

d_llama = r"C:\Users\Admin\.cache\huggingface\hub\models--unsloth--Llama-3.2-1B-Instruct\snapshots\5a8abab4a5d6f164389b1079fb721cfab8d7126c\model.safetensors"

sys.path.insert(0, r'C:\02_QUILLAN\03 - Training & Model\scripts')
from compile_wsp_tripartite import project_dim, to_bitnet_ternary_wsp, remove_attractor_direction, normalize_layernorm_vector
from build_true_tripartite_models import extract_svd_expert

with safe_open(d_llama, framework="pt", device="cpu") as f_llama:
    prefix = "model.layers.0"
    print("Testing LN1...")
    raw_ln1 = f_llama.get_tensor(f"{prefix}.input_layernorm.weight")
    ln1 = normalize_layernorm_vector(raw_ln1, 1152, target_mean=1.0)
    print("LN1 OK:", ln1.shape)
    
    print("Testing Q, K, V...")
    raw_q = f_llama.get_tensor(f"{prefix}.self_attn.q_proj.weight")
    w_q = project_dim(raw_q, 1152, 1152)
    w_q_t = to_bitnet_ternary_wsp(w_q, target_gain=1.0)
    print("Q OK:", w_q_t.shape)
    
    print("Testing O proj + attractor removal...")
    raw_o = f_llama.get_tensor(f"{prefix}.self_attn.o_proj.weight")
    w_o = project_dim(raw_o, 1152, 1152)
    w_o_clean = remove_attractor_direction(w_o, num_directions=1)
    print("O clean OK:", w_o_clean.shape)
    
    print("Testing FFN...")
    raw_down = f_llama.get_tensor(f"{prefix}.mlp.down_proj.weight")
    raw_down_clean = remove_attractor_direction(raw_down, num_directions=1)
    print("Down clean OK:", raw_down_clean.shape)
    
    print("Testing SVD expert extraction...")
    w0, w1 = extract_svd_expert(raw_down_clean, 1152, rank=8)
    print("Expert OK:", w0.shape, w1.shape)

print("ALL TESTS PASSED!")

