import sys

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")
from quillan_sovereign_bitnet import QuillanSovereignConfig, QuillanSovereignBitNetLM

# Proven mini recipe: hidden 896, 14 heads, head_dim 64, 24L -> passes 7/10.
# For flagship scale up within <=1B keeping head_dim=64 and host vocab 151936.
# Use hidden 1024 (16 heads x 64), ffn 3072, more layers.
candidates = [
    (24, 1024, 3072, 16, 2, 64),
    (28, 1024, 3072, 16, 2, 64),
    (32, 1024, 3072, 16, 2, 64),
    (24, 896, 4864, 14, 2, 64),  # exactly mini recipe
]
for layers, hidden, ffn, heads, kv, hd in candidates:
    c = QuillanSovereignConfig.for_model("flagship")
    c.num_hidden_layers = layers
    c.hidden_size = hidden
    c.intermediate_size = ffn
    c.num_attention_heads = heads
    c.num_key_value_heads = kv
    c.head_dim = hd
    c.vocab_size = 151936
    c.moe_layer_indices = (layers // 2,)
    c.moe_ffn_dim = 256
    if heads * hd != hidden:
        print(f"SKIP heads*hd {heads * hd} != hidden {hidden}")
        continue
    m = QuillanSovereignBitNetLM(c)
    p = m.count_logical_parameters()
    print(
        f"{layers}L/{hidden}h/{ffn}ffn hd={hd}: {p / 1e6:.0f}M logical {'OK' if p <= 1e9 else 'OVER'}"
    )
    del m
