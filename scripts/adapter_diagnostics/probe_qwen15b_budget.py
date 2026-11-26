import sys

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")
from quillan_sovereign_bitnet import QuillanSovereignConfig, QuillanSovereignBitNetLM

# Qwen2.5-1.5B: hidden 1536, 12 heads -> head_dim MUST be 128, kv heads 2.
candidates = [
    (24, 1536, 4864, 12, 2, 128),
    (20, 1536, 4864, 12, 2, 128),
    (24, 1536, 3840, 12, 2, 128),
    (16, 1536, 4864, 12, 2, 128),
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
    qdim = heads * hd
    kvdim = kv * hd
    if qdim != hidden:
        print(f"SKIP {layers}L/{hidden}h: qdim {qdim} != hidden {hidden}")
        continue
    m = QuillanSovereignBitNetLM(c)
    p = m.count_logical_parameters()
    status = "OK" if p <= 1_000_000_000 else "OVER"
    print(f"{layers}L/{hidden}h/{ffn}ffn hd={hd}: {p / 1e6:.0f}M logical {status}")
    del m
