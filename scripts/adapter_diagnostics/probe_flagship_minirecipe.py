import sys

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model")
from quillan_sovereign_bitnet import QuillanSovereignConfig, QuillanSovereignBitNetLM

# Mini recipe (PROVEN passes 7/10): hidden 896, 14 heads, head_dim 64, 24L,
# vocab 151936, 4 expert layers (0,8,15,23), moe_ffn 512 -> 690M logical.
# Flagship: same proven backbone, LARGER expert bank (moe_ffn 1024) -> bigger capacity.
for moe_ffn in [512, 768, 1024, 1280]:
    c = QuillanSovereignConfig.for_model("mini")
    c.num_hidden_layers = 24
    c.hidden_size = 896
    c.intermediate_size = 4864
    c.num_attention_heads = 14
    c.num_key_value_heads = 2
    c.head_dim = 64
    c.vocab_size = 151936
    c.moe_layer_indices = (0, 8, 15, 23)
    c.moe_ffn_dim = moe_ffn
    m = QuillanSovereignBitNetLM(c)
    p = m.count_logical_parameters()
    a = sum(x.numel() for x in m.parameters()) + sum(x.numel() for x in m.buffers())
    print(
        f"moe_ffn={moe_ffn}: {p / 1e6:.0f}M logical / {a / 1e6:.0f}M actual {'OK' if p <= 1e9 else 'OVER'}"
    )
    del m
