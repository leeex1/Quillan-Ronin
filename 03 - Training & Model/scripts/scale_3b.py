import sys
sys.path.insert(0, r"C:\02_QUILLAN\scripts")
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

for hidden, ffn in [(1024, 2048), (1536, 4096), (2048, 4096), (2048, 8192)]:
    cfg = QuillanOniConfig(vocab_size=50262, hidden_dim=hidden, ffn_dim=ffn,
                           n_layer=12, num_experts=34, top_k=4, max_seq_len=512)
    m = QuillanRoninOni(cfg)
    n = sum(p.numel() for p in m.parameters())
    print(f"12L h={hidden} ffn={ffn}: {n/1e9:.2f}B params", flush=True)
    del m
