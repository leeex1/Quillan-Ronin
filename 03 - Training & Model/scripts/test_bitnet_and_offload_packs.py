"""
Tests for BitNetFamilyPack (rows 32-39) and HeterogeneousOffloadPack (rows 41, 43, 51).
"""
import torch

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from paper_122_bitnet_family_pack import BitNetFamilyPack
from paper_123_heterogeneous_offload_pack import HeterogeneousOffloadPack


def test_bitnet_family_standalone():
    """Verify all BitNet variants execute forward passes with valid loss."""
    variants = ["b158", "a48", "v2", "2b4t", "beyond158", "scaling"]
    x = torch.randn(2, 64)
    for v in variants:
        pack = BitNetFamilyPack(hidden_dim=64, variant=v)
        res = pack(x)
        assert res["output"].shape == (2, 64), f"Variant {v} failed output shape"
        assert res["quant_loss"].dim() == 0 and torch.isfinite(res["quant_loss"]), f"Variant {v} invalid loss"
        assert res["variant"] == v


def test_hetero_offload_standalone():
    """Verify DALI + DOS offload pack executes forward and reports telemetry."""
    pack = HeterogeneousOffloadPack(embed_dim=64, num_experts=8, top_k=2)
    x = torch.randn(2, 64)
    res = pack(x)
    assert res["output"].shape == (2, 64), "Offload output shape mismatch"
    assert "cache_hit_rate" in res
    telem = pack.get_telemetry()
    assert telem["hetero_offload_active"] is True
    assert telem["num_experts"] == 8


def test_model_integration_bitnet_and_offload():
    """Verify QuillanRoninOni integration with both packs active."""
    cfg = QuillanOniConfig(
        hidden_dim=64,
        n_layer=2,
        num_experts=34,
        vocab_size=200,
        use_bitnet_family=True,
        bitnet_variant="b158",
        use_hetero_offload=True
    )
    model = QuillanRoninOni(cfg)
    assert model.bitnet_family is not None
    assert model.hetero_offload is not None

    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    model.zero_grad()
    logits, ce, aux = model(x, labels=y)
    assert "bitnet_family_loss" in aux, "aux must contain bitnet_family_loss"
    total_loss = ce + model.total_aux_loss(aux)
    total_loss.backward()

    # Verify gradients flow into bitnet family parameters
    g_b158 = model.bitnet_family.b158_linear.weight.grad
    assert g_b158 is not None and g_b158.norm().item() > 0, "Gradients must flow to BitNet linear"


if __name__ == "__main__":
    test_bitnet_family_standalone()
    print("  [PASS] test_bitnet_family_standalone")
    test_hetero_offload_standalone()
    print("  [PASS] test_hetero_offload_standalone")
    test_model_integration_bitnet_and_offload()
    print("  [PASS] test_model_integration_bitnet_and_offload")
    print("ALL BITNET & OFFLOAD INTEGRATION TESTS PASSED!")
