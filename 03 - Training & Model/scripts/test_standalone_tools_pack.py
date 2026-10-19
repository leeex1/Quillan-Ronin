"""Execution tests for StandaloneToolsPack (checklist rows 28, 30, 31, 40)."""
import torch

from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni
from paper_121_standalone_tools_pack import StandaloneToolsPack


def _build(flag: bool) -> QuillanRoninOni:
    torch.manual_seed(0)
    cfg = QuillanOniConfig(hidden_dim=64, n_layer=2, num_experts=34, vocab_size=200,
                           use_standalone_tools=flag)
    return QuillanRoninOni(cfg)


def test_off_by_default():
    assert QuillanOniConfig().use_standalone_tools is False
    assert _build(False).standalone_tools is None


def test_pack_shapes_and_validation():
    pack = StandaloneToolsPack(hidden_dim=64)
    out = pack(torch.randn(3, 64))
    assert out["loss"].dim() == 0 and torch.isfinite(out["loss"])
    assert out["consciousness_score"].shape == (3,)
    assert ((out["consciousness_score"] >= 0) & (out["consciousness_score"] <= 1)).all()
    assert out["tactic_entropy"].shape == (3,)
    assert pack.handshake_ok is True
    try:
        pack(torch.randn(3, 10))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong hidden dim must raise ValueError")


def test_model_aux_loss_and_gradients():
    model = _build(True)
    assert model.standalone_tools is not None
    x = torch.randint(0, 200, (2, 8))
    y = torch.randint(0, 200, (2, 8))
    model.zero_grad()
    logits, ce, aux = model(x, labels=y)
    assert "standalone_tools_loss" in aux, "aux must carry the pack regularizer"
    total_with = ce + model.total_aux_loss(aux)
    aux_without = {k: v for k, v in aux.items() if k != "standalone_tools_loss"}
    total_without = ce + model.total_aux_loss(aux_without)
    assert abs(float(total_with.detach()) - float(total_without.detach())) > 0, \
        "pack must change the total loss"
    total_with.backward()
    g1 = model.standalone_tools.consciousness.proj.weight.grad
    g2 = model.standalone_tools.prover_scorer.goal_proj.weight.grad
    assert g1 is not None and g1.norm().item() > 0, "consciousness head must get gradients"
    assert g2 is not None and g2.norm().item() > 0, "prover scorer must get gradients"


def test_logits_unaffected():
    model = _build(True).eval()
    x = torch.randint(0, 200, (2, 8))
    with torch.no_grad():
        torch.manual_seed(123)
        on = model(x)[0]
        pack, model.standalone_tools = model.standalone_tools, None
        torch.manual_seed(123)
        off = model(x)[0]
        model.standalone_tools = pack
    assert torch.allclose(on, off, atol=1e-6), "pack is a regularizer and must not change logits"


def test_sol_pi_tool_output_path():
    pack = StandaloneToolsPack(hidden_dim=64)
    out = pack.process_tool_output("ls", 0, "short output", 1)
    assert isinstance(out, str) and len(out) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  [PASS] {name}")
    print("ALL STANDALONE-TOOLS-PACK TESTS PASSED")
