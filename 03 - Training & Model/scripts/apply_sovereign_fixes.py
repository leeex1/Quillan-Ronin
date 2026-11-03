import os
import sys

def patch_oni():
    oni_path = r"C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py"
    with open(oni_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. RoPE base: 10000.0 -> 500000.0
    s_target = "def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):"
    s_repl = "def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 500000.0):"
    assert s_target in content, "Failed to find RotaryEmbedding init"
    content = content.replace(s_target, s_repl, 1)

    # 2. ln_1 and ln_2 in UnrolledTransformerBlock: LayerNorm -> RMSNorm
    s_target = """class UnrolledTransformerBlock(nn.Module):
    def __init__(self, cfg: QuillanOniConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_dim, eps=1e-5)
        if cfg.use_mamba and _FORMAL_PAPERS_WIRED and MambaBlock is not None:
            self.attn = MambaBlock(cfg.hidden_dim)
        else:
            self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.hidden_dim, eps=1e-5)"""
    
    s_repl = """class UnrolledTransformerBlock(nn.Module):
    def __init__(self, cfg: QuillanOniConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(cfg.hidden_dim, eps=1e-5)
        if cfg.use_mamba and _FORMAL_PAPERS_WIRED and MambaBlock is not None:
            self.attn = MambaBlock(cfg.hidden_dim)
        else:
            self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.RMSNorm(cfg.hidden_dim, eps=1e-5)"""
    assert s_target in content, "Failed to find UnrolledTransformerBlock norms"
    content = content.replace(s_target, s_repl, 1)

    # 3. ln_f in QuillanRoninOni: LayerNorm -> RMSNorm
    s_target = "self.ln_f = nn.LayerNorm(cfg.hidden_dim, eps=1e-5)"
    s_repl = "self.ln_f = nn.RMSNorm(cfg.hidden_dim, eps=1e-5)"
    assert s_target in content, "Failed to find ln_f"
    content = content.replace(s_target, s_repl, 1)

    # 4. Clean residual bypass during inference
    s_target = """        # Recirculation: deep->shallow feedback bias (Law 6 recirculation loop)
        if recirc_state is not None:
            if recirc_state.dim() == 2:
                recirc_state = recirc_state.unsqueeze(1)
            recirc_in = recirc_state.to(x.dtype)
            x = x + self.recirc_proj(recirc_in).to(x.dtype) + 0.1 * (recirc_in - x.detach())
        else:
            # Active in-graph self-recirculation anchor across layers
            x = x + 0.01 * self.recirc_proj(x).to(x.dtype)

        # Law 1: Quillan Core (Dual-Hemisphere q1/q2) + Personal Swarm (c0-quillan-swarm) Ingest & Route
        q1 = self.q1_bridge(x)
        q2 = self.q2_bridge(x)
        c0_swarm_idea = torch.tanh(q1) * q2 + 0.467 * (torch.sin(q2) * q1)
        g_ingest = torch.sigmoid(self.ingest_gate(torch.cat([q1, q2], dim=-1)))
        g_f32 = g_ingest.float().clamp(1e-5, 1.0 - 1e-5)
        ingest_entropy = -(g_f32 * torch.log(g_f32) + (1.0 - g_f32) * torch.log(1.0 - g_f32)).mean()
        x = x + 0.05 * (g_ingest * q1 + (1.0 - g_ingest) * q2 + 0.25 * c0_swarm_idea)
        x_embed = x"""

    s_repl = """        if self.training:
            if recirc_state is not None:
                if recirc_state.dim() == 2:
                    recirc_state = recirc_state.unsqueeze(1)
                recirc_in = recirc_state.to(x.dtype)
                x = x + self.recirc_proj(recirc_in).to(x.dtype) + 0.1 * (recirc_in - x.detach())
            else:
                x = x + 0.01 * self.recirc_proj(x).to(x.dtype)

            q1 = self.q1_bridge(x)
            q2 = self.q2_bridge(x)
            c0_swarm_idea = torch.tanh(q1) * q2 + 0.467 * (torch.sin(q2) * q1)
            g_ingest = torch.sigmoid(self.ingest_gate(torch.cat([q1, q2], dim=-1)))
            g_f32 = g_ingest.float().clamp(1e-5, 1.0 - 1e-5)
            ingest_entropy = -(g_f32 * torch.log(g_f32) + (1.0 - g_f32) * torch.log(1.0 - g_f32)).mean()
            x = x + 0.05 * (g_ingest * q1 + (1.0 - g_ingest) * q2 + 0.25 * c0_swarm_idea)
        x_embed = x"""
    assert s_target in content, "Failed to find recirculation/ingest bridge"
    content = content.replace(s_target, s_repl, 1)

    # 5. Finalizer clean bypass
    s_target = """        # Law 5: Dual Quillan Finalizer Consensus (Everything reports back to main Quillan Core before output)
        q1_out = self.quillan_finalizer_q1(hidden)
        q2_out = self.quillan_finalizer_q2(hidden)
        q1_fused = q1_out + 0.1 * q2_out
        q2_fused = q2_out + 0.1 * q1_out
        gate_final = torch.sigmoid(self.quillan_comm_gate(torch.cat([q1_fused, q2_fused], dim=-1)))
        fused = F.rms_norm(hidden + (gate_final * q1_fused + (1.0 - gate_final) * q2_fused), (hidden.size(-1),), eps=1e-5)
        self._last_hidden = hidden.detach()
        self._last_fused = fused.detach()"""

    s_repl = """        # Law 5: Dual Quillan Finalizer Consensus (Everything reports back to main Quillan Core before output)
        if self.training:
            q1_out = self.quillan_finalizer_q1(hidden)
            q2_out = self.quillan_finalizer_q2(hidden)
            q1_fused = q1_out + 0.1 * q2_out
            q2_fused = q2_out + 0.1 * q1_out
            gate_final = torch.sigmoid(self.quillan_comm_gate(torch.cat([q1_fused, q2_fused], dim=-1)))
            fused = F.rms_norm(hidden + (gate_final * q1_fused + (1.0 - gate_final) * q2_fused), (hidden.size(-1),), eps=1e-5)
        else:
            fused = hidden
        self._last_hidden = hidden.detach()
        self._last_fused = fused.detach()"""
    assert s_target in content, "Failed to find finalizer block"
    content = content.replace(s_target, s_repl, 1)

    with open(oni_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("quillan_v5_4_oni.py patched successfully!", flush=True)

if __name__ == "__main__":
    patch_oni()

