#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for QuillanCrossModelWeightAdapter (2026 Sovereign Release)
"""

import unittest
import torch
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[3]
ONI_DIR = REPO_DIR / "09 - Projects" / "projects" / "oni"
sys.path.insert(0, str(ONI_DIR))
sys.path.insert(0, str(REPO_DIR))

from quillan_weight_adapter import QuillanCrossModelWeightAdapter
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

class TestQuillanWeightAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = QuillanCrossModelWeightAdapter(
            target_hidden_dim=128,
            target_ffn_dim=256,
            target_vocab_size=1000,
            num_experts=8,
            expert_rank=4,
            ternarize=True
        )

    def test_shape_projection_and_slicing(self):
        """Test adapting larger foreign tensor (e.g. 512x256) to target shape (128x128)."""
        large_tensor = torch.randn(512, 256)
        adapted = self.adapter.adapt_tensor_shape(large_tensor, (128, 128))
        self.assertEqual(adapted.shape, (128, 128))

    def test_synthetic_llama_to_quillan_adaptation(self):
        """Simulate a foreign Llama-style state dict and verify adaptation to Quillan."""
        src_hidden = 256
        src_ffn = 512
        src_vocab = 800
        
        synthetic_foreign_sd = {
            "model.embed_tokens.weight": torch.randn(src_vocab, src_hidden),
            "model.norm.weight": torch.randn(src_hidden),
            "model.layers.0.input_layernorm.weight": torch.randn(src_hidden),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(src_hidden),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(src_hidden, src_hidden),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(src_hidden, src_hidden),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(src_hidden, src_hidden),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(src_hidden, src_hidden),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(src_ffn, src_hidden),
            "model.layers.0.mlp.up_proj.weight": torch.randn(src_ffn, src_hidden),
            "model.layers.0.mlp.down_proj.weight": torch.randn(src_hidden, src_ffn),
        }

        adapted_sd = self.adapter.adapt_state_dict(synthetic_foreign_sd, source_arch="llama")
        
        # Verify mapped keys
        self.assertIn("wte.weight", adapted_sd)
        self.assertIn("ln_f.weight", adapted_sd)
        self.assertIn("h.0.ln_1.weight", adapted_sd)
        self.assertIn("h.0.attn.c_attn.weight", adapted_sd)
        self.assertIn("h.0.moe.c_fc.weight", adapted_sd)
        self.assertIn("h.0.moe.c_proj.weight", adapted_sd)
        
        # Verify target dimensions
        self.assertEqual(adapted_sd["wte.weight"].shape, (1000, 128))
        self.assertEqual(adapted_sd["h.0.attn.c_attn.weight"].shape, (384, 128))
        self.assertEqual(adapted_sd["h.0.moe.c_fc.weight"].shape, (512, 128))
        
        # Verify BitNet ternarization bounds on projected weights
        ternary_weights = adapted_sd["h.0.moe.c_proj.weight"]
        unique_vals = set(torch.unique(ternary_weights).tolist())
        self.assertTrue(unique_vals.issubset({-1.0, 0.0, 1.0}))

    def test_load_into_quillan_model(self):
        """Verify adapted weights can be loaded into QuillanRoninOni without errors."""
        cfg = QuillanOniConfig(
            vocab_size=1000,
            hidden_dim=128,
            ffn_dim=256,
            n_layer=1,
            n_head=4,
            head_dim=32,
            num_experts=8,
            expert_rank=4
        )
        model = QuillanRoninOni(cfg)
        
        synthetic_sd = {
            "model.embed_tokens.weight": torch.randn(800, 256),
            "model.norm.weight": torch.randn(256),
            "model.layers.0.input_layernorm.weight": torch.randn(256),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(256, 256),
        }
        adapted_sd = self.adapter.adapt_state_dict(synthetic_sd)
        
        missing, unexpected = model.load_state_dict(adapted_sd, strict=False)
        self.assertIn("wte.weight", adapted_sd)
        self.assertEqual(len(unexpected), 0)

if __name__ == "__main__":
    unittest.main()
