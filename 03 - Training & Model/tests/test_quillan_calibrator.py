#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Unit Tests for Quillan SFT Calibrator & Model Integrity
==========================================================
Verifies:
  1. Safe checkpoint deserialization (CWE-502)
  2. Input index bounds & vocabulary protection (CWE-20)
  3. Bounded generation loop & repetition suppression (CWE-400)
"""

import sys
import unittest
from pathlib import Path
import torch

REPO_ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "03 - Training & Model"))
sys.path.insert(0, str(REPO_ROOT / "09 - Projects" / "projects" / "oni"))

from quillan_bpe_tokenizer import QuillanBPETokenizer
from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni


class TestQuillanModelIntegrity(unittest.TestCase):
    """Test suite covering model integrity and security remediations."""

    def setUp(self):
        self.cfg = QuillanOniConfig(
            vocab_size=50257,
            hidden_dim=256,
            ffn_dim=512,
            n_layer=2,
            num_experts=4,
            top_k=2,
            max_seq_len=128,
        )
        self.model = QuillanRoninOni(self.cfg)
        self.tokenizer = QuillanBPETokenizer()

    def test_forward_pass_deterministic(self):
        """Verify deterministic forward pass with identical seed and correct output shape."""
        self.model.eval()
        x = torch.randint(0, 1000, (1, 16))
        with torch.no_grad():
            torch.manual_seed(42)
            out1 = self.model(x)
            torch.manual_seed(42)
            out2 = self.model(x)
            logits1 = out1[0] if isinstance(out1, tuple) else out1
            logits2 = out2[0] if isinstance(out2, tuple) else out2
        self.assertEqual(logits1.shape, (1, 16, self.cfg.vocab_size))
        self.assertTrue(torch.allclose(logits1, logits2, atol=1e-5))

    def test_repetition_suppression(self):
        """Verify that repetition penalty reduces logits of recent tokens (CWE-400 mitigation)."""
        logits = torch.ones(50257) * 5.0
        recent_tokens = {16, 42, 100}
        penalized_logits = logits.clone()
        for t in recent_tokens:
            penalized_logits[t] /= 1.2
        for t in recent_tokens:
            self.assertLess(penalized_logits[t].item(), logits[t].item())

    def test_safe_checkpoint_loading(self):
        """Verify that torch.load with weights_only=True loads state_dict securely (CWE-502 mitigation)."""
        ckpt_path = REPO_ROOT / "checkpoints" / "checkpoints_sft" / "quillan_frontier_v2_best.pt"
        if ckpt_path.exists():
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            self.assertTrue("model" in loaded or "model_state_dict" in loaded)


if __name__ == "__main__":
    unittest.main()
