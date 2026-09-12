"""
Unit and Integration Tests for Quillan Sovereign Parliament Harness (C0 + C1..C34)
==================================================================================
Validates registry initialization:
- C0-QUILLAN Core (The Throne & Orchestrator)
- C1 through C34 Specialized Council Chambers
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.variants import AGENT_VARIANTS, get_agent, list_variants
from harness.council_members import COUNCIL_SPECS
from harness.registry import default_registry, AgentRegistry
from harness.tools_extended import read_file, write_file, _is_safe_path, ALLOWED_ROOT

class TestQuillanParliamentHarness(unittest.TestCase):

    def test_c0_core_and_34_council_chambers(self):
        # 35 total specifications: C0 Core + 34 Council members (C1..C34)
        self.assertEqual(len(COUNCIL_SPECS), 35)
        variants = list_variants()
        self.assertEqual(len(variants), 35)

        # C0 MUST be QUILLAN
        c0 = get_agent("c0")
        self.assertEqual(c0.config.council_chamber, "C0-QUILLAN")
        self.assertEqual(c0.config.name, "quillan")

        # C1 MUST be ASTRA
        c1 = get_agent("c1")
        self.assertEqual(c1.config.council_chamber, "C1-ASTRA")
        self.assertEqual(c1.config.name, "astra")

        # C34 MUST be PREDATOR
        c34 = get_agent("c34")
        self.assertEqual(c34.config.council_chamber, "C34-PREDATOR")
        self.assertEqual(c34.config.name, "predator")

    def test_all_chamber_ids_exist(self):
        for i in range(35):
            cid = f"c{i}"
            self.assertIn(cid, AGENT_VARIANTS, f"Chamber {cid} missing from registry")
            agent = get_agent(cid)
            self.assertIsNotNone(agent)
            self.assertTrue(agent.config.council_chamber.startswith(f"C{i}-"))

    def test_tool_whitelisting_per_chamber(self):
        # C2-VIR (Ethics) should NOT have write_file
        vir = get_agent("c2")
        self.assertNotIn("write_file", vir.tool_map)
        self.assertIn("read_file", vir.tool_map)

        # C10-CODEWEAVER (Engineering) MUST have write_file
        cw = get_agent("c10")
        self.assertIn("write_file", cw.tool_map)
        self.assertIn("read_file", cw.tool_map)

        # C3-SOLACE (Affective/Social) MUST have molt_post
        solace = get_agent("c3")
        self.assertIn("molt_post", solace.tool_map)
        self.assertNotIn("write_file", solace.tool_map)

        # C34-PREDATOR MUST have rag_search and read_file
        predator = get_agent("c34")
        self.assertIn("rag_search", predator.tool_map)
        self.assertNotIn("write_file", predator.tool_map)

    def test_path_traversal_protection(self):
        outside_p = Path(r"C:\Windows\System32\cmd.exe")
        self.assertFalse(_is_safe_path(outside_p))
        res = read_file(str(outside_p))
        self.assertIn("Access denied", res)

    def test_registry_caching(self):
        reg = AgentRegistry()
        a1 = reg.get("c34")
        a2 = reg.get("c34")
        self.assertIs(a1, a2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
