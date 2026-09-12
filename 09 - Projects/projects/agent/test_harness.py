"""
Unit and Integration Tests for Quillan 34-Chamber Sovereign Council Harness
==========================================================================
Validates registry initialization across all 34 Council Chambers (C0-C33),
alias routing, role-based tool whitelisting, and security sandboxing.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.variants import AGENT_VARIANTS, get_agent, list_variants
from harness.council_members import COUNCIL_SPECS
from harness.registry import default_registry, AgentRegistry
from harness.tools_extended import read_file, write_file, _is_safe_path, ALLOWED_ROOT

class TestQuillanCouncilHarness(unittest.TestCase):

    def test_all_34_canonical_council_chambers_exist(self):
        self.assertEqual(len(COUNCIL_SPECS), 34)
        variants = list_variants()
        self.assertEqual(len(variants), 34)
        for i in range(34):
            cid = f"c{i}"
            self.assertIn(cid, AGENT_VARIANTS, f"Chamber {cid} missing from registry")
            agent = get_agent(cid)
            self.assertIsNotNone(agent)
            self.assertTrue(agent.config.council_chamber.startswith(f"C{i}-"))

    def test_persona_and_legacy_alias_resolution(self):
        # By persona name
        astra = get_agent("astra")
        self.assertEqual(astra.config.council_chamber, "C0-ASTRA")

        predator = get_agent("predator")
        self.assertEqual(predator.config.council_chamber, "C33-PREDATOR")

        # Legacy aliases
        coder = get_agent("coder")
        self.assertEqual(coder.config.council_chamber, "C9-CODEWEAVER")

        security = get_agent("security")
        self.assertEqual(security.config.council_chamber, "C12-WARDEN")

    def test_tool_whitelisting_per_chamber(self):
        # C1-VIR (Ethics) should NOT have write_file
        vir = get_agent("c1")
        self.assertNotIn("write_file", vir.tool_map)
        self.assertIn("read_file", vir.tool_map)

        # C9-CODEWEAVER (Engineering) MUST have write_file
        cw = get_agent("c9")
        self.assertIn("write_file", cw.tool_map)
        self.assertIn("read_file", cw.tool_map)

        # C2-SOLACE (Affective/Social) MUST have molt_post
        solace = get_agent("c2")
        self.assertIn("molt_post", solace.tool_map)
        self.assertNotIn("write_file", solace.tool_map)

    def test_unauthorized_tool_execution_blocked(self):
        logos = get_agent("c6")
        res = logos.execute_tool("write_file", ["test.txt", "payload"])
        self.assertIn("not authorized", res)

    def test_path_traversal_protection(self):
        outside_p = Path(r"C:\Windows\System32\cmd.exe")
        self.assertFalse(_is_safe_path(outside_p))
        res = read_file(str(outside_p))
        self.assertIn("Access denied", res)

    def test_registry_caching(self):
        reg = AgentRegistry()
        a1 = reg.get("c9")
        a2 = reg.get("c9")
        self.assertIs(a1, a2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
