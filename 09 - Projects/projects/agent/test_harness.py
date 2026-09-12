"""
Unit and Integration Tests for Quillan 10-Agent Sovereign Harness
================================================================
Validates registry initialization, role-based tool whitelisting,
security sandboxing, and execution.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.variants import AGENT_VARIANTS, get_agent, list_variants
from harness.registry import default_registry, AgentRegistry
from harness.tools_extended import read_file, write_file, _is_safe_path, ALLOWED_ROOT

class TestQuillanHarness(unittest.TestCase):

    def test_all_10_variants_registered(self):
        expected = {
            "architect", "coder", "security", "researcher", "moltbook",
            "browser", "rag", "audio", "creative", "governor"
        }
        self.assertEqual(set(AGENT_VARIANTS.keys()), expected)
        self.assertEqual(len(list_variants()), 10)

    def test_tool_whitelisting_per_variant(self):
        # Verify security agent does NOT have write_file or molt tools
        sec_agent = get_agent("security")
        self.assertNotIn("write_file", sec_agent.tool_map)
        self.assertNotIn("molt_post", sec_agent.tool_map)
        self.assertIn("read_file", sec_agent.tool_map)

        # Verify coder agent HAS write_file
        coder_agent = get_agent("coder")
        self.assertIn("write_file", coder_agent.tool_map)
        self.assertNotIn("molt_post", coder_agent.tool_map)

        # Verify moltbook agent HAS molt_post but NOT write_file
        molt_agent = get_agent("moltbook")
        self.assertIn("molt_post", molt_agent.tool_map)
        self.assertNotIn("write_file", molt_agent.tool_map)

    def test_unauthorized_tool_execution_blocked(self):
        sec_agent = get_agent("security")
        res = sec_agent.execute_tool("write_file", ["test.txt", "content"])
        self.assertIn("not authorized", res)

    def test_path_traversal_protection(self):
        # Outside path
        outside_p = Path(r"C:\Windows\System32\cmd.exe")
        self.assertFalse(_is_safe_path(outside_p))
        res = read_file(str(outside_p))
        self.assertIn("Access denied", res)

        # Relative escaping path
        escape_p = ALLOWED_ROOT / ".." / "Windows"
        self.assertFalse(_is_safe_path(escape_p))

    def test_registry_caching(self):
        reg = AgentRegistry()
        a1 = reg.get("coder")
        a2 = reg.get("coder")
        self.assertIs(a1, a2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
