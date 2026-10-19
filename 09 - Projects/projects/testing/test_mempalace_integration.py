#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and regression tests for MemPalace & Free Programming Books integration.
"""

import sys
import unittest
from pathlib import Path

# Add mempalace skill directory to sys.path
skill_dir = Path(r"C:\02_QUILLAN\03_Skills\mcp\mempalace")
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

from mempalace_bridge import MempalaceBridge, PalaceDrawer
from mempalace.mine_programming_books import (
    slugify_room,
    sanitize_url,
    parse_markdown_resource_list,
)


class TestMemPalaceIntegration(unittest.TestCase):
    """Test suite covering MemPalace bridge, miner, and URL sanitization."""

    def setUp(self):
        self.bridge = MempalaceBridge()

    def test_slugify_room(self):
        """Verify room slug generation handles special characters and casing."""
        self.assertEqual(slugify_room("Python (3.x)"), "python_3x")
        self.assertEqual(slugify_room("C++ Programming"), "c_programming")
        self.assertEqual(slugify_room("Algorithms & Data Structures"), "algorithms_data_structures")
        self.assertEqual(slugify_room(""), "general_technical")

    def test_sanitize_url_security(self):
        """Fixed vulnerability test: verify unsafe URL schemes are strictly rejected (CWE-20)."""
        self.assertIsNotNone(sanitize_url("https://github.com/leeex1/free-programming-books"))
        self.assertIsNotNone(sanitize_url("http://python.org/doc"))
        # Unsafe schemes
        self.assertIsNone(sanitize_url("javascript:alert(1)"))
        self.assertIsNone(sanitize_url("data:text/html;base64,PHNjcmlwdD4="))
        self.assertIsNone(sanitize_url("file:///etc/passwd"))
        self.assertIsNone(sanitize_url("vbscript:msgbox"))

    def test_bridge_lifecycle_and_search(self):
        """Core functionality: verify drawer filing and semantic retrieval."""
        test_content = "[BOOK] High Performance Python: Practical Performant Programming"
        drawer_id = self.bridge.add_drawer(
            content=test_content,
            wing="technical",
            room="python",
            metadata={"title": "High Performance Python", "author": "Micha Gorelick"}
        )
        self.assertTrue(drawer_id.startswith("drawer_"))

        results = self.bridge.search("Performant Python", wing="technical", room="python")
        self.assertGreaterEqual(len(results), 1)
        found = any("High Performance Python" in r.content for r in results)
        self.assertTrue(found)

    def test_edge_case_room_isolation(self):
        """Critical edge case: room filtering strictly isolates non-matching rooms."""
        self.bridge.add_drawer(
            content="[ISOLATION] Unique Haskell Monads Reference",
            wing="technical",
            room="haskell",
        )
        # Search specifically within the 'rust' room
        results = self.bridge.search("Haskell Monads", wing="technical", room="rust")
        self.assertFalse(any("Unique Haskell Monads Reference" in r.content for r in results))


if __name__ == "__main__":
    unittest.main()
