import unittest
from unittest.mock import patch, AsyncMock  # AsyncMock for async methods
import asyncio
import sys
import os

# Adjust sys.path to allow importing quantum_master from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_master import QuantumMaster


class TestQuantumMaster(unittest.TestCase):
    def setUp(self):
        self.master = QuantumMaster(copies=3)  # Using 3 copies for manageable testing

    def test_merge_results_empty_input(self):
        results = []
        merged = self.master.merge_results(results)
        self.assertEqual(merged, {"status": "success", "total_boost": 0, "results": []})

    def test_merge_results_successful_chunks(self):
        results = [
            {"status": "success", "boost_applied": 10, "data": ["d1"]},
            {"status": "success", "boost_applied": 5, "data": ["d2", "d3"]},
        ]
        merged = self.master.merge_results(results)
        self.assertEqual(merged["total_boost"], 15)
        self.assertEqual(merged["results"], ["d1", "d2", "d3"])
        self.assertEqual(merged["status"], "success")

    def test_merge_results_mixed_chunks(self):
        results = [
            {"status": "success", "boost_applied": 10, "data": ["d1"]},
            {"status": "error", "message": "Failed chunk"},
            {"status": "success", "boost_applied": 20, "data": ["d2"]},
        ]
        merged = self.master.merge_results(results)
        self.assertEqual(merged["total_boost"], 30)
        self.assertEqual(merged["results"], ["d1", "d2"])
        self.assertEqual(merged["status"], "success")

    @patch("quantum_master.QuantumMaster.send_to_instance", new_callable=AsyncMock)
    def test_coordinate_task_data_chunking_and_calls(self, mock_send_to_instance):
        # Mock the return value of send_to_instance
        mock_send_to_instance.return_value = {
            "status": "success",
            "boost_applied": 1,
            "data": ["chunk_data"],
        }

        task_input_data = list(
            range(self.master.copies * 5)
        )  # e.g., 15 items for 3 copies
        task_def = {"data": task_input_data, "operation": "process"}

        # Python 3.14+ has no implicit "current event loop" in the main thread.
        # asyncio.run() owns its loop lifecycle and is the supported idiom.
        final_result = asyncio.run(self.master.coordinate_task(task_def))

        self.assertEqual(mock_send_to_instance.call_count, self.master.copies)

        expected_chunk_size = len(task_input_data) // self.master.copies
        for i, call_obj in enumerate(mock_send_to_instance.call_args_list):
            url_arg, data_payload_arg = call_obj[0]  # url, data
            self.assertEqual(url_arg, f"http://localhost:{7000 + i + 1}")
            self.assertEqual(data_payload_arg["chunk_id"], i)
            self.assertEqual(data_payload_arg["total_chunks"], self.master.copies)
            self.assertEqual(data_payload_arg["operation"], "process")
            self.assertEqual(len(data_payload_arg["data"]), expected_chunk_size)
            # Verify actual chunk content
            expected_chunk_content = task_input_data[
                i * expected_chunk_size : (i + 1) * expected_chunk_size
            ]
            self.assertEqual(data_payload_arg["data"], expected_chunk_content)

        # Verify merged result based on mocked send_to_instance returns
        self.assertEqual(final_result["total_boost"], self.master.copies * 1)
        self.assertEqual(
            len(final_result["results"]), self.master.copies * 1
        )  # Each mock returns one item list ['chunk_data']
        self.assertTrue(all(item == "chunk_data" for item in final_result["results"]))


if __name__ == "__main__":
    # This allows running tests directly from this file: python -m unittest path/to/test_quantum_master.py
    # However, creating the file via the tool and then running via a test runner is typical.
    unittest.main()
