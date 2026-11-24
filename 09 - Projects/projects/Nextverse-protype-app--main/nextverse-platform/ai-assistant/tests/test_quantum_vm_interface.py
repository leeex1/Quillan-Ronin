import unittest
import sys
import os

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"TestQuantumVMInterface: Added {project_root} to sys.path") # Removed print

from core.quantum_vm_interface import QuantumVMInterface

class TestQuantumVMInterface(unittest.TestCase):

    @classmethod # Can be classmethod if qvm_interface is stateless for tests
    def setUpClass(cls):
        # print("Setting up TestQuantumVMInterface class...") # Less verbose
        cls.qvm_interface = QuantumVMInterface()

    def test_01_semantic_disambiguation(self):
        data = {"text": "The fisherman went to the bank to get some cash.", "ambiguous_term": "bank"}
        result = self.qvm_interface.process_task("quantum_semantic_disambiguation", data)

        self.assertEqual(result["status"], "processed_by_quantum_placeholder")
        self.assertEqual(result["disambiguated_term"], "clarified_bank")
        self.assertIn(data["text"][:50], result["original_text_snippet"])
        self.assertEqual(result["confidence"], 0.88)
        self.assertIn("Disambiguated 'bank'", result["message"])

    def test_02_contextual_reasoning_step(self):
        data = {"context": "User is planning a trip to Mars.", "query": "What is the next logical step?"}
        result = self.qvm_interface.process_task("quantum_contextual_reasoning_step", data)

        self.assertEqual(result["status"], "processed_by_quantum_placeholder")
        self.assertIn("consider related_quantum_concept_X", result["next_reasoning_step"])
        self.assertIn(data["query"], result["next_reasoning_step"])
        self.assertEqual(result["confidence"], 0.92)
        self.assertEqual(result["original_query"], data["query"])

    def test_03_pattern_analysis_for_nlu(self):
        data = {"entities": [{"type": "PERSON", "value": "Alice"}, {"type": "EVENT", "value": "Conference"}], "text_snippet": "Alice attended the conference."}
        result = self.qvm_interface.process_task("quantum_pattern_analysis_for_nlu", data)

        self.assertEqual(result["status"], "processed_by_quantum_placeholder")
        self.assertIn("Complex relationship found among 2 entities", result["identified_pattern"])
        self.assertEqual(result["confidence"], 0.85)
        self.assertEqual(result["original_entities"], data["entities"])

    def test_04_ethical_consistency_check(self):
        data = {"response_candidate": "You should do this action.", "dess_score": 0.95, "ethical_vector": [0.1, 0.9, 0.2, 0.7]}
        result = self.qvm_interface.process_task("quantum_ethical_consistency_check", data)

        self.assertEqual(result["status"], "processed_by_quantum_placeholder")
        self.assertTrue(result["is_consistent_quantum_check"])
        self.assertIn("aligns with ethical principle Y", result["reason"])
        self.assertIn(str(data["dess_score"]), result["reason"])
        self.assertEqual(result["confidence"], 0.90)
        self.assertIn(data["response_candidate"][:50], result["response_snippet"])

    def test_05_unknown_task_type(self):
        data = {"info": "some custom data for an unknown task"}
        task_type = "my_experimental_quantum_flugelhorn_tuning"
        result = self.qvm_interface.process_task(task_type, data)

        self.assertEqual(result["status"], "error_unknown_task_type")
        self.assertEqual(result["task_type"], task_type)
        self.assertIn("Task type not specifically handled", result["message"])
        self.assertEqual(result["original_data"], data)

if __name__ == '__main__':
    unittest.main()
