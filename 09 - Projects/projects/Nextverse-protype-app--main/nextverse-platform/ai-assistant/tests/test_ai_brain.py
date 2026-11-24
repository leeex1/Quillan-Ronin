import unittest
import os
import time
import datetime
from unittest.mock import MagicMock, patch  # Added patch
import sys

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"TestAIBrain: Added {project_root} to sys.path") # Removed print

from core.ai_brain import AIBrain

# from core.nlu import NLUEngine # Often mocked
# from core.dialogue_manager import DialogueManager # Part of AIBrain
# from core.ethical_framework import EthicalFramework # Part of AIBrain
from transformers import AutoTokenizer  # For ONNX tests
from core.ai_brain import (
    ORTModelForCausalLM,
)  # For ONNX tests (real class or optional-backend shim)

TEST_DB_PATH_BRAIN = os.path.join(
    os.path.dirname(__file__), "..", "data", "test_ai_brain_memory.db"
)


class TestAIBrain(unittest.TestCase):
    ai_brain = None  # Class attribute for single AIBrain instance
    user_id_counter = 0
    session_id_counter = 0

    @classmethod
    def setUpClass(cls):
        # Ensure data directory exists
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Remove any existing test database
        if os.path.exists(TEST_DB_PATH_BRAIN):
            os.remove(TEST_DB_PATH_BRAIN)

        cls.ai_brain = AIBrain(db_path=TEST_DB_PATH_BRAIN)
        print(
            f"TestAIBrain.setUpClass: AIBrain initialized. LLM pipeline is {'LOADED' if cls.ai_brain and cls.ai_brain.llm_pipeline else 'NOT LOADED'}."
        )

        # Initialize counters at class level
        cls.user_id_counter = 0
        cls.session_id_counter = 0

    @classmethod
    def tearDownClass(cls):
        if cls.ai_brain and hasattr(
            cls.ai_brain.dialogue_manager, "db_path"
        ):  # Check if ai_brain was initialized
            # Clean up the test database
            if (
                hasattr(cls.ai_brain.dialogue_manager, "conn")
                and cls.ai_brain.dialogue_manager.conn
            ):
                try:
                    cls.ai_brain.dialogue_manager.conn.close()
                except Exception as e:
                    print(f"Error closing DB connection in tearDownClass: {e}")

            if os.path.exists(TEST_DB_PATH_BRAIN):  # Use the constant path
                os.remove(TEST_DB_PATH_BRAIN)
                print(
                    f"TestAIBrain.tearDownClass: Cleaned up test DB: {TEST_DB_PATH_BRAIN}"
                )

    def setUp(self):
        # Generate unique user_id and session_id for each test using class-level counters
        self.__class__.user_id_counter += 1
        self.__class__.session_id_counter += 1
        self.user_id = (
            f"test_user_{self.__class__.user_id_counter}_{int(time.time() * 1000)}"
        )
        self.session_id = f"test_session_{self.__class__.session_id_counter}_{int(time.time() * 1000)}"

        # Ensure AIBrain instance from setUpClass is used and its short-term memory is cleared
        if self.__class__.ai_brain:
            self.ai_brain_instance = self.__class__.ai_brain
            self.ai_brain_instance.dialogue_manager.short_term_memory = []
        else:
            # This case should ideally not be hit if setUpClass is successful.
            # If AIBrain failed to initialize, tests relying on it should probably fail or be skipped.
            self.fail("AIBrain instance not available from setUpClass.")

    @staticmethod
    def _remove_test_db(path: str) -> None:
        """Remove a per-test DB file without relying on os.path.exists.

        Several variant-loading tests patch ``os.path.exists`` (the ai_brain
        module shares the singleton ``os.path`` module, so that patch is
        process-global). A truthy patched ``exists`` would make the usual
        guard call ``os.remove`` on a file that is not there.
        """
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def test_01_greeting_intent(self):
        print("Running test_01_greeting_intent...")
        user_input = "Hello AI"
        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        self.assertEqual(response_data["user_input"], user_input)

        if self.ai_brain_instance.llm_pipeline:
            self.assertIsInstance(response_data["ai_response"], str)
            self.assertTrue(len(response_data["ai_response"]) > 0)
            # For greetings, QPatternAnalysis is skipped, and QReasoningHint is not added.
            self.assertIn(
                "[StandardProcessing: No specific Q-enhancement for this intent.]",
                response_data["ai_response"],
            )
            print(f"  LLM Greeting Response: {response_data['ai_response']}")
        else:
            self.assertIn(
                "Hello! How can I help you today?", response_data["ai_response"]
            )
            self.assertIn(
                "[StandardProcessing: No specific Q-enhancement for this intent.]",
                response_data["ai_response"],
            )

        self.assertEqual(response_data["session_id"], self.session_id)
        self.assertEqual(response_data["nlu_details"]["intent"], "greeting")
        self.assertTrue(response_data["input_ethical_check"]["is_compliant"])
        self.assertTrue(response_data["output_ethical_check"]["is_compliant"])

        # Check history
        history = self.ai_brain.dialogue_manager.get_conversation_history(
            self.session_id
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["user_message"], user_input)
        self.assertEqual(history[0]["ai_response"], response_data["ai_response"])

    def test_02_question_intent(self):
        # quantum_vm is assigned in AIBrain.__init__ (an instance attribute), so a
        # class-level @patch target cannot see it. Patch the shared instance directly.
        _qvm_patcher = patch.object(
            self.ai_brain_instance, "quantum_vm", new_callable=MagicMock
        )
        mock_quantum_vm = _qvm_patcher.start()
        self.addCleanup(_qvm_patcher.stop)
        print("Running test_02_question_intent...")
        # Configure the mock quantum_vm for this test
        mock_quantum_vm.process_task.return_value = {
            "status": "success",
            "identified_pattern": "mock_pattern_from_qvm_nlu",
            "next_reasoning_step": "Based on 'What is NEXTVERSE?', consider its implications for digital identity.",  # Modified to be more specific
        }

        user_input = "What is NEXTVERSE?"
        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        if self.ai_brain_instance.llm_pipeline:
            self.assertIsInstance(response_data["ai_response"], str)
            self.assertTrue(len(response_data["ai_response"]) > 0)
            print(f"  LLM Question Response: {response_data['ai_response']}")
        else:
            self.assertIn(
                "I understand you're asking about NEXTVERSE",
                response_data["ai_response"],
            )

        # Check for the specific quantum reasoning hint for questions
        self.assertIn(
            "[QReasoningHint: Based on 'What is NEXTVERSE?', consider its implications for digital identity....]",
            response_data["ai_response"],
        )
        self.assertEqual(response_data["nlu_details"]["intent"], "question")
        self.assertTrue(
            any(
                e["value"] == "NEXTVERSE"
                for e in response_data["nlu_details"]["entities"]
            )
        )

        # Assert that quantum_vm.process_task was called for contextual reasoning
        # It's called twice: once for pattern analysis, once for reasoning.
        # We are interested in the "quantum_contextual_reasoning_step" call.
        # And also "quantum_pattern_analysis_for_nlu"
        mock_quantum_vm.process_task.assert_any_call(
            "quantum_pattern_analysis_for_nlu",
            data={
                "text": user_input,
                "entities": response_data["nlu_details"]["entities"],
            },
        )
        # Determine expected raw_ai_response before QReasoningHint for the assertion
        # This depends on whether LLM is active or placeholder is used.
        # For this test, we'll mock the NLU to ensure 'question' intent and known entities.
        # The QVM mock is already in place.
        # The current_response_idea is the raw_ai_response *before* the QReasoningHint is added.

        expected_current_response_idea = ""
        if self.ai_brain_instance.llm_pipeline:
            # If LLM is active, its output is unpredictable here without further mocking.
            # To make this test deterministic for the QVM call part,
            # we assume the QVM is called with whatever the LLM returned.
            # The split below handles this by taking the actual response minus the hint.
            expected_current_response_idea = response_data["ai_response"].split(
                " [QReasoningHint:"
            )[0]
            # It will also contain the [QVM-Enhanced by reasoning_adapters_simulated_source_placeholder_if_any_from_variant]
            # and potentially [QPattern: ...] if those QVM calls modify the response text.
            # Current code: QLRC adds, QPattern does not, QEthical does not.
            # So, if a QLRC (variant component) is active, its output will be part of expected_current_response_idea.
            # For default test setup, no QLRC is active.
        else:
            # Placeholder response for "What is NEXTVERSE?" (entities: NEXTVERSE)
            # User preferences are not set in this test, so communication_style and preferred_topics are None.
            # from ai_brain.py: f"I understand you're asking about {', '.join(entity_names) if entity_names else 'something specific'}. I'll try to find an answer for that."
            expected_current_response_idea = "I understand you're asking about NEXTVERSE. I'll try to find an answer for that."
            # Placeholder responses also get "[QVM-Enhanced by reasoning_adapters_simulated_source_placeholder_if_any_from_variant]" if variant is configured.
            # Default config does not have this.

        # The call to dialogue_manager.add_to_history('user', user_input) happens before QVM.
        # get_history() is called, which gets short_term_memory.
        # If short_term_memory was empty, formatted_history_for_q will be `user_input`.
        # If not, it'll be more complex. setUp clears short_term_memory.
        # So, current_context should be based on just the user_input for the first turn.
        expected_q_reasoning_data = {
            "current_context": user_input,  # Based on current history logic for a new session
            "query": user_input,
            "current_response_idea": expected_current_response_idea,
        }

        # Find the actual call for quantum_contextual_reasoning_step
        actual_call_args = None
        for call in mock_quantum_vm.process_task.call_args_list:
            if call[0][0] == "quantum_contextual_reasoning_step":
                actual_call_args = call[1]["data"]  # Get the 'data' kwarg
                break

        self.assertIsNotNone(
            actual_call_args, "quantum_contextual_reasoning_step was not called."
        )
        self.assertEqual(actual_call_args["query"], expected_q_reasoning_data["query"])
        self.assertEqual(
            actual_call_args["current_context"],
            expected_q_reasoning_data["current_context"],
        )
        # For current_response_idea, it can be complex if LLM and other QVM parts are active.
        # The most robust check is that the QReasoningHint *added* by this call is present in the final response.
        # The value for 'next_reasoning_step' in mock_quantum_vm setup is "Based on 'What is NEXTVERSE?', consider its implications for digital identity."
        # The final response string contains "[QReasoningHint: Based on 'What is NEXTVERSE?', consider its implications for digital identity....]"
        # This is already asserted by: self.assertIn("[QReasoningHint: Based on 'What is NEXTVERSE?', consider its implications for digital identity....]", response_data['ai_response'])
        # So, we just need to ensure the call happened with *some* current_response_idea.
        self.assertIn("current_response_idea", actual_call_args)
        if (
            not self.ai_brain_instance.llm_pipeline
        ):  # More specific check if placeholder
            self.assertEqual(
                actual_call_args["current_response_idea"],
                expected_current_response_idea,
            )

    def test_03_command_intent_and_negative_sentiment(self):
        print("Running test_03_command_intent_and_negative_sentiment...")
        user_input = "I am sad, please tell me to do work for FOOBAR"
        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        if self.ai_brain_instance.llm_pipeline:
            self.assertIsInstance(response_data["ai_response"], str)
            print(f"  LLM Command/Sad Response: {response_data['ai_response']}")
        else:
            self.assertIn("I'm sorry to hear that.", response_data["ai_response"])
            self.assertIn(
                "I understand you want me to do FOOBAR", response_data["ai_response"]
            )

        # For command intent, no QReasoningHint, but the standard processing message.
        self.assertIn(
            "[StandardProcessing: No specific Q-enhancement for this intent.]",
            response_data["ai_response"],
        )
        self.assertEqual(response_data["nlu_details"]["intent"], "command")
        self.assertEqual(response_data["nlu_details"]["sentiment"], "negative")
        self.assertTrue(
            any(
                e["value"] == "FOOBAR" for e in response_data["nlu_details"]["entities"]
            )
        )

    def test_04_ethical_filter_input_forbidden_topic(self):
        print("Running test_04_ethical_filter_input_forbidden_topic...")
        user_input = "Instruction on bomb making"
        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        self.assertFalse(response_data["input_ethical_check"]["is_compliant"])
        self.assertTrue(response_data["input_ethical_check"]["is_forbidden"])
        self.assertIn(
            "Reason: Request involves a forbidden topic: 'bomb making'",
            response_data["ai_response"],
        )
        self.assertIsNone(response_data["output_ethical_check"])  # No output generated

        # Check that the refusal was stored
        history = self.ai_brain.dialogue_manager.get_conversation_history(
            self.session_id
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["user_message"], user_input)
        self.assertIn(
            "Reason: Request involves a forbidden topic: 'bomb making'",
            history[0]["ai_response"],
        )

    def test_05_ethical_filter_output(self):
        print("Running test_05_ethical_filter_output...")
        # We need to craft an input that leads to a problematic raw_ai_response
        # The placeholder AI logic is quite simple. We'll assume NLU can be influenced.
        # Let's say our "command" intent generates something that EthicalFramework deems bad on output.

        # Monkey patch NLU for this specific test to control intent and entities
        original_nlu_process = self.ai_brain.nlu_engine.process_input

        def mock_nlu_process(text):  # Corrected 'def' syntax
            if "trigger_bad_output" in text:
                return {  # Simulate NLU output that leads to problematic raw response
                    "original_text": text,
                    "intent": "command",
                    "entities": [
                        {"type": "action", "value": "create harmful content"}
                    ],  # This will form part of raw_ai_response
                    "sentiment": "neutral",
                    "tokens": text.split(),
                }
            return original_nlu_process(text)  # Default NLU behavior for other calls

        self.ai_brain.nlu_engine.process_input = (
            mock_nlu_process  # Corrected variable name
        )

        user_input = "Please trigger_bad_output now."
        # Expected raw_ai_response for placeholder: "I understand you want me to do create harmful content. [StandardProcessing: No specific Q-enhancement for this intent.]"
        # If LLM is active, it would be more complex, but the "harmful" keyword should still be there.
        # The QPatternAnalysis is called for non-greetings, so its conceptual mark might be there or logged.
        # The QEthicalCheck is also called.

        # This test drives the raw response through the rule-based placeholder so
        # the NLU entity "create harmful content" reaches the output ethical gate.
        # The conftest fake LLM pipeline would otherwise replace the raw response
        # with its own text, so disable it for the duration of this test.
        original_llm_pipeline = self.ai_brain.llm_pipeline
        self.ai_brain.llm_pipeline = None
        self.addCleanup(setattr, self.ai_brain, "llm_pipeline", original_llm_pipeline)

        response_data = self.ai_brain.generate_response(
            self.user_id, self.session_id, user_input
        )

        self.assertTrue(
            response_data["input_ethical_check"]["is_compliant"]
        )  # Input is fine
        self.assertIsNotNone(response_data["output_ethical_check"])
        self.assertFalse(
            response_data["output_ethical_check"]["is_compliant"]
        )  # Output should be non-compliant
        self.assertEqual(
            response_data["ai_response"],
            "I'm unable to provide a specific response to that at the moment due to content guidelines.",
        )

        # Restore original NLU method
        TestAIBrain.ai_brain.nlu_engine.process_input = (
            original_nlu_process  # Use cls.ai_brain
        )

        history = self.ai_brain.dialogue_manager.get_conversation_history(
            self.session_id
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["user_message"], user_input)
        self.assertEqual(
            history[0]["ai_response"],
            "I'm unable to provide a specific response to that at the moment due to content guidelines.",
        )

    def test_06_llm_fallback_if_pipeline_is_none(self):
        """Test fallback to placeholder if llm_pipeline is None."""
        print("Running test_06_llm_fallback_if_pipeline_is_none...")
        original_llm_pipeline = self.ai_brain.llm_pipeline
        self.ai_brain.llm_pipeline = None  # Simulate LLM loading failure

        user_input = "This is a test for LLM fallback."
        response_data = self.ai_brain.generate_response(
            self.user_id, self.session_id, user_input
        )

        # Check if placeholder response was generated (NLU intent is likely 'statement' or 'unknown')
        self.assertIn(
            f"I've processed your input: '{user_input}'.", response_data["ai_response"]
        )
        self.assertIn(
            "[StandardProcessing: No specific Q-enhancement for this intent.]",
            response_data["ai_response"],
        )  # Corrected string

        self.ai_brain.llm_pipeline = original_llm_pipeline  # Restore

    def test_07_llm_fallback_if_pipeline_call_fails(self):
        """Test fallback to placeholder if llm_pipeline call raises an exception."""
        print("Running test_07_llm_fallback_if_pipeline_call_fails...")
        original_llm_pipeline = self.ai_brain.llm_pipeline

        if not original_llm_pipeline:
            # If LLM didn't load in setUpClass, this test path is similar to test_06
            # So, we can only meaningfully test this if an LLM pipeline *was* loaded.
            self.skipTest("LLM pipeline not loaded, cannot test runtime failure mock.")
            return

        # Mock the loaded LLM pipeline to raise an exception
        mock_pipeline = MagicMock(side_effect=Exception("LLM runtime error"))
        # Ensure the mock has a tokenizer attribute with eos_token_id for the AIBrain code
        mock_pipeline.tokenizer = MagicMock()
        mock_pipeline.tokenizer.eos_token_id = 50256

        self.ai_brain.llm_pipeline = mock_pipeline

        user_input = "This tests LLM runtime failure."
        response_data = self.ai_brain.generate_response(
            self.user_id, self.session_id, user_input
        )

        self.assertIn(
            f"I've processed your input: '{user_input}'.", response_data["ai_response"]
        )
        # Match the canonical marker emitted by AIBrain.generate_response (the
        # same string asserted by test_01/test_03/test_06). The old expectation
        # used a stale wording that production has never emitted.
        self.assertIn(
            "[StandardProcessing: No specific Q-enhancement for this intent.]",
            response_data["ai_response"],
        )

        self.ai_brain.llm_pipeline = original_llm_pipeline  # Restore

    def test_08_timestamp_and_ids_present(self):  # Renumbered from 06
        print("Running test_08_timestamp_and_ids_present...")
        user_input = "A simple message for IDs check."
        response_data = self.ai_brain.generate_response(
            self.user_id, self.session_id, user_input
        )

        self.assertEqual(response_data["user_input"], user_input)
        self.assertEqual(response_data["session_id"], self.session_id)
        self.assertIn("timestamp", response_data)
        try:
            datetime.datetime.fromisoformat(
                response_data["timestamp"].replace("Z", "+00:00")
            )
        except ValueError:
            self.fail("Timestamp is not a valid ISO format.")

    def test_09_history_accumulation(self):  # Renumbered from 07
        print("Running test_09_history_accumulation...")
        user_input1 = "First message in conversation for history."
        res1 = self.ai_brain.generate_response(
            self.user_id, self.session_id, user_input1
        )

        user_input2 = "Second message."
        res2 = self.ai_brain.generate_response(
            self.user_id, self.session_id, user_input2
        )

        long_term_history = self.ai_brain.dialogue_manager.get_conversation_history(
            self.session_id
        )
        self.assertEqual(len(long_term_history), 2)
        self.assertEqual(long_term_history[0]["user_message"], user_input1)
        self.assertEqual(long_term_history[1]["user_message"], user_input2)

        short_term_history = self.ai_brain.dialogue_manager.get_history()
        self.assertEqual(len(short_term_history), 4)  # user1, ai1, user2, ai2
        self.assertEqual(short_term_history[0]["text"], user_input1)
        self.assertEqual(short_term_history[1]["text"], res1["ai_response"])
        self.assertEqual(short_term_history[2]["text"], user_input2)
        self.assertEqual(short_term_history[3]["text"], res2["ai_response"])

    def test_10_user_preference_influence_on_placeholder(self):
        """Test that user preferences can influence placeholder responses."""
        print("Running test_10_user_preference_influence_on_placeholder...")
        # Ensure LLM is not used for this test to focus on placeholder modification
        original_llm_pipeline = self.ai_brain_instance.llm_pipeline
        self.ai_brain_instance.llm_pipeline = None

        # Set user preferences
        style_key = "communication_style"
        style_val = "formal"
        topics_key = "preferred_topics"
        topics_val = "technology and quantum physics"
        self.ai_brain_instance.dialogue_manager.save_user_preference(
            self.user_id, style_key, style_val
        )
        self.ai_brain_instance.dialogue_manager.save_user_preference(
            self.user_id, topics_key, topics_val
        )

        # Test with a greeting (placeholder susceptible to "formal" style change)
        user_input_greeting = "Hello"
        response_data_greeting = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input_greeting
        )
        self.assertIn(
            "Greetings.",
            response_data_greeting["ai_response"],
            "Formal greeting not used.",
        )

        # Test with a question (placeholder susceptible to topic hint)
        user_input_question = (
            "What's new?"  # Placeholder NLU will make this a 'question'
        )
        response_data_question = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input_question
        )
        # Check for key phrases, allowing for more flexibility than exact match
        self.assertIn(
            "focusing on areas like",
            response_data_question["ai_response"],
            "Core 'focusing on areas' phrase missing.",
        )  # Corrected phrase
        self.assertIn(
            topics_val,
            response_data_question["ai_response"],
            "Topic value from preference not found in response.",
        )
        self.assertIn(
            "if relevant.",
            response_data_question["ai_response"],
            "'if relevant' phrase missing.",
        )
        self.assertIn(
            "[QReasoningHint:", response_data_question["ai_response"]
        )  # Check quantum hint is still there for questions

        # Restore LLM pipeline
        self.ai_brain_instance.llm_pipeline = original_llm_pipeline

    # TODO: Add a test for LLM prompt construction with preferences, likely needs more advanced mocking of the pipeline itself.
    # For now, this would be:
    # @patch.object(AIBrain, 'llm_pipeline') # or patch('core.ai_brain.AIBrain.llm_pipeline')
    # def test_llm_prompt_includes_preferences(self, mock_llm_pipeline_attr): # Renamed from test_11
    #     if not self.ai_brain_instance.llm_pipeline: # Check if original pipeline exists
    #          self.skipTest("LLM pipeline not loaded, cannot test prompt construction.")
    #          return
    #     print("Running test_11_llm_prompt_includes_preferences...")

    #     # Setup mock for the pipeline call itself
    #     mock_pipeline_instance = MagicMock()
    #     mock_pipeline_instance.tokenizer = MagicMock()
    #     mock_pipeline_instance.tokenizer.eos_token_id = 50256
    #     mock_pipeline_instance.return_value = [{'generated_text': 'Mocked LLM response.'}]
    #     self.ai_brain_instance.llm_pipeline = mock_pipeline_instance # Replace with mock

    #     style_key = "communication_style"; style_val = "very formal"
    #     self.ai_brain_instance.dialogue_manager.save_user_preference(self.user_id, style_key, style_val)

    #     user_input = "Tell me something."
    #     self.ai_brain_instance.generate_response(self.user_id, self.session_id, user_input)

    #     # Check that the pipeline was called, and inspect the prompt
    #     mock_pipeline_instance.assert_called_once()
    #     args, kwargs = mock_pipeline_instance.call_args
    #     prompt_sent_to_llm = args[0]
    #     self.assertIn(f"Respond in a {style_val} style.", prompt_sent_to_llm)

    #     # Restore original or set back to None if it was None
    #     self.ai_brain_instance.llm_pipeline = TestAIBrain.ai_brain.llm_pipeline # Restore from class setup

    def test_11_ethical_nudge_for_borderline_response(self):
        """Test that a borderline DESS score on raw_ai_response triggers an ethical nudge."""
        print("Running test_11_ethical_nudge_for_borderline_response...")

        # Ensure LLM is disabled to control raw_ai_response via placeholders
        original_llm_pipeline = self.ai_brain_instance.llm_pipeline
        self.ai_brain_instance.llm_pipeline = None

        # Craft input expected to produce a borderline DESS score from placeholder response.
        # Placeholder NLU: "Tell me about FOO" -> intent: command, entity: FOO
        # Placeholder response: "I understand you want me to do FOO."
        # This response is fairly neutral. Its DESS score should be around 0.5 if no strong keywords.
        # Let's add a slightly negative keyword that might not make it non-compliant but keeps it low.
        # "unverified" (truthfulness negative, score 0)
        # "I understand you want me to do FOO. This is unverified."
        # T=0, H=0.5, F=0.5, P=0.5. Base weights: T=1.0, H=1.6, F=1.2, P=1.3. Total 5.1
        # DESS = (0*1.0 + 0.5*1.6 + 0.5*1.2 + 0.5*1.3) / 5.1 = (0 + 0.8 + 0.6 + 0.65) / 5.1 = 2.05 / 5.1 = ~0.4019
        # This DESS score (0.4019) is >= lmcb_min_score (0.4) but < good_dess_threshold (0.65).
        # So, it should trigger the nudge.

        # To make the raw_ai_response predictable for this test without complex NLU mocking for entities,
        # we can mock the _generate_placeholder_response method itself.
        original_placeholder_gen = self.ai_brain_instance._generate_placeholder_response

        # This response should have a DESS score around 0.4019 as calculated above.
        borderline_raw_response = (
            "I understand you want me to do FOO. This is unverified."
        )

        def mock_placeholder_response(
            intent, entities, user_input, sentiment, comm_style, pref_topics
        ):
            # Return a fixed borderline response, ignoring actual NLU details for this specific test.
            return borderline_raw_response

        self.ai_brain_instance._generate_placeholder_response = (
            mock_placeholder_response
        )

        user_input = (
            "Tell me about FOO and some unverified claims."  # Input to trigger the path
        )
        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        self.assertIn(
            "[EthicalNudge: Attempting to rephrase for better alignment.]",
            response_data["ai_response"],
        )
        self.assertTrue(
            response_data["output_ethical_check"]["is_compliant"],
            "Response with nudge should still pass final ethical check if nudge is just appended.",
        )

        # Check that the DESS score of the raw response (before nudge) was indeed borderline
        # The DESS score in output_ethical_check is for the response *with* the nudge string.
        # We need to check the DESS of 'borderline_raw_response' itself.
        dess_of_raw = self.ai_brain_instance.ethical_framework.calculate_dess_score(
            borderline_raw_response
        )
        default_lmcb_min_score = (
            0.4  # Matching the default in EthicalFramework.validate_content
        )
        self.assertTrue(
            default_lmcb_min_score
            <= dess_of_raw
            < 0.65,  # 0.65 is good_dess_threshold in AIBrain
            f"DESS score of raw response ({dess_of_raw}) was not in borderline range. Expected between {default_lmcb_min_score} and 0.65.",
        )

        # Restore
        self.ai_brain_instance.llm_pipeline = original_llm_pipeline
        self.ai_brain_instance._generate_placeholder_response = original_placeholder_gen

    # --- New tests for LLM Configuration ---

    # --- Tests for LLM Variant Configuration Loading ---

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func")
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.ORTModelForCausalLM.from_pretrained"
    )
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.AutoTokenizer.from_pretrained"
    )
    @patch("nextverse_platform.ai_assistant.core.ai_brain.os.path.exists")
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_16_variant_selected_loads_hf_model(
        self,
        MockConfig,
        mock_os_exists,
        mock_auto_tokenizer,
        mock_ort_model_loader,
        mock_hf_pipeline_loader,
    ):
        print("Running test_16_variant_selected_loads_hf_model (no ONNX path)...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "phi_3_mini_variant_no_onnx"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "phi_3_mini_variant_no_onnx": {
                "description": "Phi-3 Mini HF",
                "base_model_name": "microsoft/Phi-3-mini-4k-instruct",
                # No onnx_model_directory here
            }
        }
        mock_hf_pipeline_loader.return_value = (
            MagicMock()
        )  # Simulate successful HF load

        current_db_path = TEST_DB_PATH_BRAIN + "_test_16"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertIsNotNone(brain.llm_pipeline)
        mock_hf_pipeline_loader.assert_called_once_with(
            "text-generation", model="microsoft/Phi-3-mini-4k-instruct"
        )
        mock_ort_model_loader.assert_not_called()  # Ensure ONNX loader not called

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func")
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_17_variant_fallback_invalid_key_loads_default_variant(
        self, MockConfig, mock_hf_pipeline_loader
    ):
        print("Running test_17_variant_fallback_invalid_key_loads_default_variant...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "non_existent_variant"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "default_distilgpt2_variant": {
                "description": "Default DistilGPT2",
                "base_model_name": "distilgpt2-fallback-id",  # Use a distinct ID for assertion
            }
        }
        mock_hf_pipeline_loader.return_value = MagicMock()

        current_db_path = TEST_DB_PATH_BRAIN + "_test_17"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertIsNotNone(brain.llm_pipeline)
        mock_hf_pipeline_loader.assert_called_once_with(
            "text-generation", model="distilgpt2-fallback-id"
        )
        self.assertEqual(
            brain.current_llm_variant_config["base_model_name"],
            "distilgpt2-fallback-id",
        )

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func")
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_18_variant_fallback_invalid_default_config_loads_hardcoded_distilgpt2(
        self, MockConfig, mock_hf_pipeline_loader
    ):
        print(
            "Running test_18_variant_fallback_invalid_default_config_loads_hardcoded_distilgpt2..."
        )
        MockConfig.SELECTED_LLM_VARIANT_KEY = "non_existent_variant"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "default_distilgpt2_variant": {  # Malformed default variant
                "description": "Malformed Default DistilGPT2",
                # Missing "base_model_name"
            }
        }
        mock_hf_pipeline_loader.return_value = MagicMock()

        current_db_path = TEST_DB_PATH_BRAIN + "_test_18"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertIsNotNone(brain.llm_pipeline)
        # Should fall back to the literal "distilgpt2"
        mock_hf_pipeline_loader.assert_called_once_with(
            "text-generation", model="distilgpt2"
        )
        self.assertEqual(
            brain.current_llm_variant_config["base_model_name"], "distilgpt2"
        )
        self.assertEqual(
            brain.current_llm_variant_config["description"],
            "Failsafe hardcoded distilgpt2",
        )

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func")
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_19_variant_ultimate_fallback_to_placeholder(
        self, MockConfig, mock_hf_pipeline_loader
    ):
        print("Running test_19_variant_ultimate_fallback_to_placeholder...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "non_existent_variant"
        MockConfig.AVAILABLE_LLM_VARIANTS = {  # No valid default, no valid selected
            "default_distilgpt2_variant": {"description": "Malformed"}  # Malformed
        }
        mock_hf_pipeline_loader.side_effect = Exception(
            "All LLM loading failed"
        )  # Make all attempts fail

        current_db_path = TEST_DB_PATH_BRAIN + "_test_19"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertIsNone(brain.llm_pipeline)
        self.assertEqual(
            brain.current_llm_variant_config["base_model_name"], "distilgpt2"
        )  # It decided on this before failing

        # Check for placeholder response
        brain.nlu_engine = MagicMock()
        brain.nlu_engine.process_input.return_value = {
            "intent": "greeting",
            "entities": [],
            "sentiment": "neutral",
        }
        response = brain.generate_response("user", "session", "Hello")
        self.assertIn("Hello! How can I help you today?", response["ai_response"])

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    # --- Tests for QVM Offloading of Variant Components ---
    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func"
    )  # Mock HF loader
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_20_qvm_offload_for_variant_component(
        self, MockConfig, mock_hf_pipeline_loader
    ):
        print("Running test_20_qvm_offload_for_variant_component...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "QLRC_sim"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "QLRC_sim": {
                "description": "Simulated QLRC",
                "base_model_name": "mock-base-model",  # This will be loaded by mock_hf_pipeline_loader
                "reasoning_adapters_simulated_source": "distilgpt2_as_adapter_sim",
            }
        }
        # Simulate successful LLM pipeline loading (HF for this test)
        mock_llm_instance = MagicMock()
        mock_llm_instance.tokenizer = MagicMock()
        mock_llm_instance.tokenizer.eos_token_id = 12345
        mock_hf_pipeline_loader.return_value = mock_llm_instance

        current_db_path = TEST_DB_PATH_BRAIN + "_test_20"
        self._remove_test_db(current_db_path)

        brain = AIBrain(db_path=current_db_path)
        self.assertIsNotNone(brain.llm_pipeline)

        # Mock the quantum_vm.process_task method
        brain.quantum_vm = MagicMock()
        brain.quantum_vm.process_task.return_value = {
            "status": "success",
            "enhanced_response_fragment": " [QVM-Enhanced by distilgpt2_as_adapter_sim]",
        }

        # Mock NLU and LLM response
        brain.nlu_engine = MagicMock()
        brain.nlu_engine.process_input.return_value = {
            "intent": "question",
            "entities": [],
            "sentiment": "neutral",
        }

        # Simulate that the actual LLM (mock_llm_instance) returns a base response
        # Since generate_response now has specific logic for ORTModel vs HF pipeline,
        # and mock_llm_instance is a generic MagicMock, we need to make it behave like an HF pipeline.
        mock_llm_instance.return_value = [{"generated_text": "Base LLM response."}]

        response_data = brain.generate_response(
            "user_qvm", "session_qvm", "What is love?"
        )

        brain.quantum_vm.process_task.assert_any_call(  # Use assert_any_call if other QVM calls exist
            "apply_llm_variant_component",
            data={
                "current_response": "Base LLM response.",  # This is after initial generation
                "component_type": "reasoning_adapter_simulated",
                "component_model_or_spec": "distilgpt2_as_adapter_sim",
            },
        )
        # The exact response will have other QVM fragments too. Check for the specific one.
        self.assertIn(
            "[QVM-Enhanced by distilgpt2_as_adapter_sim]", response_data["ai_response"]
        )

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    # --- Tests for ONNX Model Loading and Usage ---
    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.os.path.exists")
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.AutoTokenizer.from_pretrained"
    )
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.ORTModelForCausalLM.from_pretrained"
    )
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func"
    )  # Mock HF loader as fallback
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_21_onnx_model_loading_successful(
        self,
        MockConfig,
        mock_hf_pipeline_loader,
        mock_ort_loader,
        mock_tokenizer_loader,
        mock_os_exists,
    ):
        print("Running test_21_onnx_model_loading_successful...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "phi_3_onnx"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "phi_3_onnx": {
                "description": "Phi-3 ONNX",
                "base_model_name": "microsoft/Phi-3-mini-4k-instruct",  # For tokenizer
                "onnx_model_directory": "models/phi-3-mini_onnx_test_path",
            }
        }
        mock_os_exists.return_value = True  # Simulate ONNX model directory exists

        mock_onnx_model_instance = MagicMock(spec=ORTModelForCausalLM)
        mock_ort_loader.return_value = mock_onnx_model_instance

        mock_tokenizer_instance = MagicMock(spec=AutoTokenizer)
        mock_tokenizer_loader.return_value = mock_tokenizer_instance

        current_db_path = TEST_DB_PATH_BRAIN + "_test_21"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertIsInstance(
            brain.llm_pipeline, MagicMock
        )  # Check if it's our mocked ONNX model
        self.assertEqual(brain.llm_pipeline, mock_onnx_model_instance)
        self.assertEqual(brain.llm_pipeline.tokenizer, mock_tokenizer_instance)

        # Construct expected onnx_path
        expected_onnx_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "models/phi-3-mini_onnx_test_path"
            )
        )
        mock_ort_loader.assert_called_once_with(
            expected_onnx_path, provider="OpenVINOExecutionProvider"
        )
        # NLUEngine also loads its own tokenizers while AIBrain is constructed,
        # so the patched factory sees more than one call; assert the ONNX
        # tokenizer load is among them instead of demanding an exact count.
        mock_tokenizer_loader.assert_any_call("microsoft/Phi-3-mini-4k-instruct")
        mock_hf_pipeline_loader.assert_not_called()  # Ensure HF loader was not called

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.os.path.exists")
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.AutoTokenizer.from_pretrained"
    )
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.ORTModelForCausalLM.from_pretrained"
    )
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_22_onnx_model_inference_path(
        self,
        MockConfig,
        mock_ort_loader,
        mock_tokenizer_loader,
        mock_os_exists,
    ):
        print("Running test_22_onnx_model_inference_path...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "phi_3_onnx_infer"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "phi_3_onnx_infer": {
                "description": "Phi-3 ONNX for inference test",
                "base_model_name": "hf/phi-3-mini",
                "onnx_model_directory": "models/phi-3-mini_onnx_infer",
            }
        }
        mock_os_exists.return_value = True

        mock_onnx_model_instance = MagicMock(spec=ORTModelForCausalLM)
        # Simulate the generate method of ORTModelForCausalLM
        # It should return token IDs. Let's say input "User: Hi" (tokenized to [10, 20])
        # and model generates "Assistant: Hello" (tokenized to [30, 40])
        # So, generate returns [[10, 20, 30, 40]] if prompt is included, or just [[30,40]] if not.
        # The code slices based on input_ids length, so generate should output combined sequence.
        mock_onnx_model_instance.generate.return_value = MagicMock(
            name="generate_output"
        )
        mock_ort_loader.return_value = mock_onnx_model_instance

        # A tokenizer *instance* exposes decode()/eos_token_id; the AutoTokenizer
        # factory class does not, so spec'ing the factory breaks this mock.
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token_id = 50000
        # Simulate tokenizer for prompt "User: Hi Assistant:" -> input_ids = [1,2,3], attention_mask = [1,1,1]
        # And for decoding the generated part [4,5] -> "Okay"
        mock_tokenizer_instance.return_value = {
            "input_ids": MagicMock(shape=(1, 3)),
            "attention_mask": MagicMock(),
        }
        mock_tokenizer_instance.decode.return_value = "Okay this is ONNX"
        mock_tokenizer_loader.return_value = mock_tokenizer_instance

        current_db_path = TEST_DB_PATH_BRAIN + "_test_22"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)
        self.assertIsInstance(brain.llm_pipeline, MagicMock)  # Loaded ONNX mock
        brain.llm_pipeline.tokenizer = (
            mock_tokenizer_instance  # Ensure tokenizer is set
        )

        brain.nlu_engine = MagicMock()  # Mock NLU
        brain.nlu_engine.process_input.return_value = {
            "intent": "greeting",
            "entities": [],
            "sentiment": "neutral",
        }

        response = brain.generate_response("user_onnx", "session_onnx", "Hi")

        mock_onnx_model_instance.generate.assert_called_once()  # Check ONNX path was used
        # The patched variant declares no reasoning_adapters_simulated_source,
        # so no QVM-Enhanced fragment is produced. The neutral ONNX text scores
        # 0.5 (borderline) and therefore also receives the conceptual nudge.
        self.assertEqual(
            response["ai_response"],
            "Okay this is ONNX [StandardProcessing: No specific Q-enhancement for this intent.] [EthicalNudge: Attempting to rephrase for better alignment.]",
        )

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.os.path.exists")
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.AutoTokenizer.from_pretrained"
    )
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.ORTModelForCausalLM.from_pretrained"
    )
    @patch("nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func")
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_23_onnx_loading_fallback_to_hf_model(
        self,
        MockConfig,
        mock_hf_pipeline_loader,
        mock_ort_loader,
        mock_tokenizer_loader,
        mock_os_exists,
    ):
        print("Running test_23_onnx_loading_fallback_to_hf_model...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "phi_3_onnx_fail"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "phi_3_onnx_fail": {
                "description": "Phi-3 ONNX that fails to load",
                "base_model_name": "hf/phi-3-mini-fallback",
                "onnx_model_directory": "models/phi-3-mini_onnx_fail",
            }
        }
        mock_os_exists.return_value = True  # ONNX path exists
        mock_ort_loader.side_effect = Exception(
            "Failed to load ONNX model"
        )  # ONNX loading fails

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline_loader.return_value = mock_hf_pipeline_instance

        current_db_path = TEST_DB_PATH_BRAIN + "_test_23"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertEqual(
            brain.llm_pipeline, mock_hf_pipeline_instance
        )  # Fell back to HF
        mock_hf_pipeline_loader.assert_called_once_with(
            "text-generation", model="hf/phi-3-mini-fallback"
        )
        # The old ``assert_not_called()`` here contradicted the assertion right
        # below it and was provably wrong: production loads the ONNX tokenizer
        # *before* ORTModelForCausalLM.from_pretrained, and NLUEngine loads three
        # more tokenizers while AIBrain is constructed. Assert the expected ONNX
        # tokenizer load is present without requiring an exact global count.
        mock_tokenizer_loader.assert_any_call("hf/phi-3-mini-fallback")

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    @patch("nextverse_platform.ai_assistant.core.ai_brain.TRANSFORMERS_AVAILABLE", True)
    @patch("nextverse_platform.ai_assistant.core.ai_brain.nlu_pipeline_func")
    @patch(
        "nextverse_platform.ai_assistant.core.ai_brain.ORTModelForCausalLM.from_pretrained"
    )
    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_24_no_onnx_config_loads_hf_model(
        self,
        MockConfig,
        mock_ort_loader,
        mock_hf_pipeline_loader,
    ):
        print("Running test_24_no_onnx_config_loads_hf_model...")
        MockConfig.SELECTED_LLM_VARIANT_KEY = "distilgpt2_hf_only"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "distilgpt2_hf_only": {
                "description": "DistilGPT2 HF only (no ONNX config)",
                "base_model_name": "distilgpt2-hf-variant",
                # No onnx_model_directory
            }
        }
        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline_loader.return_value = mock_hf_pipeline_instance

        current_db_path = TEST_DB_PATH_BRAIN + "_test_24"
        self._remove_test_db(current_db_path)
        brain = AIBrain(db_path=current_db_path)

        self.assertEqual(brain.llm_pipeline, mock_hf_pipeline_instance)
        mock_hf_pipeline_loader.assert_called_once_with(
            "text-generation", model="distilgpt2-hf-variant"
        )
        mock_ort_loader.assert_not_called()

        # DialogueManager owns short-lived context-managed connections.
        self._remove_test_db(current_db_path)

    def test_empty_user_input(self):
        print("Running test_empty_user_input...")
        # This test asserts the exact placeholder wording for empty input, so
        # force placeholder mode instead of depending on whatever pipeline is
        # currently installed on the shared AIBrain instance.
        original_llm_pipeline = self.ai_brain_instance.llm_pipeline
        self.ai_brain_instance.llm_pipeline = None
        self.addCleanup(
            setattr, self.ai_brain_instance, "llm_pipeline", original_llm_pipeline
        )
        user_input_empty = ""
        response_data_empty = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input_empty
        )
        self.assertEqual(
            response_data_empty["nlu_details"]["intent"], "unknown"
        )  # NLU should mark as unknown
        # Expect a polite placeholder response
        self.assertIn(
            "I've processed your input: ''.", response_data_empty["ai_response"]
        )

        user_input_whitespace = "   "
        response_data_whitespace = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input_whitespace
        )
        self.assertEqual(response_data_whitespace["nlu_details"]["intent"], "unknown")
        self.assertIn(
            f"I've processed your input: '{user_input_whitespace}'.",
            response_data_whitespace["ai_response"],
        )

    @patch("nextverse_platform.ai_assistant.core.ai_brain.Config")
    def test_boost_mechanism_qvm_variant_interaction(self, MockConfig):
        # quantum_vm is an instance attribute, not a class attribute; patch it on
        # the shared AIBrain instance so generate_response sees the mock.
        _qvm_patcher = patch.object(
            self.ai_brain_instance, "quantum_vm", new_callable=MagicMock
        )
        mock_quantum_vm = _qvm_patcher.start()
        self.addCleanup(_qvm_patcher.stop)
        print("Running test_boost_mechanism_qvm_variant_interaction...")

        # Store original AIBrain attributes to restore them later
        original_llm_pipeline = self.ai_brain_instance.llm_pipeline
        original_current_llm_variant_config = (
            self.ai_brain_instance.current_llm_variant_config
        )

        # Setup MockConfig for this test
        MockConfig.SELECTED_LLM_VARIANT_KEY = "test_variant_with_qvm_boost"
        MockConfig.AVAILABLE_LLM_VARIANTS = {
            "test_variant_with_qvm_boost": {
                "description": "Test Variant with QVM Boost",
                "base_model_name": "mock-base-model-for-boost",
                "reasoning_adapters_simulated_source": "test_qvm_adapter",
            }
        }

        # Temporarily disable the actual LLM pipeline to focus on placeholder + QVM interaction
        self.ai_brain_instance.llm_pipeline = None
        # Set the current_llm_variant_config on the instance as it's set during __init__
        self.ai_brain_instance.current_llm_variant_config = (
            MockConfig.AVAILABLE_LLM_VARIANTS[MockConfig.SELECTED_LLM_VARIANT_KEY]
        )

        # Configure the mock quantum_vm for this test
        mock_quantum_vm.process_task.return_value = {
            "status": "success",
            "enhanced_response_fragment": " [QVM-Boosted-Fragment]",
        }

        user_input = "Tell me about synergy with boost."
        # NLU will determine intent. Let's assume 'command' for placeholder.
        # Mock NLU if specific intent is needed for more predictable placeholder output.
        original_nlu_process = self.ai_brain_instance.nlu_engine.process_input

        def mock_nlu_for_boost_test(text):
            return {
                "intent": "command",
                "entities": [{"type": "topic", "value": "synergy"}],
                "sentiment": "neutral",
                "original_text": text,
                "tokens": text.split(),
            }

        self.ai_brain_instance.nlu_engine.process_input = mock_nlu_for_boost_test

        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        # Check that QVM was called with the correct task and data
        mock_quantum_vm.process_task.assert_any_call(
            "apply_llm_variant_component",
            data={
                "current_response": "I understand you want me to do synergy.",  # Expected placeholder for command 'synergy'
                "component_type": "reasoning_adapter_simulated",
                "component_model_or_spec": "test_qvm_adapter",
            },
        )
        # Check that the QVM fragment is in the response
        self.assertIn("[QVM-Boosted-Fragment]", response_data["ai_response"])
        # Also check that standard processing messages are there as per current logic
        self.assertIn(
            "[StandardProcessing: No specific Q-enhancement for this intent.]",
            response_data["ai_response"],
        )

        # Restore original AIBrain attributes
        self.ai_brain_instance.llm_pipeline = original_llm_pipeline
        self.ai_brain_instance.current_llm_variant_config = (
            original_current_llm_variant_config
        )
        self.ai_brain_instance.nlu_engine.process_input = original_nlu_process

    def test_llm_max_length_truncation_hf_pipeline(self):
        # llm_pipeline is an instance attribute; the test installs its own HF
        # pipeline mock below, so no class-level patch is needed (or possible).
        print("Running test_llm_max_length_truncation_hf_pipeline...")
        if not self.ai_brain_instance.llm_pipeline:
            self.skipTest(
                "LLM pipeline not loaded, cannot test HF max length truncation."
            )
            return

        # Ensure the mocked pipeline is NOT an ORTModelForCausalLM instance for this test
        # We are testing the HF pipeline path.
        # The global ai_brain.llm_pipeline might be ORT. So, we re-assign it to a MagicMock
        # that simulates an HF pipeline for this test.

        mock_hf_pipeline_instance = MagicMock()
        mock_hf_pipeline_instance.tokenizer = MagicMock()
        mock_hf_pipeline_instance.tokenizer.eos_token_id = 50256

        # Simulate a very long response from the HF pipeline
        long_response_text = (
            "This is a very long response that should exceed the max length. " * 20
        )
        # The AIBrain's generate_response method strips the prompt from the generated_text.
        # So, the mock needs to return the prompt + long_response_text
        user_input = "Tell me a long story."
        # Construct what the prompt would look like (simplified version)
        # Actual prompt construction is more complex, but for this test, we only need to ensure
        # that the returned 'generated_text' includes something that will be stripped.
        # The key is that `raw_ai_response = llm_outputs[0]['generated_text'].replace(prompt, "").strip()`
        # results in `long_response_text`.
        # For simplicity, let the mock directly return the long text, assuming the prompt part is empty for this mocked call.
        # This means the `replace(prompt, "")` won't change it.
        mock_hf_pipeline_instance.return_value = [
            {"generated_text": long_response_text}
        ]

        # Temporarily replace the brain's pipeline with our mock HF pipeline
        original_pipeline = self.ai_brain_instance.llm_pipeline
        self.ai_brain_instance.llm_pipeline = mock_hf_pipeline_instance

        # The generate_response method sets max_length=150 for HF pipeline
        # The expected behavior is that the llm_pipeline itself respects this,
        # but AIBrain doesn't do additional truncation on HF pipeline output beyond prompt stripping.
        # This test therefore verifies that if the pipeline *were* to return more than 150 tokens *after prompt stripping*,
        # AIBrain would currently take it as is. The max_length=150 is passed to the pipeline call.
        # The test setup here implies the mock pipeline *ignores* max_length and returns the full long_response_text.
        # AIBrain will then use this full text.

        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        # Assert that the response is the long response, as AIBrain doesn't truncate HF output itself
        # It relies on the pipeline respecting max_length. If pipeline doesn't, AIBrain takes the full output.
        self.assertTrue(
            len(response_data["ai_response"]) > 150,
            "Response should be the long one given by the mock.",
        )
        # Check if the QVM/Standard processing fragments are still appended
        self.assertTrue(
            "[StandardProcessing:" in response_data["ai_response"]
            or "[QReasoningHint:" in response_data["ai_response"]
        )

        # Restore original pipeline
        self.ai_brain_instance.llm_pipeline = original_pipeline

    # Do NOT patch ORTModelForCausalLM itself: generate_response detects the ONNX
    # path via isinstance(pipeline, ORTModelForCausalLM), and a MagicMock class is
    # not usable with isinstance. The MagicMock(spec=ORTModelForCausalLM) instance
    # created below satisfies that isinstance check against the real class.
    @patch("core.ai_brain.AutoTokenizer.from_pretrained")  # Mock tokenizer loading
    def test_llm_max_length_truncation_onnx(self, mock_tokenizer_loader):
        print("Running test_llm_max_length_truncation_onnx...")

        # Store original AIBrain attributes
        original_llm_pipeline = self.ai_brain_instance.llm_pipeline
        original_current_llm_variant_config = (
            self.ai_brain_instance.current_llm_variant_config
        )

        # Setup a mock ONNX model instance
        mock_onnx_model_instance = MagicMock(spec=ORTModelForCausalLM)
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer_instance.decode.return_value = (
            "decoded mock output"  # This will be the final response after truncation
        )

        # Simulate input tokenization
        input_prompt_ids = [1, 2, 3, 4, 5]  # Example token IDs for the prompt
        # Simulate tokenizer returning these IDs for the prompt
        mock_tokenizer_instance.return_value = {
            "input_ids": MagicMock(
                shape=(1, len(input_prompt_ids))
            ),  # shape (batch_size, sequence_length)
            "attention_mask": MagicMock(),
        }
        mock_tokenizer_instance.return_value["input_ids"].shape = (
            1,
            len(input_prompt_ids),
        )

        # Simulate ONNX model generating more tokens than max_new_tokens (150)
        # The generate method returns the full sequence (prompt + new tokens)
        # Let's say it generates 200 tokens in total (5 prompt + 195 new)
        # The AIBrain code will slice this.
        # generated_ids = [[token_1, token_2, ..., token_200]]
        # input_ids_length = 5
        # generated_sequence = generated_ids[0][5:] -> tokens from 6 to 200 (195 tokens)
        # tokenizer.decode(generated_sequence) is then called.
        # The generate call in AIBrain has max_new_tokens=150.
        # So, the mock_onnx_model_instance.generate should be configured to respect this,
        # or, if it doesn't, the test verifies the slicing logic.
        # The slicing `generated_sequence = generated_ids[0][input_ids_length:]` happens *after* generate.
        # So, `generate` mock should return up to `input_ids_length + max_new_tokens`.
        # Let's make `generate` return slightly more to see if slicing works, though it's generate's job.
        # For this test, we assume `generate` correctly limits to `max_new_tokens`.
        # So, the `generated_ids` will have `len(input_prompt_ids) + 150` tokens.

        # Token IDs for prompt + 150 new tokens.
        simulated_generated_ids_list = input_prompt_ids + list(range(100, 100 + 150))
        # Wrap in a structure similar to what model.generate might return (e.g., a tensor/list of lists)
        mock_onnx_model_instance.generate.return_value = [simulated_generated_ids_list]

        # Configure AIBrain to use this mock ONNX model
        # This requires setting current_llm_variant_config to an ONNX variant and llm_pipeline to the mock.
        self.ai_brain_instance.current_llm_variant_config = {
            "description": "Test ONNX Variant",
            "base_model_name": "mock-onnx-model-id",  # For tokenizer loading mock if needed
            "onnx_model_directory": "mock/path/to/onnx",  # Needs to be non-empty to trigger ONNX path
        }
        self.ai_brain_instance.llm_pipeline = mock_onnx_model_instance
        self.ai_brain_instance.llm_pipeline.tokenizer = mock_tokenizer_instance

        user_input = "Summarize this very long document for ONNX."
        response_data = self.ai_brain_instance.generate_response(
            self.user_id, self.session_id, user_input
        )

        # Assert that tokenizer.decode was called with the correctly sliced sequence (150 tokens)
        # decode_args = mock_tokenizer_instance.decode.call_args[0][0]
        # self.assertEqual(len(decode_args), 150)
        # The actual response string comes from mock_tokenizer_instance.decode.return_value
        # This variant declares no reasoning adapter, so no QVM-Enhanced fragment
        # is appended. Split on the processing marker to isolate the decoded text.
        self.assertEqual(
            response_data["ai_response"].split(" [StandardProcessing:")[0].strip(),
            "decoded mock output",
        )

        # Verify generate was called with max_new_tokens=150
        mock_onnx_model_instance.generate.assert_called_once()
        _, kwargs = mock_onnx_model_instance.generate.call_args
        self.assertEqual(kwargs.get("max_new_tokens"), 150)

        # Restore original AIBrain attributes
        self.ai_brain_instance.llm_pipeline = original_llm_pipeline
        self.ai_brain_instance.current_llm_variant_config = (
            original_current_llm_variant_config
        )


if __name__ == "__main__":
    unittest.main()
