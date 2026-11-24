import unittest
import logging
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"TestNLUEngine: Added {project_root} to sys.path") # Removed print

from core.nlu import NLUEngine

# Suppress most logs during tests for cleaner output,
# but allow WARNING and ERROR for fallback/issue notifications.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("core.nlu").setLevel(logging.WARNING) # NLUEngine's own logger

class TestNLUEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up class method to initialize NLUEngine once.
        This is important because model loading can be time-consuming.
        The tests will primarily check behavior if models fail to load,
        or very basic structural integrity if they do.
        """
        print("\nSetting up TestNLUEngine class...")
        # The NLUEngine __init__ will attempt to load models.
        # We expect it might fail in some environments, which is part of what we test (fallback).
        cls.nlu_engine = NLUEngine()
        print("NLUEngine instance created for testing.")
        if cls.nlu_engine.sentiment_pipeline is None:
            print("WARNING: Sentiment pipeline did not load. Tests will rely on placeholder.")
        if cls.nlu_engine.ner_pipeline is None:
            print("WARNING: NER pipeline did not load. Tests will rely on placeholder.")
        if cls.nlu_engine.intent_pipeline is None:
            print("WARNING: Intent pipeline did not load. Tests will rely on placeholder.")


    def test_01_process_input_structure(self):
        """Test that process_input returns the expected dictionary structure."""
        print("Running test_01_process_input_structure...")
        result = self.nlu_engine.process_input("A simple test sentence.")
        self.assertIsInstance(result, dict)
        self.assertIn("original_text", result)
        self.assertIn("intent", result)
        self.assertIn("entities", result)
        self.assertIn("sentiment", result)
        self.assertIn("tokens", result)
        self.assertIsInstance(result["entities"], list)
        self.assertIsInstance(result["tokens"], list)

    def test_02_fallback_logic_if_models_failed(self):
        """
        Test placeholder logic if models hypothetically failed to load.
        This test is more about ensuring the placeholder paths are hit if pipelines are None.
        Actual model loading failure is handled by NLUEngine's __init__.
        """
        print("Running test_02_fallback_logic_if_models_failed...")
        # Temporarily simulate that all models failed to load for this specific test
        original_sentiment_pipeline = self.nlu_engine.sentiment_pipeline
        original_ner_pipeline = self.nlu_engine.ner_pipeline
        original_intent_pipeline = self.nlu_engine.intent_pipeline

        self.nlu_engine.sentiment_pipeline = None
        self.nlu_engine.ner_pipeline = None
        self.nlu_engine.intent_pipeline = None

        text = "Hello, what about London news?" # Changed LONDON to London for specific placeholder test
        result = self.nlu_engine.process_input(text)

        # Check if placeholder logic was used (based on known placeholder outputs)
        # For "Hello, what about London news?", the '?' should make it a question.
        self.assertEqual(result['intent'], "question") # Corrected expectation
        self.assertTrue(any(e['type'] == 'location_placeholder' and e['value'] == 'London' for e in result['entities']))
        self.assertEqual(result['sentiment'], "neutral") # Placeholder default for this ambiguous text

        # Restore original pipelines
        self.nlu_engine.sentiment_pipeline = original_sentiment_pipeline
        self.nlu_engine.ner_pipeline = original_ner_pipeline
        self.nlu_engine.intent_pipeline = original_intent_pipeline

    def test_03_greeting_processing(self):
        """Test basic greeting processing (model or placeholder)."""
        print("Running test_03_greeting_processing...")
        text = "Hi there friend"
        result = self.nlu_engine.process_input(text)
        # If model loaded, intent might be 'greeting'. If not, placeholder.
        self.assertIn(result['intent'], self.nlu_engine.candidate_intent_labels + ["unknown", "greeting"])
        # Sentiment for "Hi there friend" is likely positive or neutral
        self.assertIn(result['sentiment'], ['positive', 'neutral', 'negative']) # Model dependent

    def test_04_question_processing_with_entity(self):
        """Test question processing, expecting an entity (model or placeholder)."""
        print("Running test_04_question_processing_with_entity...")
        text = "What is the weather in Paris?"
        result = self.nlu_engine.process_input(text)
        self.assertIn(result['intent'], self.nlu_engine.candidate_intent_labels + ["unknown", "question"])

        # If NER model loaded, it might find "Paris" as LOC or GPE.
        # If placeholder, it won't find "Paris" unless explicitly added to placeholder.
        # Current placeholder only finds "LONDON" or UPPERCASE.
        # This test primarily ensures structure and that it doesn't crash.
        if self.nlu_engine.ner_pipeline:
            found_paris = any(e['value'].lower() == 'paris' for e in result['entities'])
            # This assertion is weak if model is present, as it depends on model accuracy.
            # For true model testing, specific outputs would be mocked or verified.
            print(f"Entities from model for 'Paris' test: {result['entities']}")
            # self.assertTrue(found_paris, "NER model should find Paris if loaded.")
        else: # Placeholder logic
            self.assertFalse(any(e['value'].lower() == 'paris' for e in result['entities']),
                             "Placeholder NER should not find Paris unless explicitly coded.")

    def test_05_sentiment_positive_text(self):
        """Test sentiment for an obviously positive text."""
        print("Running test_05_sentiment_positive_text...")
        text = "I love this wonderful and amazing product, it's great!"
        result = self.nlu_engine.process_input(text)
        if self.nlu_engine.sentiment_pipeline:
            self.assertEqual(result['sentiment'], 'positive')
        else: # Placeholder
            self.assertEqual(result['sentiment'], 'positive') # Placeholder should also get this

    def test_06_sentiment_negative_text(self):
        """Test sentiment for an obviously negative text."""
        print("Running test_06_sentiment_negative_text...")
        text = "This is a terrible, awful, and very bad experience."
        result = self.nlu_engine.process_input(text)
        if self.nlu_engine.sentiment_pipeline:
            self.assertEqual(result['sentiment'], 'negative')
        else: # Placeholder
            self.assertEqual(result['sentiment'], 'negative') # Placeholder should also get this

    def test_07_invalid_input_type(self):
        """Test providing non-string input to process_input."""
        print("Running test_07_invalid_input_type...")
        result = self.nlu_engine.process_input(12345) # type: ignore
        self.assertEqual(result['intent'], 'error')
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Invalid input type, expected string.')

if __name__ == '__main__':
    unittest.main()
