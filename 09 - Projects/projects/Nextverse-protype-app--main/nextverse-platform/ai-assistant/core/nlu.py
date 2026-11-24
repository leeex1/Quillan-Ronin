"""
Natural Language Understanding (NLU) module for the AI Assistant.

This module is responsible for processing raw text input and extracting
meaningful information such as intent, entities, and sentiment.
It attempts to use pre-trained Hugging Face Transformer models for these tasks,
with a fallback to simpler rule-based placeholder logic if models cannot be loaded
or if an error occurs during model-based processing.
"""
import re
import logging

# Attempt to import transformers components, but allow fallback if not installed
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Mock the pipeline function if transformers is not available, so NLUEngine can still be defined
    def pipeline(*args, **kwargs): # pylint: disable=unused-argument
        """Mock pipeline function for when transformers is not installed."""
        logging.warning("Transformers library not found. NLP models will not be loaded. Using placeholder logic.")
        return None

logger = logging.getLogger(__name__)

# Define model names for clarity and easy modification
MODEL_SENTIMENT = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_NER = "dbmdz/bert-large-cased-finetuned-conll03-english"
MODEL_ZERO_SHOT_INTENT = "facebook/bart-large-mnli"


class NLUEngine:
    """
    Performs NLU tasks: intent recognition, entity extraction, sentiment analysis.

    Attempts to use Hugging Face Transformers models. If model loading fails
    (e.g., due to network issues, resource limits, or missing models),
    it falls back to simpler rule-based placeholder logic for each respective task.
    """
    def __init__(self):
        """
        Initializes the NLUEngine by attempting to load pre-trained models.

        If model loading fails, placeholder logic will be used for the affected tasks.
        """
        logger.info("Initializing NLUEngine...")
        self.candidate_intent_labels = ["greeting", "question", "command", "farewell", "statement", "other"]
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        self.intent_pipeline = None

        if TRANSFORMERS_AVAILABLE:
            # Sentiment Analysis Pipeline
            try:
                logger.info(f"Loading sentiment analysis model: {MODEL_SENTIMENT}")
                self.sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_SENTIMENT)
                logger.info("Sentiment analysis model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load sentiment analysis model '{MODEL_SENTIMENT}'. Error: {e}. Falling back to placeholder logic for sentiment.", exc_info=True)
                self.sentiment_pipeline = None

            # NER Pipeline
            try:
                logger.info(f"Loading NER model: {MODEL_NER}")
                self.ner_pipeline = pipeline("ner", model=MODEL_NER, grouped_entities=True)
                logger.info("NER model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NER model '{MODEL_NER}'. Error: {e}. Falling back to placeholder logic for NER.", exc_info=True)
                self.ner_pipeline = None

            # Zero-shot Classification Pipeline for Intent
            try:
                logger.info(f"Loading zero-shot classification model for intent: {MODEL_ZERO_SHOT_INTENT}")
                self.intent_pipeline = pipeline("zero-shot-classification", model=MODEL_ZERO_SHOT_INTENT)
                logger.info("Zero-shot classification model for intent loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load zero-shot model '{MODEL_ZERO_SHOT_INTENT}'. Error: {e}. Falling back to placeholder logic for intent.", exc_info=True)
                self.intent_pipeline = None

            if not all([self.sentiment_pipeline, self.ner_pipeline, self.intent_pipeline]):
                logger.warning("One or more NLP models failed to load during initialization. NLUEngine will use placeholder logic for affected tasks.")
            else:
                logger.info("All NLP models loaded successfully for NLUEngine.")
        else:
            logger.warning("Transformers library not available. NLUEngine will use placeholder logic for all NLP tasks.")


    def process_input(self, text: str) -> dict:
        """
        Processes the input text to extract NLU features.

        Args:
            text: The raw user input string.

        Returns:
            A dictionary containing the NLU analysis results:
            - 'original_text': The input text.
            - 'intent': The recognized intent (e.g., 'greeting', 'question').
            - 'entities': A list of extracted entities (e.g., names, locations).
            - 'sentiment': The analyzed sentiment (e.g., 'positive', 'negative').
            - 'tokens': A list of tokens from the input text.
        """
        if not isinstance(text, str):
            # Basic type check for input
            return {
                "original_text": text,
                "intent": "error",
                "entities": [],
                "sentiment": "unknown",
                "tokens": [],
                "error": "Invalid input type, expected string."
            }

        text_lower = text.lower() # Keep for some placeholder logic if needed

        # --- Intent Classification ---
        if self.intent_pipeline:
            try:
                intent_result = self.intent_pipeline(text, self.candidate_intent_labels)
                intent = intent_result['labels'][0] # Highest scoring label
                logger.debug(f"Intent (model): '{intent}' from result: {intent_result}")
            except Exception as e:
                logger.error(f"Error during model-based intent classification: {e}. Falling back to placeholder.", exc_info=True)
                intent = self._placeholder_intent(text_lower, text) # Pass both for flexibility
        else: # Fallback to placeholder if pipeline is None
            intent = self._placeholder_intent(text_lower, text)
            logger.debug(f"Intent (placeholder): '{intent}'")

        # --- Entity Extraction (NER) ---
        if self.ner_pipeline:
            try:
                ner_results = self.ner_pipeline(text)
                entities = [{"type": entity['entity_group'], "value": entity['word'], "score": entity['score']} for entity in ner_results]
                logger.debug(f"Entities (model): {entities}")
            except Exception as e:
                logger.error(f"Error during model-based NER: {e}. Falling back to placeholder.", exc_info=True)
                entities = self._placeholder_entities(text, text_lower)
        else: # Fallback
            entities = self._placeholder_entities(text, text_lower)
            logger.debug(f"Entities (placeholder): {entities}")

        # --- Sentiment Analysis ---
        if self.sentiment_pipeline:
            try:
                sentiment_result = self.sentiment_pipeline(text)
                # Output is usually like [{'label': 'POSITIVE', 'score': 0.999...}]
                sentiment = sentiment_result[0]['label'].lower()
                logger.debug(f"Sentiment (model): '{sentiment}' from result: {sentiment_result}")
            except Exception as e:
                logger.error(f"Error during model-based sentiment analysis: {e}. Falling back to placeholder.", exc_info=True)
                sentiment = self._placeholder_sentiment(text_lower)
        else: # Fallback
            sentiment = self._placeholder_sentiment(text_lower)
            logger.debug(f"Sentiment (placeholder): '{sentiment}'")

        # --- Tokenization (using NLU model's tokenizer if available, else simple split) ---
        # This is a simplified tokenization for the output dictionary.
        # Actual models perform their own, more complex tokenization internally.
        tokens = text.lower().split() # Default simple tokenization
        # if self.ner_pipeline: # Or any other pipeline that has a tokenizer
        #     try:
        #         # Accessing the tokenizer from a pipeline can be tricky and isn't standard.
        #         # This is a conceptual placeholder. Usually, you'd use the tokenizer directly if loaded separately.
        #         # For simplicity, we'll stick to the .split() for the 'tokens' field in the output dict.
        #         # tokens = self.ner_pipeline.tokenizer.tokenize(text)
        #         pass
        #     except Exception as e:
        #         logger.warning(f"Could not use pipeline's tokenizer: {e}. Using simple split.", exc_info=True)
        #         tokens = text.lower().split()


        return {
            "original_text": text,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "tokens": tokens # Simple whitespace-based tokenization
        }

    # Placeholder methods (extracted for clarity)
    def _placeholder_intent(self, text_lower: str, original_text: str) -> str:
        intent = "unknown"
        # Use word boundaries or exact match for greetings to avoid partial matches like "this" containing "hi"
        # Tokenize for more robust greeting check
        tokens_for_intent = text_lower.split()
        if "hello" in tokens_for_intent or "hi" in tokens_for_intent:
            intent = "greeting"
        elif original_text.strip().endswith("?"): # Check original text for '?' for questions
            intent = "question"
        elif "tell me" in text_lower or "show me" in text_lower:
            intent = "command"
        elif "bye" in text_lower or "goodbye" in text_lower:
            intent = "farewell"
        elif len(text_lower.split()) > 3 : # Arbitrary: if longer than 3 words, assume statement
             intent = "statement"
        return intent

    def _placeholder_entities(self, original_text: str, text_lower: str) -> list:
        entities = []
        punctuation_to_strip = str.maketrans("", "", ",.?!;:")
        raw_tokens = original_text.split()
        for word_token in raw_tokens:
            cleaned_word = word_token.translate(punctuation_to_strip)
            if cleaned_word.isupper() and len(cleaned_word) > 1:
                entities.append({"type": "keyword_uppercase", "value": cleaned_word})
            elif "london" in word_token.lower():
                 entities.append({"type": "location_placeholder", "value": "London"})
        return entities

    def _placeholder_sentiment(self, text_lower: str) -> str:
        sentiment = "neutral"
        if any(kw in text_lower for kw in ["love", "happy", "great", "good", "excellent"]):
            sentiment = "positive"
        elif any(kw in text_lower for kw in ["terrible", "not happy", "bad", "sad", "hate", "awful"]):
            sentiment = "negative"
        return sentiment


# For basic testing and demonstration
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Enable debug logging for __main__

    logger.info("--- NLUEngine Demonstration ---")
    nlu_engine = NLUEngine() # This will attempt to download models

    test_inputs = [
        "Hello there, how are you?",
        "What is the weather like in LONDON today, my friend?",
        "I am very sad about the news from Paris.",
        "Tell me a joke about a cat and a dog.",
        "This is an amazing and wonderful experience!",
        "I hate this terrible, awful product.",
        "Goodbye and see you later."
    ]

    for i, text_input in enumerate(test_inputs):
        logger.info(f"\n--- Processing Input {i+1}: '{text_input}' ---")
        try:
            analysis_result = nlu_engine.process_input(text_input)
            logger.info(f"NLU Analysis Result for '{text_input}':")
            import json
            logger.info(json.dumps(analysis_result, indent=2))
        except Exception as e:
            logger.error(f"Error during NLU processing for '{text_input}': {e}", exc_info=True)

    logger.info("\n--- NLUEngine Demonstration Finished ---")
