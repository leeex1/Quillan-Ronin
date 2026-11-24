import unittest
import os
import sqlite3
import time
import sys
import random # Added for unique IDs
import pickle # For checking embeddings
import numpy as np # For mock embeddings and potential checks
from unittest.mock import patch # For mocking SentenceTransformer

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.dialogue_manager import DialogueManager, MAX_HISTORY_SIZE

# Define a path for the test database
TEST_DB_PATH = os.path.join(project_root, 'data', 'test_dialogue_manager_memory.db')

# Mock SentenceTransformer to avoid actual model loading/downloading in tests
MOCK_EMBEDDING_VECTOR = np.array([0.1, 0.2, 0.3], dtype=np.float32) # Ensure dtype for consistency

class MockSentenceTransformer:
    def __init__(self, model_name_or_path):
        self.model_name = model_name_or_path
        print(f"MockSentenceTransformer initialized with model: {model_name_or_path}")

    def encode(self, sentences, convert_to_tensor=False, normalize_embeddings=False):
        print(f"MockSentenceTransformer.encode called for: {sentences}")
        if isinstance(sentences, str):
            return MOCK_EMBEDDING_VECTOR
        elif isinstance(sentences, list):
            return np.array([MOCK_EMBEDDING_VECTOR for _ in sentences])
        return np.array([])

# Removed class-level patching as it caused issues with unittest discovery/loader
# when the target module ('core.dialogue_manager.SentenceTransformer') might not exist
# if sentence_transformers library itself is not installed.
# Mocking will be done per-test or by direct assignment if needed.

class TestDialogueManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure the data directory exists
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Initial cleanup of test DB if it exists from a failed previous run
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def setUp(self):
        # Remove any existing test database before each test to ensure isolation
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

        self.dialogue_manager = DialogueManager(db_path=TEST_DB_PATH)
        # In this environment, self.dialogue_manager.embedding_model will be None
        # because sentence_transformers is not installed (due to disk space).
        # Tests need to account for this.

        if self.dialogue_manager.embedding_model is not None:
            print("WARNING (setUp): DialogueManager embedding_model is NOT None. This test environment might have sentence-transformers installed.")


        self.test_session_id = f"test_session_{int(time.time() * 1000)}_{random.randint(0, 1000)}"
        self.test_user_id = f"test_user_{int(time.time() * 1000)}_{random.randint(0, 1000)}"
        # Add a small delay to ensure unique session IDs if tests run very fast
        time.sleep(0.001)


    def tearDown(self):
        # Clean up the test database after each test
        if os.path.exists(TEST_DB_PATH):
            # Close any connection held by the dialogue_manager instance if it has one.
            # Current DM implementation opens/closes connections per method, so direct cleanup is fine.
            os.remove(TEST_DB_PATH)

    def test_01_initialization_creates_db_and_tables(self):
        """Test if __init__ creates the DB and all tables including conversation_embeddings."""
        self.assertTrue(os.path.exists(TEST_DB_PATH))
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()

        tables_to_check = ["conversations", "user_preferences", "conversation_embeddings"]
        for table_name in tables_to_check:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
            self.assertIsNotNone(cursor.fetchone(), f"{table_name} table not created.")

        # Check for index on conversation_embeddings
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conv_embeddings_session_id';")
        self.assertIsNotNone(cursor.fetchone(), "Index idx_conv_embeddings_session_id not created.")

        conn.close()

    def test_02_add_to_history_and_get_history(self):
        self.dialogue_manager.add_to_history("user", "Hello")
        self.dialogue_manager.add_to_history("assistant", "Hi there")
        history = self.dialogue_manager.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["text"], "Hello")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["text"], "Hi there")

    def test_03_history_trimming(self):
        for i in range(MAX_HISTORY_SIZE + 5):
            self.dialogue_manager.add_to_history("user", f"Message {i}")
        history = self.dialogue_manager.get_history()
        self.assertEqual(len(history), MAX_HISTORY_SIZE)
        self.assertEqual(history[0]["text"], "Message 5") # Check if the oldest are trimmed

    def test_04_store_and_get_conversation_history(self):
        self.dialogue_manager.store_conversation(self.test_session_id, "User query 1", "AI response 1")
        self.dialogue_manager.store_conversation(self.test_session_id, "User query 2", "AI response 2")

        retrieved_history = self.dialogue_manager.get_conversation_history(self.test_session_id)
        self.assertEqual(len(retrieved_history), 2)
        self.assertEqual(retrieved_history[0]["user_message"], "User query 1")
        self.assertEqual(retrieved_history[1]["ai_response"], "AI response 2")

    def test_05_get_conversation_history_limit(self):
        for i in range(15):
            self.dialogue_manager.store_conversation(self.test_session_id, f"U{i}", f"A{i}")

        retrieved_history = self.dialogue_manager.get_conversation_history(self.test_session_id, limit=5)
        self.assertEqual(len(retrieved_history), 5)
        self.assertEqual(retrieved_history[0]["user_message"], "U10") # Ordered by timestamp DESC, then reversed
        self.assertEqual(retrieved_history[4]["user_message"], "U14")

    def test_06_save_and_get_user_preference(self):
        self.dialogue_manager.save_user_preference(self.test_user_id, "language", "fr")
        pref = self.dialogue_manager.get_user_preference(self.test_user_id, "language")
        self.assertEqual(pref, "fr")

    def test_07_update_user_preference(self):
        self.dialogue_manager.save_user_preference(self.test_user_id, "language", "fr")
        self.dialogue_manager.save_user_preference(self.test_user_id, "language", "en") # Update
        pref = self.dialogue_manager.get_user_preference(self.test_user_id, "language")
        self.assertEqual(pref, "en")

    def test_08_get_non_existent_user_preference(self):
        pref = self.dialogue_manager.get_user_preference(self.test_user_id, "non_existent_key")
        self.assertIsNone(pref)

    def test_09_get_response_updates_history_and_stores_conversation(self):
        user_input = "Tell me a joke."
        initial_short_term_len = len(self.dialogue_manager.get_history())

        ai_response = self.dialogue_manager.get_response(user_input, session_id=self.test_session_id, user_id=self.test_user_id)

        # Check short-term history
        current_short_term_len = len(self.dialogue_manager.get_history())
        self.assertEqual(current_short_term_len, initial_short_term_len + 2) # user and assistant message
        last_messages = self.dialogue_manager.get_history()[-2:]
        self.assertEqual(last_messages[0]["role"], "user")
        self.assertEqual(last_messages[0]["text"], user_input)
        self.assertEqual(last_messages[1]["role"], "assistant")
        self.assertEqual(last_messages[1]["text"], ai_response)

        # Check long-term storage
        stored_conv = self.dialogue_manager.get_conversation_history(self.test_session_id, limit=1)
        self.assertEqual(len(stored_conv), 1)
        self.assertEqual(stored_conv[0]["user_message"], user_input)
        self.assertEqual(stored_conv[0]["ai_response"], ai_response)

    def test_10_get_response_uses_and_saves_preferences(self):
        user_id_pref_test = f"pref_user_{int(time.time() * 1000)}"
        # Ensure no preference initially
        self.assertIsNone(self.dialogue_manager.get_user_preference(user_id_pref_test, "language"))

        # First response, should save default "English"
        response1 = self.dialogue_manager.get_response("Hello", user_id=user_id_pref_test)
        self.assertIn("(Responding in your preferred language: English)", response1)
        self.assertEqual(self.dialogue_manager.get_user_preference(user_id_pref_test, "language"), "English")

        # Change preference
        self.dialogue_manager.save_user_preference(user_id_pref_test, "language", "Spanish")

        # Second response, should use "Spanish"
        response2 = self.dialogue_manager.get_response("Hola", user_id=user_id_pref_test) # Uses self.dialogue_manager
        self.assertIn("(Responding in your preferred language: Spanish)", response2)

    def test_11_store_conversation_stores_embedding_if_model_available(self):
        """Test that store_conversation attempts to store embedding if model is present (by mocking it)."""
        original_model = self.dialogue_manager.embedding_model
        self.dialogue_manager.embedding_model = MockSentenceTransformer(model_name_or_path='mocked-model-for-test11')
        self.assertIsNotNone(self.dialogue_manager.embedding_model, "Embedding model should be mocked for this test.")

        success = self.dialogue_manager.store_conversation(self.test_session_id, "User query with embedding", "AI response for embedding")
        self.assertTrue(success, "Storing conversation text failed.")

        # Verify embedding was stored
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        # Get the last inserted conversation id
        cursor.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        self.assertIsNotNone(row, "No conversation found to check embedding for.")
        last_conv_id = row[0]

        cursor.execute("SELECT embedding FROM conversation_embeddings WHERE conversation_id = ?", (last_conv_id,))
        embedding_row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(embedding_row, "No embedding found in conversation_embeddings table for the last conversation.")
        retrieved_embedding = pickle.loads(embedding_row[0])
        np.testing.assert_array_almost_equal(retrieved_embedding, MOCK_EMBEDDING_VECTOR, decimal=5)

    def test_12_retrieve_relevant_conversation_turns_fallback_no_model(self):
        """Test retrieve_relevant_conversation_turns falls back to recent history if no model."""
        # Simulate no embedding model
        original_model = self.dialogue_manager.embedding_model
        self.dialogue_manager.embedding_model = None

        self.dialogue_manager.store_conversation(self.test_session_id, "Q1", "A1")
        time.sleep(0.01) # ensure distinct timestamps if system is very fast
        self.dialogue_manager.store_conversation(self.test_session_id, "Q2", "A2")
        time.sleep(0.01)
        self.dialogue_manager.store_conversation(self.test_session_id, "Q3", "A3")

        relevant_turns = self.dialogue_manager.retrieve_relevant_conversation_turns(self.test_session_id, "query", top_k=2)
        self.assertEqual(len(relevant_turns), 2)
        self.assertEqual(relevant_turns[0]["user_message"], "Q2") # Fallback gets recent, which are Q2, Q3, then reversed.
        self.assertEqual(relevant_turns[1]["user_message"], "Q3") # So Q2 is older of the two, Q3 is newer.

        self.dialogue_manager.embedding_model = original_model # Restore

    @patch('core.dialogue_manager.DialogueManager._cosine_similarity')
    def test_13_retrieve_relevant_conversation_turns_with_mocked_semantic_search(self, mock_cosine_similarity):
        """Test retrieve_relevant_conversation_turns with semantic search (mocked model and similarity)."""
        original_model = self.dialogue_manager.embedding_model
        self.dialogue_manager.embedding_model = MockSentenceTransformer(model_name_or_path='mocked-model-for-test13')
        self.assertIsNotNone(self.dialogue_manager.embedding_model, "Embedding model should be explicitly mocked for this test.")

        # Store some conversations
        conv_ids = []
        conv_data = [
            ("Q1: topic apple", "A1: about fruit"),
            ("Q2: topic banana", "A2: also fruit"),
            ("Q3: topic carrot", "A3: a vegetable")
        ]
        for q, a in conv_data:
            self.dialogue_manager.store_conversation(self.test_session_id, q, a)
            conn = sqlite3.connect(TEST_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1")
            conv_ids.append(cursor.fetchone()[0])
            conn.close()
            time.sleep(0.01) # ensure distinct timestamps

        # Define mock cosine similarity behavior: make "carrot" most similar
        def mock_sim_func(query_emb, stored_emb):
            # This is highly simplified. A real test might need to inspect stored_emb
            # if it's tied to the content of the stored text.
            # For now, assume the last one (carrot) is most similar.
            # We need to associate similarity with conversation_id to make this robust.
            # The current implementation fetches all, then calculates.
            # Let's say query is "carrot"
            # We need to know which stored_emb corresponds to which text.
            # The DialogueManager fetches all, then calls _cosine_similarity for each.
            # We can make the mock return higher for the embedding that corresponds to "carrot".
            # This requires knowing which MOCK_EMBEDDING_VECTOR instance is which.
            # Since all mock embeddings are identical, we can't distinguish this way.
            # Instead, we'll rely on the order they are processed if the mock needs to be stateful,
            # or make the mock return values based on some property of stored_emb if it's not just a fixed vector.
            # For this test, let's assume the query "carrot" makes the "carrot" conversation most similar.
            # The mock `encode` always returns MOCK_EMBEDDING_VECTOR.
            # The simplest way is to have mock_cosine_similarity return values in a specific order
            # or based on some passed data if the real method did.
            # Given our mock always returns the same vector, this test becomes tricky
            # unless we mock the *result* of the DB query within retrieve_relevant_conversation_turns,
            # or make the mock_cosine_similarity stateful based on call count.

            # Let's simplify: assume the test query "carrot" will result in a specific call order.
            # The order of comparison in retrieve_relevant_conversation_turns is based on DB fetch (DESC timestamp).
            # So, "carrot" (newest) is processed first.
            if mock_cosine_similarity.call_count == 1: return 0.9 # Carrot (newest)
            if mock_cosine_similarity.call_count == 2: return 0.5 # Banana
            if mock_cosine_similarity.call_count == 3: return 0.1 # Apple
            return 0.0

        mock_cosine_similarity.side_effect = mock_sim_func

        relevant_turns = self.dialogue_manager.retrieve_relevant_conversation_turns(self.test_session_id, "carrot", top_k=2)
        self.assertEqual(len(relevant_turns), 2)
        self.assertEqual(relevant_turns[0]["user_message"], "Q3: topic carrot") # Most similar
        self.assertEqual(relevant_turns[1]["user_message"], "Q2: topic banana") # Second most
        self.assertAlmostEqual(relevant_turns[0]["similarity_score"], 0.9)


if __name__ == '__main__':
    unittest.main()
