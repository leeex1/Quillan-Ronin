"""
Dialogue Management System for the AI Assistant.

This module handles the conversation flow, maintains dialogue history (both short-term
and long-term using an SQLite database), and manages user preferences.
It provides functionalities to store and retrieve conversation turns and user-specific
settings, which are crucial for context-aware interactions.
"""
import sqlite3
import datetime
import os
import logging
import pickle # For storing/retrieving BLOB embeddings
import numpy as np # For cosine similarity calculation

# Attempt to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # This logger might not be configured if this module is imported before basicConfig is called elsewhere.
    # Consider configuring a default handler for the library if it's a standalone entry point.
    logging.getLogger(__name__).warning(
        "sentence_transformers library not found. Semantic search capabilities will be disabled."
    )


# Configure basic logging if not already configured by another module
# This ensures logger output if DialogueManager is used standalone or tested directly.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the default path for the database within the ai-assistant/data directory
DEFAULT_DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'long_term_memory.db')
MAX_HISTORY_SIZE = 20 # Max number of turns to keep in short-term memory
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

class DialogueManager:
    """
    Manages dialogue state, history, and user preferences.
    Includes semantic search capabilities for long-term memory if sentence-transformers is available.

    Attributes:
        short_term_memory (list): In-memory list of recent conversation turns.
        db_path (str): Path to the SQLite database file for long-term storage.
        embedding_model (SentenceTransformer | None): Model for generating text embeddings.
                                                    None if sentence-transformers is unavailable or model loading fails.
    """
    def __init__(self, db_path=DEFAULT_DATABASE_PATH):
        """
        Initializes the DialogueManager, including setting up the database
        and attempting to load the sentence embedding model.

        Args:
            db_path (str, optional): The path to the SQLite database file.
                Defaults to DEFAULT_DATABASE_PATH.
        """
        self.short_term_memory = []
        self.db_path = db_path
        self.embedding_model = None

        try:
            data_dir = os.path.dirname(self.db_path)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                logger.info(f"Created data directory: {data_dir}")
        except OSError as e:
            logger.error(f"Error creating database directory {os.path.dirname(self.db_path)}: {e}", exc_info=True)

        self._init_db() # Initializes DB tables including conversation_embeddings
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Attempts to load the sentence transformer model if the library is available."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Attempting to load sentence embedding model '{EMBEDDING_MODEL_NAME}'...")
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info(f"Sentence embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load sentence embedding model '{EMBEDDING_MODEL_NAME}'. Error: {e}. Semantic search will be degraded or disabled.", exc_info=True)
                self.embedding_model = None
        else:
            # This case is hit if the import of SentenceTransformer failed at the top.
            # The warning about the library missing is already issued there.
            logger.info("embedding_model remains None as sentence-transformers library is unavailable.")
            self.embedding_model = None


    def _init_db(self):
        """Initializes the SQLite database and creates tables if they don't exist."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create 'conversations' table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create 'user_preferences' table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    preference_key TEXT NOT NULL,
                    preference_value TEXT,
                    UNIQUE(user_id, preference_key)
                )
            ''')

            # Create 'conversation_embeddings' table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_embeddings (
                    conversation_id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            ''')
            # Add index for faster lookup by session_id in embeddings table
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conv_embeddings_session_id ON conversation_embeddings(session_id);
            ''')

            conn.commit()
            logger.info(f"Database tables (conversations, user_preferences, conversation_embeddings) ensured/created at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database error during _init_db at {self.db_path}: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()

    def add_to_history(self, role: str, text: str):
        """
        Adds a message to the short-term conversation history.

        The history is trimmed if it exceeds MAX_HISTORY_SIZE.

        Args:
            role (str): The role of the message sender ('user' or 'assistant').
            text (str): The content of the message.

        Raises:
            ValueError: If the role is not 'user' or 'assistant'.
        """
        if role not in ['user', 'assistant']:
            logger.warning(f"Attempted to add message with invalid role: {role}")
            raise ValueError("Role must be 'user' or 'assistant'")

        message = {
            "role": role,
            "text": text,
            "timestamp": datetime.datetime.now(datetime.timezone.utc) # Use timezone-aware UTC time
        }
        self.short_term_memory.append(message)

        # Trim history if it exceeds max size, keeping the most recent entries
        if len(self.short_term_memory) > MAX_HISTORY_SIZE:
            self.short_term_memory = self.short_term_memory[-MAX_HISTORY_SIZE:]
        logger.debug(f"Added to short-term history. Current size: {len(self.short_term_memory)}")


    def get_history(self) -> list:
        """
        Returns the current short-term conversation history.

        Returns:
            list: A list of message dictionaries.
                  Returns a copy to prevent external modification of the internal list.
        """
        return list(self.short_term_memory)

    def store_conversation(self, session_id: str, user_message: str, ai_response: str) -> bool:
        """
        Stores a full conversation turn (user message and AI response) in the database,
        and also generates and stores its embedding if the model is available.

        Args:
            session_id (str): The ID of the current conversation session.
            user_message (str): The user's message.
            ai_response (str): The AI's response.

        Returns:
            bool: True if storage of conversation text was successful, False otherwise.
                  Embedding storage failure is logged but doesn't make this method return False.
        """
        conn = None
        conversation_stored_successfully = False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Store conversation text
            cursor.execute(
                '''INSERT INTO conversations (session_id, user_message, ai_response) VALUES (?, ?, ?)''',
                (session_id, user_message, ai_response)
            )
            conversation_id = cursor.lastrowid # Get the ID of the inserted row
            conn.commit()

            if conversation_id is not None:
                logger.info(f"Conversation turn stored (ID: {conversation_id}) for session_id: {session_id}")
                conversation_stored_successfully = True

                # Generate and store embedding if model is available
                if self.embedding_model:
                    try:
                        combined_text = f"User: {user_message}\nAssistant: {ai_response}"
                        embedding_vector = self.embedding_model.encode(combined_text)
                        serialized_embedding = pickle.dumps(embedding_vector) # Serialize numpy array to BLOB

                        cursor.execute(
                            '''INSERT INTO conversation_embeddings (conversation_id, session_id, embedding, timestamp)
                               VALUES (?, ?, ?, ?)''',
                            (conversation_id, session_id, serialized_embedding, datetime.datetime.now(datetime.timezone.utc))
                        )
                        conn.commit()
                        logger.info(f"Embedding stored for conversation_id: {conversation_id}")
                    except Exception as e:
                        logger.error(f"Failed to generate or store embedding for conversation_id {conversation_id}: {e}", exc_info=True)
            else:
                logger.error(f"Failed to get conversation_id after insert for session_id: {session_id}")
                conversation_stored_successfully = False

        except sqlite3.Error as e:
            logger.error(f"Database error while storing conversation for session_id {session_id}: {e}", exc_info=True)
            conversation_stored_successfully = False
        finally:
            if conn:
                conn.close()
        return conversation_stored_successfully


    def get_conversation_history(self, session_id: str, limit: int = 10) -> list:
        """
        Retrieves the last N conversation turns for a given session_id from the database.

        Args:
            session_id (str): The ID of the conversation session.
            limit (int, optional): The maximum number of conversation turns to retrieve.
                Defaults to 10.

        Returns:
            list: A list of dictionaries, where each dictionary represents a
                  conversation turn with 'user_message', 'ai_response', and 'timestamp'.
                  Returns an empty list if no history is found or an error occurs.
        """
        history = []
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Select 'id' as well to ensure consistent ordering if timestamps are identical
            cursor.execute('''
                SELECT user_message, ai_response, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
            ''', (session_id, limit))
            rows = cursor.fetchall()
            # Fetched rows are newest first due to ORDER BY DESC.
            # Reverse to get chronological order (oldest of the limited set first).
            for row in reversed(rows): # row[0]=user_message, row[1]=ai_response, row[2]=timestamp
                history.append({"user_message": row[0], "ai_response": row[1], "timestamp": row[2]})
            logger.debug(f"Retrieved {len(history)} turns for session_id {session_id} with limit {limit} using get_conversation_history.")
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving history for session_id {session_id} using get_conversation_history: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()
        return history

    def retrieve_relevant_conversation_turns(self, session_id: str, query_text: str, top_k: int = 3) -> list:
        """
        Retrieves conversation turns semantically relevant to a query, or falls back to recent history.

        If an embedding model is available, it performs a semantic search over stored
        conversation embeddings for the given session. Otherwise, it returns the
        most recent `top_k` turns for that session.

        Args:
            session_id (str): The ID of the conversation session.
            query_text (str): The text to find relevant conversation turns for.
            top_k (int, optional): The number of most relevant turns to retrieve. Defaults to 3.

        Returns:
            list: A list of relevant conversation turn dictionaries, ordered by relevance
                  (most relevant first) if semantic search is used, or by recency
                  (most recent first, then reversed to be chronological) if fallback is used.
                  Returns empty list on error or if no relevant turns found.
        """
        if not self.embedding_model:
            logger.info(f"No embedding model loaded. Falling back to retrieving recent history for session {session_id}.")
            # Fallback: return last top_k turns (chronological order)
            return self.get_conversation_history(session_id, limit=top_k)

        logger.info(f"Performing semantic search for session {session_id} with query: '{query_text[:50]}...'")
        conn = None
        relevant_turns = []
        try:
            query_embedding = self.embedding_model.encode(query_text)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Retrieve all embeddings for the session
            cursor.execute('''
                SELECT ce.conversation_id, ce.embedding, c.user_message, c.ai_response, c.timestamp
                FROM conversation_embeddings ce
                JOIN conversations c ON ce.conversation_id = c.id
                WHERE ce.session_id = ?
                ORDER BY c.timestamp DESC, c.id DESC
            ''', (session_id,))

            all_session_embeddings = []
            for row in cursor.fetchall():
                conv_id, serialized_embedding, user_msg, ai_msg, ts = row
                try:
                    stored_embedding = pickle.loads(serialized_embedding)
                    all_session_embeddings.append({
                        "conv_id": conv_id,
                        "embedding": stored_embedding,
                        "user_message": user_msg,
                        "ai_response": ai_msg,
                        "timestamp": ts
                    })
                except pickle.UnpicklingError as pe:
                    logger.error(f"Failed to unpickle embedding for conv_id {conv_id}: {pe}", exc_info=True)

            if not all_session_embeddings:
                logger.info(f"No embeddings found for session {session_id} to perform semantic search.")
                return []

            # Calculate cosine similarities
            similarities = []
            for item in all_session_embeddings:
                sim = self._cosine_similarity(query_embedding, item["embedding"])
                similarities.append((sim, item)) # Store score with the item data

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Get top_k results
            for i in range(min(top_k, len(similarities))):
                sim_score, turn_data = similarities[i]
                relevant_turns.append({
                    "user_message": turn_data["user_message"],
                    "ai_response": turn_data["ai_response"],
                    "timestamp": turn_data["timestamp"],
                    "similarity_score": sim_score # Optionally include score
                })
            logger.info(f"Semantic search retrieved {len(relevant_turns)} turns for session {session_id}.")

        except Exception as e: # Catch sentence_transformers errors or DB errors
            logger.error(f"Error during semantic retrieval for session {session_id}: {e}", exc_info=True)
            # Fallback to recent history on error during semantic search itself
            return self.get_conversation_history(session_id, limit=top_k)
        finally:
            if conn:
                conn.close()
        return relevant_turns

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine similarity between two numpy vectors."""
        if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
            logger.warning("Inputs to cosine_similarity are not numpy arrays.")
            return 0.0

        # Ensure vectors are 1-D
        if vec1.ndim > 1: vec1 = vec1.flatten()
        if vec2.ndim > 1: vec2 = vec2.flatten()

        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0 # Avoid division by zero

        dot_product = np.dot(vec1, vec2)
        return dot_product / (norm_vec1 * norm_vec2)


    def save_user_preference(self, user_id: str, key: str, value: str) -> bool:
        """
        Saves or updates a user preference in the database.

        Args:
            user_id (str): The ID of the user.
            key (str): The preference key (e.g., 'language', 'theme').
            value (str): The preference value.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_preferences (user_id, preference_key, preference_value)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, preference_key) DO UPDATE SET preference_value = excluded.preference_value
            ''', (user_id, key, value))
            conn.commit()
            logger.info(f"Saved preference for user_id {user_id}: {key} = {value}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database error saving preference for user_id {user_id}, key {key}: {e}", exc_info=True)
            return False
        finally:
            if conn:
                conn.close()
        return False


    def get_user_preference(self, user_id: str, key: str) -> str | None:
        """
        Retrieves a user preference from the database.

        Args:
            user_id (str): The ID of the user.
            key (str): The preference key.

        Returns:
            str | None: The preference value if found, otherwise None.
        """
        preference_value = None
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT preference_value
                FROM user_preferences
                WHERE user_id = ? AND preference_key = ?
            ''', (user_id, key))
            row = cursor.fetchone()
            if row:
                preference_value = row[0]
                logger.debug(f"Retrieved preference for user_id {user_id}, key {key}: {preference_value}")
            else:
                logger.debug(f"No preference found for user_id {user_id}, key {key}")
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving preference for user_id {user_id}, key {key}: {e}", exc_info=True)
        finally:
            if conn:
                conn.close()
        return preference_value

    # Placeholder method: This method is primarily for illustration within DialogueManager.
    # The main response generation logic, including NLU, is typically orchestrated by AIBrain.
    def get_response(self, user_input: str, session_id: str = "default_session", user_id: str = "default_user") -> str:
        """
        (Placeholder) Processes user input and generates a basic AI response.

        This method demonstrates interaction with history and preferences but is
        not intended for full AI response generation, which is handled by `AIBrain`.

        Args:
            user_input (str): The user's message.
            session_id (str, optional): The current session ID.
            user_id (str, optional): The current user ID.

        Returns:
            str: A placeholder AI response.
        """
        # NLU processing would typically occur in AIBrain before calling DM or specific DM methods.

        self.add_to_history("user", user_input)

        ai_response_text = f"Placeholder response to: '{user_input}' from DialogueManager." # Clarified source

        preferred_language = self.get_user_preference(user_id, "language")
        if not preferred_language:
            logger.info(f"No language preference found for user {user_id} in get_response. Setting to 'English'.")
            self.save_user_preference(user_id, "language", "English")
            preferred_language = "English"

        if preferred_language:
            ai_response_text += f" (Responding in your preferred language: {preferred_language})"

        self.add_to_history("assistant", ai_response_text)
        if not self.store_conversation(session_id, user_input, ai_response_text):
            logger.warning(f"Failed to store conversation for session {session_id} within get_response.")

        return ai_response_text

if __name__ == '__main__':
    # This block is for basic, direct testing or demonstration of DialogueManager.
    # For comprehensive unit tests, please run tests from the 'tests' directory
    # (e.g., python -m unittest tests.test_dialogue_manager).

    # Setup for direct script execution: use a dedicated test DB in a 'data' subdirectory
    # relative to this script's location.
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path like: .../ai-assistant/data/dm_direct_test_memory.db
    test_db_main_path = os.path.join(module_dir, '..', 'data', 'dm_direct_test_memory.db')

    # Ensure the 'data' directory exists for the test DB
    data_dir_main_test = os.path.dirname(test_db_main_path)
    if not os.path.exists(data_dir_main_test):
        os.makedirs(data_dir_main_test)
        logger.info(f"Created data directory for direct test DB: {data_dir_main_test}")

    if os.path.exists(test_db_main_path):
        os.remove(test_db_main_path) # Clean up before test

    logger.info("--- DialogueManager Direct Test/Demonstration Start ---")
    # Note: The logger configuration at the top of the file will apply.
    manager = DialogueManager(db_path=test_db_main_path)

    logger.info("Dialogue Manager Initialized for direct test.")

    # Short-term memory test
    manager.add_to_history("user", "Hello AI from direct test!")
    manager.add_to_history("assistant", "Hello User from direct test!")
    logger.info(f"Short-term history: {manager.get_history()}")

    # Store and retrieve conversation
    test_session_id = "direct_session_123"
    test_user_id = "direct_user_abc"
    manager.store_conversation(test_session_id, "What is the weather like today?", "It's partly cloudy.")
    manager.store_conversation(test_session_id, "Any chance of rain?", "A slight chance later this evening.")
    logger.info(f"Long-term history for {test_session_id}: {manager.get_conversation_history(test_session_id)}")

    # User preferences
    manager.save_user_preference(test_user_id, "theme", "dark_mode")
    retrieved_theme = manager.get_user_preference(test_user_id, "theme")
    logger.info(f"Retrieved theme for {test_user_id}: {retrieved_theme}")

    manager.save_user_preference(test_user_id, "language", "Klingon") # Qapla'!
    retrieved_lang = manager.get_user_preference(test_user_id, "language")
    logger.info(f"Retrieved language for {test_user_id}: {retrieved_lang}")

    # Test placeholder get_response
    response1 = manager.get_response("This is a test message for get_response.", session_id="direct_s1", user_id="direct_u1")
    logger.info(f"AI Response 1 (get_response): {response1}")
    response2 = manager.get_response("Another test message for get_response.", session_id="direct_s1", user_id="direct_u1")
    logger.info(f"AI Response 2 (get_response): {response2}")

    logger.info(f"Long-term history for direct_s1 after get_response: {manager.get_conversation_history('direct_s1')}")
    logger.info(f"User direct_u1 language preference: {manager.get_user_preference('direct_u1', 'language')}")

    # Clean up the test database after the direct script run
    if os.path.exists(test_db_main_path):
        os.remove(test_db_main_path)
        logger.info(f"Cleaned up direct test DB: {test_db_main_path}")
    else:
        logger.warning(f"Direct test DB not found for cleanup: {test_db_main_path}")
    logger.info("--- DialogueManager Direct Test/Demonstration Finished ---")
