import unittest
import sys
import os
import time

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import Flask components, skip tests if not available (due to env issues)
FLASK_AVAILABLE = False
try:
    from api.chat_api import app, socketio, ai_brain # Import ai_brain to potentially reconfigure for tests
    from core.dialogue_manager import DialogueManager # To manage DB cleanup for AIBrain's DM
    FLASK_AVAILABLE = True
except ImportError:
    print("WARNING: Flask or related components not found. Skipping TestChatAPI.")
    # To allow the file to be parsed, app and socketio might need to be dummy objects if not imported.
    # However, unittest.skipIf decorator is cleaner. For now, will rely on FLASK_AVAILABLE check.

# Define a dedicated test DB path for API tests to avoid conflicts if AIBrain's DM is used
TEST_API_DB_PATH = os.path.join(project_root, 'data', 'test_api_chat_memory.db')

@unittest.skipIf(not FLASK_AVAILABLE, "Flask or its dependencies are not installed, skipping API tests.")

class TestChatAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Configure app for testing
        app.config['TESTING'] = True
        # Use a separate, ephemeral database for AIBrain instance used by the API during tests
        if os.path.exists(TEST_API_DB_PATH):
            os.remove(TEST_API_DB_PATH)

        # Re-initialize AIBrain's dialogue manager with the test DB
        # This assumes ai_brain is a global instance in chat_api.py that we can modify
        # If ai_brain is instantiated per request or in a factory, this approach would differ
        ai_brain.dialogue_manager = DialogueManager(db_path=TEST_API_DB_PATH)
        print(f"TestChatAPI: AIBrain's DialogueManager reconfigured to use DB: {TEST_API_DB_PATH}")


    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_API_DB_PATH):
            os.remove(TEST_API_DB_PATH)
            print(f"TestChatAPI: Cleaned up test DB: {TEST_API_DB_PATH}")

    def setUp(self):
        self.app = app
        self.socketio = socketio
        # Note: test_client() should be called after app configuration and AIBrain re-init if needed
        self.client = self.socketio.test_client(self.app)
        # Wait a tiny bit for connect event to be processed if necessary
        # For simple connect/disconnect, usually not an issue, but for message chains can be.
        # time.sleep(0.01)
        # self.client.get_received() # Clear any connection messages if not explicitly tested for them right after connect

    def tearDown(self):
        if self.client.is_connected():
            self.client.disconnect()
        # time.sleep(0.01) # Allow time for disconnect to process if server-side handlers are slow

    def test_01_connect_disconnect_events(self):
        print("Running test_01_connect_disconnect_events...")
        self.assertTrue(self.client.is_connected())

        # Check for connection_status event (as per chat_api.py)
        received = self.client.get_received()
        self.assertTrue(len(received) > 0, "No events received on connect")
        connection_event_found = False
        for r_event in received:
            if r_event['name'] == 'connection_status':
                self.assertEqual(r_event['args'][0]['status'], 'connected')
                connection_event_found = True
                break
        self.assertTrue(connection_event_found, "connection_status event not received or has wrong data.")

        self.client.disconnect()
        self.assertFalse(self.client.is_connected())

    def test_02_user_message_receives_ai_response(self):
        print("Running test_02_user_message_receives_ai_response...")
        if not self.client.is_connected(): # Ensure client is connected for this test
            self.client.connect()
            self.client.get_received() # Clear connection messages

        self.assertTrue(self.client.is_connected())

        test_message = {
            'message': 'Hello from API test',
            'user_id': 'api_test_user',
            'session_id': 'api_test_session'
        }
        self.client.emit('user_message', test_message)

        # Wait for and get the response
        received_events = self.client.get_received() # Might need timeout if async tasks are very slow

        self.assertTrue(len(received_events) > 0, "No events received after sending user_message")

        ai_response_event = None
        for event in received_events:
            if event['name'] == 'ai_response':
                ai_response_event = event
                break

        self.assertIsNotNone(ai_response_event, "ai_response event not received.")

        response_data = ai_response_event['args'][0]
        self.assertIsInstance(response_data, dict)
        self.assertEqual(response_data['user_input'], test_message['message'])
        self.assertIn('ai_response', response_data)
        self.assertTrue(len(response_data['ai_response']) > 0)
        self.assertIn('nlu_details', response_data)
        self.assertIn('input_ethical_check', response_data)
        # output_ethical_check might be None if input was unethical
        # self.assertIn('output_ethical_check', response_data)
        self.assertIn('timestamp', response_data)

    def test_03_multiple_messages_same_session(self):
        print("Running test_03_multiple_messages_same_session...")
        if not self.client.is_connected():
            self.client.connect()
        self.client.get_received() # Clear connection messages
        self.assertTrue(self.client.is_connected())

        user_id = 'user_multi_msg'
        session_id = 'session_multi_msg'

        messages = [
            "Hello AI, this is the first message.",
            "This is the second message, are you remembering me?",
            "And a third one for good measure."
        ]

        for i, msg_text in enumerate(messages):
            print(f"  Sending message {i+1}: {msg_text}")
            test_message = {
                'message': msg_text,
                'user_id': user_id,
                'session_id': session_id
            }
            self.client.emit('user_message', test_message)

            # Wait for and get the response
            # Need to ensure we're not timing out if server is slow, but for local tests usually fine.
            # A timeout can be added to get_received: self.client.get_received(timeout=1)
            received_events = self.client.get_received()

            self.assertTrue(len(received_events) > 0, f"No events received after sending message {i+1}")

            ai_response_event = None
            for event in received_events:
                if event['name'] == 'ai_response':
                    ai_response_event = event
                    break

            self.assertIsNotNone(ai_response_event, f"ai_response event not received for message {i+1}")
            response_data = ai_response_event['args'][0]
            self.assertEqual(response_data['user_input'], msg_text)
            self.assertIn('ai_response', response_data)
            self.assertTrue(len(response_data['ai_response']) > 0, f"Empty AI response for message {i+1}")
            self.assertEqual(response_data['session_id'], session_id, f"Session ID mismatch for message {i+1}")
            print(f"  Received AI response for message {i+1}: {response_data['ai_response'][:50]}...")

        # Verify history in AIBrain's dialogue manager for this session
        # The AIBrain instance used by the API is `ai_brain` imported from `api.chat_api`
        history = ai_brain.dialogue_manager.get_conversation_history(session_id)
        self.assertEqual(len(history), len(messages), "Incorrect number of interactions stored in history.")
        for i, entry in enumerate(history):
            self.assertEqual(entry['user_message'], messages[i])
            self.assertIsNotNone(entry['ai_response'])


    def test_04_empty_message_error_response(self):
        print("Running test_04_empty_message_error_response...")
        if not self.client.is_connected():
            self.client.connect()
        self.client.get_received() # Clear connection messages
        self.assertTrue(self.client.is_connected())

        empty_message = {
            'message': '', # Empty message
            'user_id': 'user_empty_msg',
            'session_id': 'session_empty_msg'
        }
        self.client.emit('user_message', empty_message)

        received_events = self.client.get_received()
        self.assertTrue(len(received_events) > 0, "No events received after sending empty message")

        error_response_event = None
        for event in received_events:
            if event['name'] == 'error_response':
                error_response_event = event
                break

        self.assertIsNotNone(error_response_event, "error_response event not received for empty message.")
        error_data = error_response_event['args'][0]
        self.assertIsInstance(error_data, dict)
        self.assertIn('error', error_data)
        self.assertEqual(error_data['error'], 'Message content cannot be empty.')
        print(f"  Received expected error response: {error_data['error']}")

        # Ensure no ai_response was sent
        ai_response_event = None
        for event in received_events: # Check again in case multiple events were received
            if event['name'] == 'ai_response':
                ai_response_event = event
                break
        self.assertIsNone(ai_response_event, "ai_response event should not be received for an empty message.")


if __name__ == '__main__':
    unittest.main()
