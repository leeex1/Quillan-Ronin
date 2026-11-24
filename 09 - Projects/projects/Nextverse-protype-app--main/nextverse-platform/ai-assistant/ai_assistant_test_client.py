import socketio
import time

sio = socketio.Client(logger=True, engineio_logger=True)
AI_ASSISTANT_URL = 'ws://localhost:5000'

@sio.event
def connect():
    print('Connected to AI Assistant API')
    test_message = {
        'message': 'Hello AI, what is the QLRC boost status?', # Example message
        'user_id': 'manual_test_user_001',
        'session_id': 'manual_test_session_001'
    }
    print(f"Sending message: {test_message}")
    sio.emit('user_message', test_message)

@sio.event
def connect_error(data):
    print(f"Connection failed: {data}")

@sio.event
def disconnect():
    print('Disconnected from AI Assistant API')

@sio.on('connection_status')
def on_connection_status(data):
    print(f"Received connection_status: {data}")

@sio.on('ai_response')
def on_ai_response(data):
    print(f"Received ai_response: {data}")
    # Manual check for boost fragment:
    if "QVM-Enhanced by distilgpt2" in data.get('ai_response', ''): # Specific to QLRC-6B_simulated
        print("SUCCESS: Confirmed '[QVM-Enhanced by distilgpt2]' fragment!")
    else:
        print("NOTE: Expected QVM enhancement fragment NOT found.")
    sio.disconnect() # Disconnect after test

@sio.on('error_response')
def on_error_response(data):
    print(f"Received error_response: {data}")
    sio.disconnect()

if __name__ == '__main__':
    try:
        print(f"Attempting to connect to AI Assistant at {AI_ASSISTANT_URL}...")
        sio.connect(AI_ASSISTANT_URL, transports=['websocket'])
        sio.wait()
    except socketio.exceptions.ConnectionError as e:
        print(f"Could not connect to server: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if sio.connected:
            sio.disconnect()
        print("Client finished.")
