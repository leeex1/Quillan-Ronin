"""
Chat API for the AI Assistant using Flask and Flask-SocketIO.

This module sets up a web server to handle real-time chat interactions.
It provides a simple HTML frontend and uses SocketIO for communication
between the client (browser) and the AI Brain.
"""

from flask import Flask, render_template, request, jsonify  # Added jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import sys
import logging

# Configure basic logging for the API
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Path Setup for Core Module Import ---
# This ensures that 'core.ai_brain' can be imported correctly when chat_api.py
# is run directly or as part of a larger application structure.
# It assumes 'chat_api.py' is in 'api/' and 'ai_brain.py' is in 'core/',
# both under the main 'ai-assistant' module directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

from core.ai_brain import AIBrain  # pylint: disable=import-error,wrong-import-position
# Pylint disable comments are used here because pylint might complain in some IDE setups
# due to the dynamic sys.path modification. In a packaged application, this would be handled differently.

app = Flask(
    __name__,
    template_folder=os.path.join(project_root, "templates"),
    static_folder=os.path.join(project_root, "static"),
)
"""
The main Flask application instance.
Configured to serve templates and static files from standard locations
relative to the project root.
"""

# Fail closed: no predictable fallback key is compiled in. main.py refuses to
# serve until FLASK_SECRET_KEY is supplied by the environment.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "")

# CORS: explicit loopback allowlist only; wildcard origins are never permitted.
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "NEXTVERSE_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/.*": {"origins": cors_origins}})
logger.info("Flask-CORS initialized with explicit origin allowlist: %s", cors_origins)

socketio = SocketIO(app, cors_allowed_origins=cors_origins)
logger.info(
    "Flask-SocketIO initialized with explicit origin allowlist: %s", cors_origins
)


# --- AIBrain Initialization ---
# The AIBrain instance is created globally here. For more complex applications,
# consider using Flask's application context (g) or a factory pattern if AIBrain
# requires more dynamic configuration or has significant setup overhead.
# The database path for AIBrain's DialogueManager is constructed to be
# within the 'data' folder of the 'ai-assistant' module.
ai_brain_db_path = os.path.join(project_root, "data", "long_term_memory.db")
try:
    ai_brain = AIBrain(db_path=ai_brain_db_path)
    logger.info(f"AIBrain instantiated globally, using DB path: {ai_brain_db_path}")
except Exception as e:
    logger.error(
        f"Critical error: Failed to instantiate AIBrain with DB path {ai_brain_db_path}. Error: {e}",
        exc_info=True,
    )
    # Depending on the desired behavior, the application might exit or run in a degraded mode.
    # For now, it will likely fail if AIBrain is essential for all routes.
    ai_brain = None  # Ensure ai_brain is None if instantiation fails.
# --- End AIBrain Initialization ---


@app.route("/")
def index():
    """Serves the main chat interface page (index.html)."""
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    """Handles new client connections via SocketIO."""
    logger.info(f"Client connected: {request.sid}")
    emit("connection_status", {"status": "connected", "sid": request.sid})


@socketio.on("disconnect")
def handle_disconnect():
    """Handles client disconnections."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on("user_message")
def handle_user_message(data: dict):
    """
    Handles incoming user messages sent via SocketIO.

    It processes the message using the AIBrain and emits an 'ai_response'
    event back to the originating client.

    Args:
        data (dict): A dictionary containing the message data from the client.
                     Expected keys: 'message' (str), 'user_id' (str), 'session_id' (str).
    """
    logger.info(f"Received 'user_message' from SID {request.sid}: {data}")

    message_text = data.get("message")
    user_id = data.get(
        "user_id", "default_web_user"
    )  # Default if not provided by client
    session_id = data.get(
        "session_id", request.sid
    )  # Use SocketIO session ID if client doesn't send one

    if not message_text:
        logger.warning(f"Empty message received from SID {request.sid}.")
        emit("error_response", {"error": "Message content cannot be empty."})
        return

    if ai_brain is None:
        logger.error("AIBrain is not available. Cannot process user message.")
        emit(
            "error_response",
            {"error": "AI service is currently unavailable. Please try again later."},
        )
        return

    try:
        # Call AIBrain to generate a response
        ai_response_data = ai_brain.generate_response(
            user_id=user_id, session_id=session_id, user_input=message_text
        )

        logger.info(
            f"Sending 'ai_response' to SID {request.sid}: {ai_response_data.get('ai_response', '')[:100]}..."
        )
        emit("ai_response", ai_response_data)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(
            f"Error processing message or generating AI response for SID {request.sid}: {e}",
            exc_info=True,
        )
        emit(
            "error_response",
            {
                "error": "An unexpected error occurred on the server while processing your message."
            },
        )


# The main execution block for running the Flask-SocketIO server is typically in 'main.py'
# or a dedicated run script (e.g., 'run.py').
# Example for direct execution (useful for development/testing this API file):
# if __name__ == '__main__':
#     logger.info("Starting Flask-SocketIO server directly from chat_api.py for development...")
#     # For production, use a proper WSGI server like Gunicorn or uWSGI.
#     # Debug is OFF by default; use main.py, which fails closed without FLASK_SECRET_KEY.
#     socketio.run(app, host='127.0.0.1', port=5000, debug=False)


@app.route("/contextual_query", methods=["POST"])
def handle_contextual_query():
    """
    Handles contextual queries sent from the Game Engine.
    """
    if not request.is_json:
        logger.warning(
            f"/contextual_query: Received non-JSON request from {request.remote_addr}"
        )
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    logger.info(f"/contextual_query: Received data: {data} from {request.remote_addr}")

    event_type = data.get("event_type")
    object_id = data.get("object_id")
    player_location = data.get("player_location")  # Optional for now

    if not event_type or not object_id:
        logger.warning(
            f"/contextual_query: Missing event_type or object_id in request: {data}"
        )
        return jsonify({"error": "Missing event_type or object_id"}), 400

    if ai_brain is None:
        logger.error("/contextual_query: AIBrain instance is not available.")
        return jsonify({"error": "AI service component unavailable"}), 503

    try:
        # Call the new method in AIBrain
        ai_message = ai_brain.process_game_context_query(
            event_type, object_id, player_location
        )

        response_payload = {
            "status": "success",
            "ai_generated_message": ai_message,
            "original_query": data,
        }
        logger.info(f"/contextual_query: Sending response: {response_payload}")
        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(
            f"Error processing /contextual_query with AIBrain: {e}", exc_info=True
        )
        return jsonify({"error": "Internal server error processing query"}), 500
