import sys
import os

# Ensure the root of the 'ai-assistant' module is in the Python path
# to allow for absolute imports like 'from api.chat_api import app, socketio'
# This assumes main.py is in 'nextverse-platform/ai-assistant/'
project_module_root = os.path.abspath(os.path.dirname(__file__))
if project_module_root not in sys.path:
    sys.path.insert(0, project_module_root)

from api.chat_api import app, socketio


def _env_flag(name: str) -> bool:
    """Parse a boolean environment flag; default is always False (safe default)."""
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    # Fail closed: never serve with a predictable or empty signing key.
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise SystemExit(
            "REFUSED TO START: FLASK_SECRET_KEY must be set in the environment. "
            "No fallback secret is compiled into this service."
        )

    # Debug is off by default and binds to loopback unless explicitly overridden.
    # A pinned Werkzeug PIN is never honoured, so the debugger cannot be unlocked
    # with a credential committed to source or logs.
    os.environ.pop("WERKZEUG_DEBUG_PIN", None)
    debug_enabled = _env_flag("NEXTVERSE_DEBUG")
    host = os.getenv("NEXTVERSE_HOST", "127.0.0.1")
    port = int(os.getenv("NEXTVERSE_PORT", "5000"))

    print(f"Starting NextVerse AI Assistant server on {host}:{port}...")
    if debug_enabled:
        print("WARNING: debug mode enabled; do not expose this server publicly.")
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug_enabled,
        allow_unsafe_werkzeug=debug_enabled,
    )
