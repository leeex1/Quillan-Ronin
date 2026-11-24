#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure the project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from nextverse_validation_suite.ui.backend.app import app as flask_app, socketio
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import Flask app or SocketIO components: {e}")
    print(
        "Please ensure 'nextverse_validation_suite.ui.backend.app' is correct and all dependencies are installed."
    )
    sys.exit(1)


def run_server(host, port, debug_mode):
    print("🚀 Starting NextVerse Formula Validation Suite...")
    print(f"🌐 Access dashboard at: http://{host}:{port}")
    if debug_mode:
        print("⚠️ Running in DEBUG mode.")

    # allow_unsafe_werkzeug is enabled ONLY when debug was explicitly requested.
    # A pinned Werkzeug PIN is never honoured, so a committed debugger PIN cannot
    # be used as a trusted credential.
    os.environ.pop("WERKZEUG_DEBUG_PIN", None)
    try:
        socketio.run(
            flask_app,
            host=host,
            port=port,
            debug=debug_mode,
            use_reloader=debug_mode,
            allow_unsafe_werkzeug=debug_mode,
        )
    except TypeError:  # Fallback for older Werkzeug versions
        socketio.run(
            flask_app, host=host, port=port, debug=debug_mode, use_reloader=debug_mode
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NextVerse Validation Suite Runner")
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", "127.0.0.1"),
        help="Host to bind the server to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", 5000)),
        help="Port to run the server on.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (sets reloader=True)."
    )

    args = parser.parse_args()

    # Fail closed: no signing key is embedded; the operator must supply one.
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise SystemExit(
            "REFUSED TO START: FLASK_SECRET_KEY must be set in the environment. "
            "No fallback secret is compiled into this service."
        )

    run_server(host=args.host, port=args.port, debug_mode=args.debug)
