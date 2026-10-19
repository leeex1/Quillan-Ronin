#!/usr/bin/env python3
"""Local-only event collector for the Quillan dashboard.

Pair with: C:\\02_QUILLAN\\06 - Deployment & Platforms\\system prompts\\quillan-dashboard.html
Dashboard polls GET /api/events on 127.0.0.1:8777.
Event producers (n8n pipeline, agents) POST {"agent","type","stage","status","detail"}.
Stdlib only. Localhost only. Nothing is saved.
"""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
from collections import deque
import json

HOST, PORT = "127.0.0.1", 8777
EVENTS = deque(maxlen=1000)


class Handler(BaseHTTPRequestHandler):
    def send_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_json_headers(204)

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/api/events":
            self.send_json_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return
        self.send_json_headers()
        self.wfile.write(json.dumps({"events": list(EVENTS)}).encode())

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/events":
            self.send_json_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            event = json.loads(self.rfile.read(size) or b"{}")
            if not isinstance(event, dict) or not event.get("agent") or not event.get("type"):
                raise ValueError("agent and type are required")
            event.setdefault("ts", datetime.now(timezone.utc).isoformat())
            event.setdefault("detail", "")
            event.setdefault("tokens", 0)
            EVENTS.append(event)
            self.send_json_headers(202)
            self.wfile.write(b'{"accepted":true}')
        except (ValueError, json.JSONDecodeError):
            self.send_json_headers(400)
            self.wfile.write(b'{"error":"invalid event"}')

    def log_message(self, fmt, *args):
        return


print(f"Quillan collector listening on http://{HOST}:{PORT}")
print("POST events to /api/events; press Ctrl+C to stop.")
ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
