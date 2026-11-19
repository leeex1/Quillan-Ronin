#!/usr/bin/env python3
"""LLM Ears bridge — exposes SynesthesiaEngine to the browser mastering console.

Start:  python llm_ears_bridge.py
Port:   8790
CORS:   *  (local tool only — bind 127.0.0.1)
Endpoints:
  GET /analyze?path=<file>&transcribe=0|1
      -> { ok, bpm, texture, centroid_hz, segments:[{start,end,text}], error }
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EARS_DIR = Path(r"C:\02_QUILLAN\09 - Projects\projects\Audio Engineer")
sys.path.insert(0, str(EARS_DIR))

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from LLM_Ears_compat import SynesthesiaEngine  # lazy wrapper

        _engine = SynesthesiaEngine(model_size="base")
    return _engine


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/analyze":
            self._send({"ok": False, "error": "unknown endpoint"}, 404)
            return
        q = urllib.parse.parse_qs(parsed.query)
        raw = (q.get("path") or [""])[0]
        path = urllib.parse.unquote(raw)
        transcribe = (q.get("transcribe") or ["0"])[0] == "1"
        if not path or not Path(path).exists():
            self._send({"ok": False, "error": f"file not found: {path}"}, 400)
            return
        try:
            eng = get_engine()
            bpm, texture, centroid = eng.acoustics(path)
            segments = []
            if transcribe:
                segments = eng.transcribe(path)
            self._send(
                {
                    "ok": True,
                    "bpm": bpm,
                    "texture": texture,
                    "centroid_hz": centroid,
                    "segments": segments[:40],
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._send({"ok": False, "error": str(exc)}, 500)


if __name__ == "__main__":
    print("LLM Ears bridge on http://127.0.0.1:8790", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8790), Handler).serve_forever()
