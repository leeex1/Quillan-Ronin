#!/usr/bin/env python3
"""Collector: add POST /api/run proxy to the n8n production webhook."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\02_QUILLAN\06 - Deployment & Platforms\n8n\quillan_collector.py"
src = open(P, encoding="utf-8").read()

old_head = "from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler\n"
assert old_head in src
src = src.replace(
    old_head,
    "from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler\n"
    "import urllib.request\n",
    1)

old_post = """    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/events":
            self.send_json_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return
"""
assert old_post in src
new_post = """    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            self.forward_run()
            return
        if path != "/api/events":
            self.send_json_headers(404)
            self.wfile.write(b'{"error":"not found"}')
            return
"""
src = src.replace(old_post, new_post, 1)

anchor = "    def log_message(self, fmt, *args):\n        return\n"
assert anchor in src
forward = '''
    def forward_run(self):
        """Proxy a council run to the n8n production webhook (CORS-safe)."""
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size) or b"{}")
            query = (payload.get("query") or "").strip()
            if not query:
                self.send_json_headers(400)
                self.wfile.write(b'{"error":"query is required"}')
                return
            body = json.dumps({
                "query": query, "context": payload.get("context", ""),
                "documents": payload.get("documents", []),
            }).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:5678/webhook/quillan-think",
                data=body, headers={"Content-Type": "application/json"},
                method="POST")
            try:
                resp = urllib.request.urlopen(req, timeout=170)
                out = resp.read()
                self.send_json_headers(200)
                self.wfile.write(out)
            except Exception as e:
                self.send_json_headers(502)
                self.wfile.write(json.dumps(
                    {"error": f"n8n unreachable: {e}"}).encode())
        except (ValueError, json.JSONDecodeError):
            self.send_json_headers(400)
            self.wfile.write(b'{"error":"invalid request"}')

'''
src = src.replace(anchor, forward + anchor, 1)
open(P, "w", encoding="utf-8").write(src)
print("collector /api/run added")
