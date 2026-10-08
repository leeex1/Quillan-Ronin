#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕷️ QUILLAN SPIDER BRIDGE — CORS static server + task API for the dashboard.
Serves C:\\02_QUILLAN\\09 - Projects\\mcp\\quillan_spider on 127.0.0.1:8082
  GET  /api/trail -> trail.json (live crawl state)
  GET  /api/tasks -> tasks.json (queue)
  POST /api/task  {task, job, root, max_files, delay_ms} -> queues + worker runs spider_work_next
Dashboard Spider tab polls these; CORS * so file:// dashboard works.
"""

import json
import sys
import threading
import time
import os
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SPIDER_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_spider")
sys.path.insert(0, str(SPIDER_DIR))
import server as spider  # noqa: E402

WORK_LOCK = threading.Lock()
STARTED = time.time()
STATS = {"tasks_done": 0, "files": 0, "findings": 0, "wires": 0, "task_secs": 0.0,
         "sweeps": 0, "sweep_files": 0}
WATCH = {"on": False, "root": r"C:\02_QUILLAN\09 - Projects\mcp", "interval": 30,
         "watermark": 0.0, "heartbeat": 0, "last": None}


def _watch_loop():
    while WATCH["on"]:
        start = time.time()
        if WORK_LOCK.acquire(blocking=False):
            try:
                res = spider.spider_sweep(WATCH["root"], WATCH["watermark"], 0)
                WATCH["watermark"] = start
                WATCH["heartbeat"] = time.time()
                WATCH["last"] = {"files": res.get("files", 0), "findings": res.get("total_findings", 0),
                                 "wires": res.get("wires", 0), "at": round(start, 1)}
                STATS["sweeps"] += 1
                STATS["sweep_files"] += res.get("files", 0)
            except Exception as e:
                WATCH["last"] = {"error": str(e)}
            finally:
                try:
                    WORK_LOCK.release()
                except Exception:
                    pass
        else:
            WATCH["last"] = {"skipped": "task worker busy", "at": round(start, 1)}
            WATCH["heartbeat"] = time.time()
        for _ in range(int(WATCH["interval"]) * 2):
            if not WATCH["on"]:
                break
            time.sleep(0.5)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(SPIDER_DIR), **kw)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/trail"):
            try:
                return self._json(json.loads((SPIDER_DIR / "trail.json").read_text(encoding="utf-8")))
            except Exception as e:
                return self._json({"error": "no trail yet: %s" % e}, 404)
        if self.path.startswith("/api/tasks"):
            try:
                return self._json(json.loads((SPIDER_DIR / "tasks.json").read_text(encoding="utf-8")))
            except Exception:
                return self._json([])
        if self.path.startswith("/api/memory"):
            try:
                mem = json.loads((SPIDER_DIR / "memory_graph.json").read_text(encoding="utf-8"))
            except Exception as e:
                return self._json({"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0,
                                   "runs": 0, "note": "no memory yet: %s" % e})
            mn = mem.get("nodes", {})
            paths = sorted(mn.keys())[:400]
            idx = {p: i for i, p in enumerate(paths)}
            nodes = [{"id": idx[p], "path": mn[p].get("path", p[-70:]), "full": p,
                      "kind": mn[p].get("kind", "file"), "visits": mn[p].get("visits", 0),
                      "flagged": bool(mn[p].get("findings")),
                      "last_seen": mn[p].get("last_seen", 0)} for p in paths]
            edges = []
            for key, e in mem.get("edges", {}).items():
                if "\u2192" not in key:
                    continue
                a, b = key.split("\u2192", 1)
                if a in idx and b in idx:
                    edges.append({"from": idx[a], "to": idx[b], "wire": bool(e.get("wire"))})
                if len(edges) >= 800:
                    break
            return self._json({"nodes": nodes, "edges": edges, "total_nodes": len(mn),
                               "total_edges": len(mem.get("edges", {})), "runs": mem.get("runs", 0),
                               "updated": mem.get("updated", 0)})
        if self.path.startswith("/api/watch"):
            return self._json({"on": WATCH["on"], "root": WATCH["root"], "interval": WATCH["interval"],
                               "heartbeat_ago_s": round(time.time() - WATCH["heartbeat"], 1) if WATCH["heartbeat"] else -1,
                               "last": WATCH["last"]})
        if self.path.startswith("/api/stats"):
            up = time.time() - STARTED
            f = STATS["files"]
            return self._json({"uptime_s": round(up, 1), "pid": os.getpid(),
                               "python": sys.version.split()[0], "cpus": os.cpu_count(),
                               "tasks_done": STATS["tasks_done"], "files": f,
                               "findings": STATS["findings"], "wires": STATS["wires"],
                               "task_secs": round(STATS["task_secs"], 1),
                               "files_per_sec": round(f / STATS["task_secs"], 1) if STATS["task_secs"] > 0 else 0,
                               "sweeps": STATS["sweeps"], "sweep_files": STATS["sweep_files"],
                               "watch_on": WATCH["on"]})
        if self.path.startswith("/api/read"):
            qs = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(qs)
            target = params.get("path", [""])[0]
            max_lines = int(params.get("lines", [120])[0])
            return self._json(spider.spider_read_file(target, max_lines))
        if self.path.startswith("/api/report"):
            try:
                return self._json(json.loads((SPIDER_DIR / "spider_report.json").read_text(encoding="utf-8")))
            except Exception as e:
                return self._json({"error": "no report yet: %s" % e}, 404)
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/takeover"):
            try:
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            mission = payload.get("mission", "Scan core workspace modules")
            target_dir = payload.get("root", r"C:\02_QUILLAN")
            max_files = int(payload.get("max_files", 30))
            delay_ms = int(payload.get("delay_ms", 160))

            def run_to():
                if WORK_LOCK.acquire(blocking=False):
                    try:
                        spider.spider_quillan_takeover(mission, target_dir, max_files, delay_ms)
                    except Exception as e:
                        print("Quillan takeover worker error:", e, flush=True)
                    finally:
                        try:
                            WORK_LOCK.release()
                        except Exception:
                            pass

            threading.Thread(target=run_to, daemon=True).start()
            return self._json({"status": "TAKEOVER_INITIATED", "mission": mission, "root": target_dir})

        if self.path.startswith("/api/task"):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except Exception:
                n = 0
            try:
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            try:
                sub = spider.spider_task_submit(
                    payload.get("task", ""),
                    payload.get("job", "scan"),
                    payload.get("root", r"C:\02_QUILLAN\09 - Projects\mcp"),
                    int(payload.get("max_files", 40)),
                )
            except Exception as e:
                return self._json({"error": str(e)}, 500)
            delay = int(payload.get("delay_ms", 0))

            def run():
                if WORK_LOCK.acquire(blocking=False):
                    try:
                        while True:
                            t0 = time.time()
                            res = spider.spider_work_next(delay)
                            if res.get("status") == "IDLE":
                                break
                            rep = res.get("report", {})
                            STATS["tasks_done"] += 1
                            STATS["files"] += rep.get("files", 0)
                            STATS["findings"] += rep.get("total_findings", 0)
                            STATS["wires"] += rep.get("wires", 0)
                            STATS["task_secs"] += time.time() - t0
                    except Exception as e:
                        print("spider worker error:", e, flush=True)
                    finally:
                        try:
                            WORK_LOCK.release()
                        except Exception:
                            pass

            threading.Thread(target=run, daemon=True).start()
            return self._json(sub)
        if self.path.startswith("/api/watch"):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except Exception:
                n = 0
            try:
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            if payload.get("action") == "stop":
                WATCH["on"] = False
                return self._json({"watch": "stopping"})
            WATCH.update({"on": True,
                          "root": payload.get("root", WATCH["root"]),
                          "interval": max(10, min(300, int(payload.get("interval_s", 30)))),
                          "watermark": float(payload.get("since", 0.0))})

            def watchrun():
                _watch_loop()

            threading.Thread(target=watchrun, daemon=True).start()
            return self._json({"watch": "started", "root": WATCH["root"], "interval": WATCH["interval"]})
        return self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    print("spider bridge on 127.0.0.1:8082 serving %s" % SPIDER_DIR, flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8082), Handler).serve_forever()

