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
import re
import sys
import threading
import time
import os
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SPIDER_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_spider")
DASH_DIR = Path(r"C:\02_QUILLAN\06 - Deployment & Platforms")
sys.path.insert(0, str(SPIDER_DIR))
import server as spider  # noqa: E402

WORK_LOCK = threading.Lock()
STARTED = time.time()
STATS = {"tasks_done": 0, "files": 0, "findings": 0, "wires": 0, "task_secs": 0.0,
         "sweeps": 0, "sweep_files": 0}
WATCH = {"on": False, "root": r"C:\02_QUILLAN\09 - Projects\mcp", "interval": 30,
         "watermark": 0.0, "heartbeat": 0, "last": None}

# ---- live agent roster: one spider body per agent, minis per sub-agent ----
# Agents heartbeat in; dashboard renders each as a colored, named spider bot.
AGENTS = {}
AGENTS_LOCK = threading.Lock()
AGENTS_FILE = SPIDER_DIR / "agents.json"
AGENT_TTL = 60  # seconds without heartbeat -> shown stale/offline
AGENT_COLORS = ["#49e0da", "#f6b84b", "#ff7b72", "#a371f7",
                "#3fb950", "#58a6ff", "#d29922", "#ff9d5c"]
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


def _load_agents():
    try:
        data = json.loads(AGENTS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_agents():
    try:
        AGENTS_FILE.write_text(json.dumps(AGENTS), encoding="utf-8")
    except Exception:
        pass


def _agent_color(agent_id, want):
    if want and HEX_COLOR.match(str(want)):
        return str(want)
    h = 0
    for ch in str(agent_id):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return AGENT_COLORS[h % len(AGENT_COLORS)]


def _clean_str(v, limit):
    s = str(v if v is not None else "")[:limit]
    return s.replace("\r", " ").replace("\n", " ")


AGENTS.update(_load_agents())


SPEC_KEYS = ["role", "objective", "context", "rules", "review", "start", "ship"]
SPEC_COLORS = {"role": "#f2f0e8", "objective": "#49e0da", "context": "#ff3d81",
               "rules": "#3fb950", "review": "#a371f7", "start": "#f6b84b",
               "ship": "#ff9d5c"}


def _spec_of(path):
    p = str(path or "").lower()
    if "memory" in p or "lancedb" in p or "02 - knowledge" in p or "obsidian" in p or "vault" in p:
        return "review"
    if "06 -" in p or "deploy" in p or "dashboard" in p or "collector" in p or "09 -" in p or "mcp" in p or "server" in p:
        return "rules"
    if "05 -" in p or "media" in p or "creative" in p:
        return "context"
    if "03 - training" in p or "model" in p or "forge" in p or "04-" in p or "skills" in p or "swarm" in p:
        return "objective"
    if "01 - core" in p or "cognition" in p or "samurai" in p:
        return "role"
    return "start"


def _spec_snapshot():
    now = time.time()
    try:
        mem = json.loads((SPIDER_DIR / "memory_graph.json").read_text(encoding="utf-8"))
    except Exception:
        mem = {}
    try:
        trail = json.loads((SPIDER_DIR / "trail.json").read_text(encoding="utf-8"))
    except Exception:
        trail = {}
    events = trail.get("trail", []) if isinstance(trail, dict) else []
    mn = mem.get("nodes", {}) if isinstance(mem, dict) else {}
    census = mem.get("census", {}) if isinstance(mem, dict) else {}
    sectors = census.get("sectors", {}) if isinstance(census, dict) else {}

    files = {k: 0 for k in SPEC_KEYS}
    flagged = {k: 0 for k in SPEC_KEYS}
    for full, meta in mn.items():
        sk = _spec_of(full)
        files[sk] += 1
        if meta.get("findings"):
            flagged[sk] += 1
    totals = {k: 0 for k in SPEC_KEYS}
    for sec, cnt in sectors.items():
        totals[_spec_of(sec)] += int(cnt or 0)
    for k in SPEC_KEYS:
        totals[k] = max(totals[k], files[k], 1)

    # active phase = section of the freshest trail event, else start
    active = "start"
    if events:
        last = events[-1]
        active = _spec_of(last.get("full", ""))
        if active not in SPEC_KEYS:
            active = "start"
    order = SPEC_KEYS
    phases = []
    ai = order.index(active) if active in order else 5
    for i, k in enumerate(order):
        st = "done" if i < ai else ("reading" if i == ai else "queued")
        phases.append({"key": k, "color": SPEC_COLORS[k], "state": st,
                       "have": files[k], "total": totals[k],
                       "words": files[k] * 31 + flagged[k] * 7})

    words_read = sum(p["words"] for p in phases)
    links = len(mem.get("edges", {})) if isinstance(mem.get("edges", {}), dict) else 0
    flags = sum(flagged.values())
    with AGENTS_LOCK:
        live = [a for a in AGENTS.values() if now - a.get("ts", 0) <= AGENT_TTL]
        minis = sum(len(a.get("children", [])) for a in live)
    cov = []
    for k in ["role", "objective", "context", "rules", "review"]:
        cov.append(round(min(1.0, files[k] / max(1, totals[k])), 3))
    cov_avg = sum(cov) / max(1, len(cov))
    wire_ratio = min(1.0, links / 800.0)
    flag_sig = min(1.0, flags / 40.0)
    score = int(round(100 * (0.5 * cov_avg + 0.3 * wire_ratio + 0.2 * min(1.0, flag_sig + 0.15))))
    score = max(0, min(100, score))

    heat = []
    fmax = max(1, max(files.values()))
    order_h = ["role", "objective", "context", "rules", "review", "start", "ship"]
    for r in range(5):
        row = []
        for cix in range(8):
            base = files[order_h[(r + cix) % len(order_h)]] / fmax
            row.append(round(base * (0.6 + 0.4 * ((r * 8 + cix) % 5) / 4), 2))
        heat.append(row)

    log = []
    for ev in events[-8:]:
        nm = str(ev.get("full", "")).split("\\")[-1].split("/")[-1]
        log.append("read %s" % (nm[:42] or "…"))
    try:
        tasks = json.loads((SPIDER_DIR / "tasks.json").read_text(encoding="utf-8"))
    except Exception:
        tasks = []
    if isinstance(tasks, list) and tasks:
        t = tasks[-1]
        log.append("%s %s" % (t.get("state", "task"), str(t.get("task", ""))[:40]))

    code = []
    if events:
        try:
            rd = spider.spider_read_file(events[-1].get("full", ""), 8)
            for ln in (rd.get("lines", []) or [])[:8]:
                code.append(str(ln)[-90:])
        except Exception:
            pass
    if not code:
        code = ["awaiting run…"]

    return {"phases": phases, "active": active, "score": score,
            "words_read": words_read, "links": links, "flags": flags,
            "legs": 8 * max(1, len(live)), "tentacles": 2 * max(0, minis),
            "coverage": cov, "heat": heat, "log": log[-8:], "code": code,
            "files": files, "totals": totals, "now": now}


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
        if self.path.startswith("/dashboard"):
            try:
                body = (DASH_DIR / "quillan-dashboard.html").read_bytes()
            except Exception as e:
                return self._json({"error": "dashboard missing: %s" % e}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/spider-graph.js"):
            try:
                body = (DASH_DIR / "spider-graph.js").read_bytes()
            except Exception as e:
                return self._json({"error": "spider-graph.js missing: %s" % e}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/spider-3d.js"):
            try:
                body = (DASH_DIR / "spider-3d.js").read_bytes()
            except Exception as e:
                return self._json({"error": "spider-3d.js missing: %s" % e}, 404)
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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
                               "updated": mem.get("updated", 0),
                               "census": mem.get("census", {})})
        if self.path.startswith("/api/agents"):
            now = time.time()
            with AGENTS_LOCK:
                out = []
                for aid, a in AGENTS.items():
                    age = now - a.get("ts", 0)
                    out.append({"id": aid, "name": a.get("name", aid),
                                "color": a.get("color", "#49e0da"),
                                "file": a.get("file", ""), "task": a.get("task", ""),
                                "children": a.get("children", []),
                                "stale": age > AGENT_TTL, "age_s": round(age, 1)})
            out.sort(key=lambda x: x["id"])
            return self._json({"agents": out, "now": now, "ttl_s": AGENT_TTL})
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
        if self.path.startswith("/api/spec"):
            try:
                return self._json(_spec_snapshot())
            except Exception as e:
                return self._json({"error": "spec failed: %s" % e}, 500)
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
        if self.path.startswith("/api/agent/heartbeat"):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except Exception:
                n = 0
            try:
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            aid = _clean_str(payload.get("id", ""), 64)
            if not aid:
                return self._json({"error": "id required"}, 400)
            kids = []
            raw_kids = payload.get("children", [])
            if isinstance(raw_kids, list):
                for k in raw_kids[:12]:
                    if not isinstance(k, dict):
                        continue
                    kid_id = _clean_str(k.get("id", ""), 64)
                    if not kid_id:
                        continue
                    kids.append({"id": kid_id,
                                 "name": _clean_str(k.get("name", kid_id), 48),
                                 "file": _clean_str(k.get("file", ""), 220),
                                 "task": _clean_str(k.get("task", ""), 160)})
            with AGENTS_LOCK:
                AGENTS[aid] = {"name": _clean_str(payload.get("name", aid), 48),
                               "color": _agent_color(aid, payload.get("color", "")),
                               "file": _clean_str(payload.get("file", ""), 260),
                               "task": _clean_str(payload.get("task", ""), 200),
                               "children": kids,
                               "ts": time.time()}
                if len(AGENTS) > 24:
                    oldest = sorted(AGENTS.items(), key=lambda kv: kv[1].get("ts", 0))[0][0]
                    AGENTS.pop(oldest, None)
                _save_agents()
            return self._json({"ok": True, "id": aid,
                               "color": AGENTS[aid]["color"],
                               "children": len(kids)})
        if self.path.startswith("/api/agent/clear"):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except Exception:
                n = 0
            try:
                payload = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                payload = {}
            aid = _clean_str(payload.get("id", ""), 64)
            with AGENTS_LOCK:
                gone = AGENTS.pop(aid, None) is not None
                _save_agents()
            return self._json({"ok": True, "cleared": gone, "id": aid})
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

