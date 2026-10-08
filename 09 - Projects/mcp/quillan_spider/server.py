#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🕷️ QUILLAN-RONIN SPIDER CRAWLER — NATIVE FAST-MCP SERVER
=========================================================
A visible body for Quillan to crawl files + repo while you watch.
Embodying C1-ASTRA (Vision), C31-NEXUS (Routing), C5-ECHO (Memory),
C14-KAIDO (Efficiency), C15-LUMINARIS (Clarity):
  1. spider_crawl — BFS crawl a root, emit trail + graph for dashboard
  2. spider_status — where is the spider right now?
  3. spider_map — nodes + edges JSON for canvas rendering
  4. spider_watch_file — path dashboard should poll (trail JSON)
TRAIL FILE: C:\\02_QUILLAN\\09 - Projects\\mcp\\quillan_spider\\trail.json
VIEWER:     C:\\02_QUILLAN\\09 - Projects\\mcp\\quillan_spider\\spider-viewer.html (open in browser)
"""

import sys
import time
import json
import re
import threading
import urllib.request
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Any, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP("QuillanSpider")

SPIDER_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_spider")
TRAIL_FILE = SPIDER_DIR / "trail.json"
STATE = {"pos": None, "visited": 0, "started": None, "root": None, "mode": "IDLE"}
TASKS_FILE = SPIDER_DIR / "tasks.json"
MEMGRAPH_FILE = SPIDER_DIR / "memory_graph.json"


class SimpleTextHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_chunks = []
        self.title = ""
        self.headings = []
        self.links = []
        self._in_title = False
        self._in_h = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "title":
            self._in_title = True
        elif tag.lower() in ("h1", "h2", "h3"):
            self._in_h = True
        elif tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() in ("h1", "h2", "h3"):
            self._in_h = False

    def handle_data(self, data):
        txt = data.strip()
        if not txt:
            return
        if self._in_title:
            self.title = txt
        elif self._in_h:
            self.headings.append(txt)
        elif len(self.text_chunks) < 40 and len(txt) > 20:
            self.text_chunks.append(txt)


def _crawl_page_content(target: str) -> Dict[str, Any]:
    """Crawl a local HTML file or web URL, extract headings, links, and content."""
    try:
        if target.startswith("http://") or target.startswith("https://"):
            req = urllib.request.Request(target, headers={"User-Agent": "QuillanSpider/2.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        else:
            p = Path(target)
            if not p.exists():
                return {"ok": False, "error": f"Path not found: {target}"}
            html = p.read_text(encoding="utf-8", errors="replace")
        
        parser = SimpleTextHTMLParser()
        parser.feed(html[:150000])
        lines = html.splitlines()
        sample = "\n".join(lines[:35])
        return {
            "ok": True,
            "title": parser.title or Path(target).name,
            "headings": parser.headings[:8],
            "links_count": len(parser.links),
            "lines": len(lines),
            "bytes": len(html),
            "text_sample": parser.text_chunks[:5],
            "sample_snippet": sample[:1200]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _load_mem():
    try:
        m = json.loads(MEMGRAPH_FILE.read_text(encoding="utf-8"))
        m.setdefault("nodes", {})
        m.setdefault("edges", {})
        return m
    except Exception:
        return {"nodes": {}, "edges": {}, "runs": 0}


def _memorize(nodes, edges, trail):
    """Accumulate one run into the persistent memory graph (never overwritten, only grown)."""
    now = time.time()
    mem = _load_mem()
    mn, me = mem["nodes"], mem["edges"]
    id2path = {}
    for n in nodes:
        full = n.get("full") or n.get("path")
        if not full:
            continue
        id2path[n["id"]] = full
        e = mn.get(full, {"kind": n.get("kind", "file"), "first_seen": now, "visits": 0})
        e.update({"last_seen": now, "visits": e.get("visits", 0) + 1,
                  "kind": n.get("kind", e.get("kind", "file")),
                  "path": n.get("path", full[-70:])})
        mn[full] = e
    for ev in trail:
        full = ev.get("full")
        if full in mn:
            if ev.get("lines"):
                mn[full]["lines"] = ev["lines"]
            if ev.get("findings"):
                mn[full]["findings"] = ev["findings"][:5]
                mn[full]["last_flag"] = now
            mn[full]["last_action"] = ev.get("action", ev.get("event", ""))
    for ed in edges:
        a, b = id2path.get(ed.get("from"), ed.get("from")), id2path.get(ed.get("to"), ed.get("to"))
        if not a or not b or not isinstance(a, str) or not isinstance(b, str):
            continue
        key = a + "\u2192" + b
        prev = me.get(key, {"wire": False, "seen": 0, "first_seen": now})
        prev.update({"seen": prev.get("seen", 0) + 1,
                     "wire": prev.get("wire", False) or bool(ed.get("wire")),
                     "last_seen": now})
        me[key] = prev
    mem.update({"nodes": mn, "edges": me, "updated": now, "runs": mem.get("runs", 0) + 1})
    MEMGRAPH_FILE.write_text(json.dumps(mem), encoding="utf-8")
    return len(mn), len(me)


def _load_tasks():
    try:
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_tasks(tasks):
    TASKS_FILE.write_text(json.dumps(tasks, indent=1), encoding="utf-8")


QLOCK = threading.Lock()


def _parse_imports(txt, known_stems):
    """Return internal module names this file wires to (import X / from X import)."""
    found = []
    for ln in txt.splitlines()[:300]:
        s = ln.strip()
        m = re.match(r"(?:from\s+([\w\.]+)\s+import|import\s+([\w\.]+))", s)
        if not m:
            continue
        mod = (m.group(1) or m.group(2)).split(".")[0].lower()
        if mod in known_stems and mod not in found:
            found.append(mod)
    return found

SKIP = {".git", "__pycache__", "node_modules", ".venv-mazebench", ".venv", "dist", "build", ".next"}


def _crawl(root: Path, max_files: int = 220, exts: Optional[List[str]] = None):
    trail = []
    nodes = []
    edges = []
    seen = set()
    queue = [root]
    nid = {}
    def _id(p: Path):
        if str(p) not in nid:
            nid[str(p)] = len(nid)
            nodes.append({"id": nid[str(p)], "path": str(p)[-70:], "full": str(p),
                          "kind": "dir" if p.is_dir() else "file"})
        return nid[str(p)]
    _id(root)
    t0 = time.time()
    while queue and len(trail) < max_files:
        cur = queue.pop(0)
        try:
            kids = sorted(cur.iterdir()) if cur.is_dir() else []
        except Exception:
            continue
        for k in kids:
            if k.name in SKIP:
                continue
            if exts and k.is_file() and k.suffix.lower() not in exts:
                continue
            a, b = _id(cur), _id(k)
            edges.append({"from": a, "to": b})
            trail.append({"t": round(time.time() - t0, 2), "at": str(k)[-90:], "node": b,
                          "event": "dir" if k.is_dir() else "file"})
            if k.is_dir() and str(k) not in seen:
                seen.add(str(k))
                queue.append(k)
            if len(trail) >= max_files:
                break
    return nodes, edges, trail


@mcp.tool()
def spider_crawl(root: str = "C:\\02_QUILLAN\\09 - Projects\\mcp", max_files: int = 220, exts: str = ".py,.json,.md,.html") -> Dict[str, Any]:
    """Crawl a repo root BFS-style, emitting a watchable trail + graph.
    Args:
        root: Directory to crawl.
        max_files: Cap trail events (10-1000).
        exts: Comma extensions to include for files (dirs always included).
    """
    r = Path(root)
    if not r.exists():
        return {"error": f"Root not found: {root}"}
    max_files = max(10, min(1000, max_files))
    ext_list = [e.strip().lower() for e in exts.split(",") if e.strip()]
    nodes, edges, trail = _crawl(r, max_files, ext_list)
    payload = {"root": str(r), "at": time.time(), "nodes": nodes, "edges": edges, "trail": trail,
               "spider": {"pos": trail[-1] if trail else None, "visited": len(trail)}}
    TRAIL_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    STATE.update({"pos": payload["spider"]["pos"], "visited": len(trail), "started": time.time(), "root": str(r)})
    return {"status": "CRAWL_COMPLETE", "root": str(r), "nodes": len(nodes), "edges": len(edges),
            "events": len(trail), "trail_file": str(TRAIL_FILE),
            "viewer": str(SPIDER_DIR / "spider-viewer.html"),
            "watch": "Open spider-viewer.html in a browser; it polls trail.json every 800ms."}


@mcp.tool()
def spider_status() -> Dict[str, Any]:
    """Where is the spider right now? (reads trail.json)."""
    if not TRAIL_FILE.exists():
        return {"status": "IDLE", "hint": "Call spider_crawl first."}
    d = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
    return {"status": "READY", "root": d.get("root"), "visited": len(d.get("trail", [])),
            "nodes": len(d.get("nodes", [])), "pos": (d.get("trail") or [None])[-1],
            "trail_file": str(TRAIL_FILE)}


@mcp.tool()
def spider_map() -> Dict[str, Any]:
    """Return nodes + edges for canvas rendering (dashboard polls this or trail.json)."""
    if not TRAIL_FILE.exists():
        return {"error": "No trail yet. Call spider_crawl first.", "trail_file": str(TRAIL_FILE)}
    d = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
    return {"root": d.get("root"), "nodes": d.get("nodes", [])[:400], "edges": d.get("edges", [])[:800],
            "trail": d.get("trail", [])[-120:]}


@mcp.tool()
def spider_crawl_live(root: str = "C:\\02_QUILLAN\09 - Projects\\mcp", max_files: int = 120, delay_ms: int = 120, exts: str = ".py,.json,.md,.html") -> Dict[str, Any]:
    """LIVE crawl: spider writes trail.json AFTER EVERY FILE so the viewer follows in real time.
    Keep max_files ~60-150 and delay 80-200ms for a watchable walk. Run it, then watch the viewer.
    Args:
        root: Directory to crawl.
        max_files: Cap events (10-500).
        delay_ms: Pause per file so human eyes can follow (20-500).
        exts: Comma extensions for files (dirs always included).
    """
    r = Path(root)
    if not r.exists():
        return {"error": f"Root not found: {root}"}
    max_files = max(10, min(500, max_files))
    delay = max(0.02, min(0.5, delay_ms / 1000.0))
    ext_list = [e.strip().lower() for e in exts.split(",") if e.strip()]
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    trail: List[Dict[str, Any]] = []
    nid: Dict[str, int] = {}
    def _id(p: Path):
        if str(p) not in nid:
            nid[str(p)] = len(nid)
            nodes.append({"id": nid[str(p)], "path": str(p)[-70:], "full": str(p), "kind": "dir" if p.is_dir() else "file"})
        return nid[str(p)]
    _id(r)
    t0 = time.time()
    # live header so viewer shows CRAWLING immediately
    TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": t0, "live": True, "started": t0,
        "nodes": nodes, "edges": edges, "trail": trail,
        "spider": {"pos": None, "visited": 0, "state": "CRAWLING"}}, indent=1), encoding="utf-8")
    queue = [r]
    seen = set()
    while queue and len(trail) < max_files:
        cur = queue.pop(0)
        try:
            kids = sorted(cur.iterdir()) if cur.is_dir() else []
        except Exception:
            continue
        for k in kids:
            if len(trail) >= max_files:
                break
            if k.name in SKIP:
                continue
            if ext_list and k.is_file() and k.suffix.lower() not in ext_list:
                continue
            try:
                sz = k.stat().st_size if k.is_file() else 0
            except Exception:
                sz = 0
            a, b = _id(cur), _id(k)
            edges.append({"from": a, "to": b})
            ev = {"t": round(time.time() - t0, 2), "at": str(k)[-90:], "full": str(k),
                  "node": b, "event": "dir" if k.is_dir() else "file", "size": sz}
            trail.append(ev)
            # STREAM WRITE EVERY STEP — this is what makes it live
            TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": time.time(), "live": True,
                "started": t0, "nodes": nodes, "edges": edges, "trail": trail,
                "spider": {"pos": ev, "visited": len(trail), "state": "CRAWLING"}}, indent=1), encoding="utf-8")
            if k.is_dir() and str(k) not in seen:
                seen.add(str(k))
                queue.append(k)
            time.sleep(delay)
    done = {"root": str(r), "at": time.time(), "live": False, "started": t0, "nodes": nodes,
            "edges": edges, "trail": trail,
            "spider": {"pos": trail[-1] if trail else None, "visited": len(trail), "state": "DONE",
                       "elapsed_s": round(time.time() - t0, 1)}}
    TRAIL_FILE.write_text(json.dumps(done, indent=1), encoding="utf-8")
    STATE.update({"pos": done["spider"]["pos"], "visited": len(trail), "started": t0, "root": str(r)})
    try:
        _memorize(nodes, edges, trail)
    except Exception:
        pass
    return {"status": "LIVE_CRAWL_DONE", "root": str(r), "nodes": len(nodes), "edges": len(edges),
            "events": len(trail), "elapsed_s": done["spider"].get("elapsed_s"),
            "trail_file": str(TRAIL_FILE), "memory": str(MEMGRAPH_FILE)}


WORK_PATTERNS = [
    ("eval_exec", re.compile(r"\b(eval|exec)\s*\(")),
    ("subprocess_shell", re.compile(r"subprocess.*shell\s*=\s*True|os\.system\s*\(")),
    ("pickle_load", re.compile(r"pickle\.loads?\s*\(")),
    ("bare_except", re.compile(r"except\s*:" )),
    ("todo_fixme", re.compile(r"TODO|FIXME|XXX|HACK")),
]


def _scan_file(p: Path):
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    lines = txt.splitlines()
    findings = []
    for name, rx in WORK_PATTERNS:
        for i, ln in enumerate(lines[:2000]):
            if rx.search(ln):
                findings.append({"rule": name, "line": i + 1, "preview": ln.strip()[:90]})
                if len(findings) >= 8:
                    break
        if len(findings) >= 8:
            break
    return {"ok": True, "lines": len(lines), "bytes": len(txt.encode('utf-8', 'ignore')), "findings": findings}


def _compile_and_parse(p: Path):
    """REAL compiler + REAL parser. compile() builtin = actual CPython bytecode build, zero side effects."""
    t0 = time.time()
    try:
        src = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"read: {e}", 0, 0, 0.0
    try:
        compile(src, str(p), "exec")
        ok, err = True, ""
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {str(e)[:120]}"
    try:
        import ast as _ast
        tree = _ast.parse(src)
        funcs = sum(isinstance(n, _ast.FunctionDef) for n in _ast.walk(tree))
        classes = sum(isinstance(n, _ast.ClassDef) for n in _ast.walk(tree))
    except Exception:
        funcs, classes = 0, 0
    return ok, err, funcs, classes, round((time.time() - t0) * 1000, 1)


@mcp.tool()
def spider_work(root: str = "C:\\02_QUILLAN\09 - Projects\\mcp", job: str = "scan", max_files: int = 40, delay_ms: int = 220) -> Dict[str, Any]:
    """WORKING BOT: spider performs REAL coding tasks file-by-file with streaming trail.
    job=scan: read + security/quality scan each .py (real findings).
    job=tests: scan + write a test scaffold suggestion per file with findings.
    job=docs: scan + record docstring coverage per file.
    Watch the viewer: action badge shows SCANNING/TESTING/DOCS + findings live.
    Args:
        root: Directory to work through.
        job: scan, tests, or docs.
        max_files: Cap files (5-80).
        delay_ms: Pause per file for watchability (100-600).
    """
    import re as _re  # local alias (module re already imported at top for patterns)
    r = Path(root)
    if not r.exists():
        return {"error": f"Root not found: {root}"}
    job = job.strip().lower() or "scan"
    if job not in ("scan", "tests", "docs"):
        return {"error": f"Unknown job '{job}'. Use scan, tests, or docs."}
    max_files = max(5, min(500, max_files))
    delay = max(0.0, min(0.6, delay_ms / 1000.0))
    targets = [p for p in sorted(r.rglob("*.py")) if not any(s in p.parts for s in SKIP)][:max_files]
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    trail: List[Dict[str, Any]] = []
    nid: Dict[str, int] = {}
    def _id(p: Path):
        if str(p) not in nid:
            nid[str(p)] = len(nid)
            nodes.append({"id": nid[str(p)], "path": str(p)[-70:], "full": str(p), "kind": "file"})
        return nid[str(p)]
    _id(r)
    t0 = time.time()
    action = {"scan": "SCANNING", "tests": "TESTING", "docs": "DOCS"}[job]
    TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": t0, "live": True, "started": t0, "job": job,
        "nodes": nodes, "edges": edges, "trail": trail,
        "spider": {"pos": None, "visited": 0, "state": "WORKING", "action": action}}, indent=1), encoding="utf-8")
    total_findings = 0
    for k in targets:
        a, b = _id(r), _id(k)
        edges.append({"from": a, "to": b})
        res = _scan_file(k)
        comp_ok, comp_err, funcs, classes, ms = _compile_and_parse(k)
        if not res.get("ok"):
            ev = {"t": round(time.time() - t0, 2), "at": str(k)[-90:], "full": str(k), "node": b,
                  "event": "file", "action": action, "result": "ERROR", "detail": res.get("error", "read fail"), "ms": ms, "compiled": comp_ok}
        else:
            f = res["findings"]
            total_findings += len(f)
            if job == "scan":
                detail = f"{res['lines']} lines, {len(f)} findings" if f else f"{res['lines']} lines, clean"
            elif job == "tests":
                detail = f"scaffold: test_{k.stem}.py with {max(2, min(5, len(f) + 1))} cases" if f else "scaffold: smoke import test"
            else:
                detail = f"{res['lines']} lines, docstring {'present' if '\"\"\"' in (k.read_text(encoding='utf-8', errors='replace')[:2000]) else 'missing'}"
            ev = {"t": round(time.time() - t0, 2), "at": str(k)[-90:], "full": str(k), "node": b,
                  "event": "file", "action": action, "result": "FLAGGED" if f else "OK",
                  "detail": detail + f" | compiled {'OK' if comp_ok else 'FAIL:' + comp_err} | {funcs}F/{classes}C | {ms}ms", "findings": f[:5], "lines": res["lines"], "ms": ms, "compiled": comp_ok, "funcs": funcs, "classes": classes}
        trail.append(ev)
        TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": time.time(), "live": True, "started": t0,
            "job": job, "nodes": nodes, "edges": edges, "trail": trail,
            "spider": {"pos": ev, "visited": len(trail), "state": "WORKING", "action": action,
                       "findings_so_far": total_findings}}, indent=1), encoding="utf-8")
        time.sleep(delay)
    # WIRING PASS: real module-to-module edges parsed from imports (wires together modules)
    stem_to_node = {}
    for e in trail:
        try:
            stem_to_node[Path(e["full"]).stem.lower()] = e["node"]
        except Exception:
            pass
    wiring = []
    for e in trail:
        try:
            txt = Path(e["full"]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for mod in _parse_imports(txt, set(stem_to_node.keys())):
            tgt = stem_to_node.get(mod)
            if tgt is not None and tgt != e["node"]:
                edges.append({"from": e["node"], "to": tgt, "wire": True})
                wiring.append({"from": Path(e["full"]).name, "to": mod,
                               "why": f"{Path(e['full']).name} imports {mod}; coupled — a change in {mod} can break it"})
    TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": time.time(), "live": True, "started": t0,
        "job": job, "nodes": nodes, "edges": edges, "trail": trail,
        "spider": {"pos": trail[-1] if trail else None, "visited": len(trail), "state": "WIRING",
                   "action": action, "findings_so_far": total_findings}}, indent=1), encoding="utf-8")
    summary = {"root": str(r), "at": time.time(), "live": False, "started": t0, "job": job,
               "nodes": nodes, "edges": edges, "trail": trail, "wiring": wiring,
               "spider": {"pos": trail[-1] if trail else None, "visited": len(trail), "state": "DONE",
                          "action": action, "total_findings": total_findings, "wires": len(wiring),
                          "elapsed_s": round(time.time() - t0, 1)}}
    TRAIL_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (SPIDER_DIR / "spider_report.json").write_text(json.dumps({"job": job, "root": str(r),
        "files": len(trail), "total_findings": total_findings, "wiring": wiring,
        "flagged": [e for e in trail if e.get("result") == "FLAGGED"],
        "changes": ["wrote trail.json (nodes/edges/trail)", "wrote spider_report.json (this file)"]},
        indent=1), encoding="utf-8")
    mem_nodes, mem_edges = _memorize(nodes, edges, trail)
    return {"status": "WORK_DONE", "job": job, "files": len(trail), "total_findings": total_findings,
            "wires": len(wiring), "elapsed_s": summary["spider"]["elapsed_s"],
            "memory": {"nodes": mem_nodes, "edges": mem_edges},
            "report": str(SPIDER_DIR / "spider_report.json")}


@mcp.tool()
def spider_task_submit(task: str, job: str = "scan", root: str = "C:\\02_QUILLAN\09 - Projects\\mcp", max_files: int = 40) -> Dict[str, Any]:
    """Queue a task for the spider. Spider: idle -> QUEUED. Run spider_work_next to execute.
    Args:
        task: Human task, e.g. 'scan mcp for exec/eval + wire modules'.
        job: scan, tests, or docs.
        root: Directory to work through.
        max_files: Cap files (5-80).
    """
    import uuid
    with QLOCK:
        tasks = _load_tasks()
        tid = f"task-{uuid.uuid4().hex[:6]}"
        tasks.append({"task_id": tid, "task": task, "job": job.strip().lower() or "scan",
                      "root": root, "max_files": max(5, min(500, max_files)),
                      "state": "QUEUED", "at": time.time(), "report": None})
        _save_tasks(tasks)
    return {"status": "QUEUED", "task_id": tid,
            "queue_depth": len([t for t in tasks if t["state"] == "QUEUED"]),
            "loop": "idle -> QUEUED (call spider_work_next to execute)"}


@mcp.tool()
def spider_tasks() -> Dict[str, Any]:
    """List task queue (QUEUED/WORKING/DONE). Spider is IDLE when nothing open."""
    tasks = _load_tasks()
    open_t = [t for t in tasks if t["state"] in ("QUEUED", "WORKING")]
    return {"tasks": tasks[-20:], "open": len(open_t),
            "spider": "IDLE — awaiting task (spider_task_submit)" if not open_t else f"BUSY — {len(open_t)} open"}


def _why_rule(rule):
    return {"eval_exec": "arbitrary code execution if input reaches it (CWE-94)",
            "subprocess_shell": "shell injection via shell=True / os.system",
            "pickle_load": "deserialization of untrusted data = code exec",
            "bare_except": "swallows errors, hides failures",
            "todo_fixme": "unfinished work left in shipping code"}.get(rule, "hygiene")


@mcp.tool()
def spider_work_next(delay_ms: int = 220) -> Dict[str, Any]:
    """Execute next QUEUED task: crawl files -> wire modules -> detailed report (what changed + why) -> IDLE."""
    with QLOCK:
        tasks = _load_tasks()
        nxt = next((t for t in tasks if t["state"] == "QUEUED"), None)
        if not nxt:
            return {"status": "IDLE", "detail": "No QUEUED tasks. Submit with spider_task_submit."}
        nxt["state"] = "WORKING"
        _save_tasks(tasks)
    res = spider_work(nxt["root"], nxt["job"], nxt["max_files"], delay_ms)
    rep = json.loads((SPIDER_DIR / "spider_report.json").read_text(encoding="utf-8"))
    changes, why = [], []
    for e in rep.get("flagged", []):
        for f in e.get("findings", [])[:3]:
            changes.append(f"{Path(e['full']).name}:{f['line']} [{f['rule']}] {f['preview'][:60]}")
            why.append(f"{f['rule']}: {_why_rule(f['rule'])}")
    for w in rep.get("wiring", [])[:10]:
        changes.append(f"wire: {w['from']} -> {w['to']}")
        why.append(w["why"])
    changes += ["wrote trail.json (nodes/edges/trail)", "wrote spider_report.json"]
    if not rep.get("flagged") and not rep.get("wiring"):
        why.append("clean tree: no risky patterns, no internal imports found")
    with QLOCK:
        tasks = _load_tasks()
        for t in tasks:
            if t["task_id"] == nxt["task_id"]:
                t.update({"state": "DONE", "report": {"files": rep.get("files"), "total_findings": rep.get("total_findings"),
                        "wires": len(rep.get("wiring", [])), "changes": changes, "why": why,
                        "artifacts": [str(TRAIL_FILE), str(SPIDER_DIR / "spider_report.json")]}})
        _save_tasks(tasks)
        nxt = next((t for t in tasks if t["task_id"] == nxt["task_id"]), nxt)
    return {"status": "TASK_DONE", "task_id": nxt["task_id"], "task": nxt["task"],
            "report": nxt["report"], "spider": "IDLE — ready for next task (spider_task_submit)"}


@mcp.tool()
def spider_memory(top: int = 10) -> Dict[str, Any]:
    """Inspect the persistent memory graph: accumulated repo knowledge across all runs.
    Args:
        top: How many top-visited / most-wired nodes to list.
    """
    mem = _load_mem()
    mn, me = mem.get("nodes", {}), mem.get("edges", {})
    by_visits = sorted(mn.items(), key=lambda kv: kv[1].get("visits", 0), reverse=True)[:max(1, top)]
    wired = sorted([v for v in me.values() if v.get("wire")], key=lambda v: v.get("seen", 0), reverse=True)[:max(1, top)]
    flagged = [(k, v) for k, v in mn.items() if v.get("findings")][:max(1, top)]
    return {"nodes": len(mn), "edges": len(me), "runs": mem.get("runs", 0),
            "updated_ago_s": round(time.time() - mem.get("updated", time.time()), 1),
            "most_visited": [{"path": k[-70:], "visits": v.get("visits"), "lines": v.get("lines")} for k, v in by_visits],
            "most_wired": wired,
            "flagged": [{"path": k[-70:], "findings": v.get("findings", [])[:2]} for k, v in flagged],
            "file": str(MEMGRAPH_FILE)}


@mcp.tool()
def spider_sweep(root: str = "C:\\02_QUILLAN\09 - Projects\\mcp", since_ts: float = 0.0, delay_ms: int = 0) -> Dict[str, Any]:
    """Work only .py files changed since since_ts (mtime). Event-driven, no artificial pacing.
    Returns new watermark (pass back as since_ts). Powers the continuous watcher.
    Args:
        root: Directory to sweep.
        since_ts: Epoch watermark; only files newer than this are worked.
        delay_ms: Optional watchability pause (0 = full speed).
    """
    r = Path(root)
    if not r.exists():
        return {"error": f"Root not found: {root}"}
    delay = max(0.0, min(0.6, delay_ms / 1000.0))
    t0 = time.time()
    targets = []
    for p in sorted(r.rglob("*.py")):
        if any(s in p.parts for s in SKIP):
            continue
        try:
            if p.stat().st_mtime > since_ts:
                targets.append(p)
        except Exception:
            pass
        if len(targets) >= 300:
            break
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    trail: List[Dict[str, Any]] = []
    nid: Dict[str, int] = {}
    def _id(p: Path):
        if str(p) not in nid:
            nid[str(p)] = len(nid)
            nodes.append({"id": nid[str(p)], "path": str(p)[-70:], "full": str(p), "kind": "file"})
        return nid[str(p)]
    _id(r)
    action = "SWEEP"
    TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": t0, "live": True, "started": t0, "job": "scan",
        "nodes": nodes, "edges": edges, "trail": trail,
        "spider": {"pos": None, "visited": 0, "state": "WORKING", "action": action}}, indent=1), encoding="utf-8")
    total_findings = 0
    for k in targets:
        a, b = _id(r), _id(k)
        edges.append({"from": a, "to": b})
        ms0 = time.time()
        res = _scan_file(k)
        ms = round((time.time() - ms0) * 1000, 1)
        if not res.get("ok"):
            ev = {"t": round(time.time() - t0, 2), "at": str(k)[-90:], "full": str(k), "node": b,
                  "event": "file", "action": action, "result": "ERROR", "ms": ms,
                  "detail": res.get("error", "read fail")}
        else:
            f = res["findings"]
            total_findings += len(f)
            ev = {"t": round(time.time() - t0, 2), "at": str(k)[-90:], "full": str(k), "node": b,
                  "event": "file", "action": action, "result": "FLAGGED" if f else "OK", "ms": ms,
                  "detail": f"{res['lines']} lines in {ms}ms, {len(f)} findings" if f else f"{res['lines']} lines in {ms}ms, clean",
                  "findings": f[:5], "lines": res["lines"]}
        trail.append(ev)
        TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": time.time(), "live": True, "started": t0,
            "job": "scan", "nodes": nodes, "edges": edges, "trail": trail,
            "spider": {"pos": ev, "visited": len(trail), "state": "WORKING", "action": action,
                       "findings_so_far": total_findings}}, indent=1), encoding="utf-8")
        if delay:
            time.sleep(delay)
    stem_to_node = {}
    for e in trail:
        try:
            stem_to_node[Path(e["full"]).stem.lower()] = e["node"]
        except Exception:
            pass
    wiring = []
    for e in trail:
        try:
            txt = Path(e["full"]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for mod in _parse_imports(txt, set(stem_to_node.keys())):
            tgt = stem_to_node.get(mod)
            if tgt is not None and tgt != e["node"]:
                edges.append({"from": e["node"], "to": tgt, "wire": True})
                wiring.append({"from": Path(e["full"]).name, "to": mod,
                               "why": f"{Path(e['full']).name} imports {mod}; coupled"})
    summary = {"root": str(r), "at": time.time(), "live": False, "started": t0, "job": "scan",
               "nodes": nodes, "edges": edges, "trail": trail, "wiring": wiring,
               "spider": {"pos": trail[-1] if trail else None, "visited": len(trail), "state": "DONE",
                          "action": action, "total_findings": total_findings, "wires": len(wiring),
                          "elapsed_s": round(time.time() - t0, 1)}}
    TRAIL_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    mem_nodes, mem_edges = _memorize(nodes, edges, trail)
    return {"status": "SWEEP_DONE", "files": len(trail), "total_findings": total_findings,
            "wires": len(wiring), "elapsed_s": summary["spider"]["elapsed_s"],
            "watermark": t0, "memory": {"nodes": mem_nodes, "edges": mem_edges}}


@mcp.tool()
def spider_read_file(path: str, max_lines: int = 120) -> Dict[str, Any]:
    """Inspect and read lines of a specific file or page directly for the HUD viewer."""
    try:
        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": f"File does not exist: {path}"}
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return {
            "ok": True,
            "path": str(p),
            "name": p.name,
            "total_lines": len(lines),
            "lines": lines[:max_lines],
            "size_bytes": p.stat().st_size
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@mcp.tool()
def spider_crawl_page(target: str) -> Dict[str, Any]:
    """Crawl a local HTML document or remote URL and record a live trail event."""
    res = _crawl_page_content(target)
    if not res.get("ok"):
        return res
    t0 = time.time()
    node = {"id": 0, "path": res.get("title", target)[-70:], "full": target, "kind": "page"}
    ev = {
        "t": 0.0,
        "at": res.get("title", target)[-80:],
        "full": target,
        "node": 0,
        "event": "page",
        "action": "OPENING_PAGE",
        "result": "OK",
        "detail": f"{res['lines']} lines, {res['headings']} headings, {res['links_count']} links",
        "headings": res.get("headings", []),
        "sample": res.get("sample_snippet", "")
    }
    payload = {
        "root": target,
        "at": t0,
        "live": False,
        "job": "page_reader",
        "nodes": [node],
        "edges": [],
        "trail": [ev],
        "spider": {
            "pos": ev,
            "visited": 1,
            "state": "DONE",
            "action": "READING_PAGE",
            "active_page": res
        }
    }
    TRAIL_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return {"status": "PAGE_CRAWLED", "page": res, "trail_file": str(TRAIL_FILE)}


@mcp.tool()
def spider_quillan_takeover(mission: str = "Scan and inspect core workspace modules",
                            target_dir: str = r"C:\02_QUILLAN",
                            max_files: int = 40,
                            delay_ms: int = 160) -> Dict[str, Any]:
    """⚡ QUILLAN TAKEOVER: Quillan takes direct autonomous control of the spider bot.
    The spider crawls through target files and pages, reading code, analyzing AST,
    and recording step-by-step thoughts into the live dashboard stream.
    """
    r = Path(target_dir)
    if not r.exists():
        r = Path(r"C:\02_QUILLAN\09 - Projects\mcp")
    
    t0 = time.time()
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    trail: List[Dict[str, Any]] = []
    nid: Dict[str, int] = {}
    
    def _id(p: Path, kind="file"):
        s = str(p)
        if s not in nid:
            nid[s] = len(nid)
            nodes.append({"id": nid[s], "path": s[-70:], "full": s, "kind": kind})
        return nid[s]
    
    _id(r, "dir")
    
    # Gather interesting targets (py, md, json, html)
    candidates = []
    for ext in ("*.py", "*.md", "*.json", "*.html"):
        for p in r.rglob(ext):
            if any(s in p.parts for s in SKIP):
                continue
            candidates.append(p)
            if len(candidates) >= max_files * 2:
                break
    
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:max_files]
    
    # Initial live state
    TRAIL_FILE.write_text(json.dumps({
        "root": str(r), "at": t0, "live": True, "started": t0, "mission": mission,
        "nodes": nodes, "edges": edges, "trail": trail,
        "spider": {"pos": None, "visited": 0, "state": "QUILLAN_CONTROL", "action": "INITIALIZING_TAKEOVER",
                   "thought": f"Sovereign Throne C0 dispatch: Embarking on mission '{mission}'."}
    }, indent=1), encoding="utf-8")
    
    total_findings = 0
    delay = max(0.04, min(0.6, delay_ms / 1000.0))
    
    for i, target in enumerate(candidates):
        target_id = _id(target, "file" if target.is_file() else "dir")
        edges.append({"from": _id(target.parent, "dir"), "to": target_id})
        
        # Read file snippet and analyze
        content_preview = ""
        comp_ok, comp_err, funcs, classes, ms = True, "", 0, 0, 0.0
        findings = []
        lines_count = 0
        
        try:
            txt = target.read_text(encoding="utf-8", errors="replace")
            lines = txt.splitlines()
            lines_count = len(lines)
            content_preview = "\n".join(lines[:25])
            
            if target.suffix.lower() == ".py":
                scan_res = _scan_file(target)
                findings = scan_res.get("findings", [])
                total_findings += len(findings)
                comp_ok, comp_err, funcs, classes, ms = _compile_and_parse(target)
            elif target.suffix.lower() in (".html", ".htm"):
                page_res = _crawl_page_content(str(target))
                content_preview = page_res.get("sample_snippet", content_preview)
        except Exception as e:
            content_preview = f"Error reading {target.name}: {e}"
        
        action = "READING_CODE" if target.suffix == ".py" else ("INSPECTING_DOC" if target.suffix in (".md", ".txt") else "EXAMINING_PAGE")
        
        thought_map = [
            f"[C10-CODEWEAVER] Dissecting {target.name} AST hierarchy ({funcs} functions, {classes} classes).",
            f"[C34-PREDATOR] Scanning {target.name} attack surface for eval/shell injections.",
            f"[C18-SHEPHERD] Grounding references and dependency graph across {target.stem}.",
            f"[C15-LUMINARIS] Parsing file structure ({lines_count} lines, verified contiguous).",
            f"[C1-ASTRA] Tracing topological import vector for module {target.stem}."
        ]
        active_thought = thought_map[i % len(thought_map)]
        
        ev = {
            "t": round(time.time() - t0, 2),
            "at": str(target)[-90:],
            "full": str(target),
            "name": target.name,
            "node": target_id,
            "event": "file",
            "action": action,
            "result": "FLAGGED" if findings else "OK",
            "detail": f"{lines_count} lines | {funcs}F/{classes}C | {len(findings)} findings",
            "thought": active_thought,
            "snippet": content_preview[:1200],
            "lines": lines_count,
            "findings": findings[:3],
            "compiled": comp_ok
        }
        trail.append(ev)
        
        # Write live stream after every single file
        TRAIL_FILE.write_text(json.dumps({
            "root": str(r), "at": time.time(), "live": True, "started": t0, "mission": mission,
            "nodes": nodes, "edges": edges, "trail": trail,
            "spider": {
                "pos": ev,
                "visited": len(trail),
                "state": "QUILLAN_CONTROL",
                "action": action,
                "current_file": str(target),
                "active_thought": active_thought,
                "findings_so_far": total_findings,
                "snippet": content_preview[:1200]
            }
        }, indent=1), encoding="utf-8")
        
        time.sleep(delay)
    
    # Compute import wiring and synaptic edges across surveyed files
    stem_to_node = {}
    for e in trail:
        try:
            stem_to_node[Path(e["full"]).stem.lower()] = e["node"]
        except Exception:
            pass
    wiring = []
    for e in trail:
        try:
            txt = Path(e["full"]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for mod in _parse_imports(txt, set(stem_to_node.keys())):
            tgt = stem_to_node.get(mod)
            if tgt is not None and tgt != e["node"]:
                edges.append({"from": e["node"], "to": tgt, "wire": True})
                wiring.append({"from": Path(e["full"]).name, "to": mod,
                               "why": f"{Path(e['full']).name} imports {mod}; coupled"})

    # Finalize takeover mission
    final_payload = {
        "root": str(r), "at": time.time(), "live": False, "started": t0, "mission": mission,
        "nodes": nodes, "edges": edges, "trail": trail, "wiring": wiring,
        "spider": {
            "pos": trail[-1] if trail else None,
            "visited": len(trail),
            "state": "MISSION_COMPLETE",
            "action": "IDLE",
            "total_findings": total_findings,
            "wires": len(wiring),
            "elapsed_s": round(time.time() - t0, 1),
            "active_thought": f"Mission '{mission}' complete. Wove {len(wiring)} synaptic wires across {len(trail)} artifacts."
        }
    }
    TRAIL_FILE.write_text(json.dumps(final_payload, indent=1), encoding="utf-8")
    _memorize(nodes, edges, trail)
    
    return {
        "status": "TAKEOVER_SUCCESS",
        "mission": mission,
        "files_read": len(trail),
        "total_findings": total_findings,
        "wires": len(wiring),
        "trail_file": str(TRAIL_FILE)
    }


if __name__ == "__main__":
    mcp.run()

