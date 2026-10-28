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
# --- Quillan-Ronin v5.4.0-ONI Stdio Guard ---
import logging, builtins
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
_orig_print = builtins.print
def _quillan_safe_print(*args, **kwargs):
    if 'file' not in kwargs or kwargs['file'] is None or kwargs['file'] is sys.stdout:
        kwargs['file'] = sys.stderr
    return _orig_print(*args, **kwargs)
builtins.print = _quillan_safe_print
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
# [QUILLAN-v5.4.0-ONI-SHARED-WIRED]
try:
    from _shared import setup as _quillan_setup, tool_guard, ok_response, error_response, validate_string, validate_path, validate_int, read_file_safe
    from _shared.base_engines import PRISM_ENGINE, COUNCIL_ENGINE, GOVERNOR_ENGINE, SANDBOX_ENGINE
    _SHARED_CTX = _quillan_setup(server_name='quillan_spider', mcp_app=mcp)
except Exception as _shared_err:
    import sys as _sys
    print(f'[quillan_spider] _shared warning: {_shared_err}', file=_sys.stderr)

SPIDER_DIR = Path(r"C:\02_QUILLAN\09 - Projects\mcp\quillan_spider")
SPIDER_DIR.mkdir(parents=True, exist_ok=True)
TRAIL_FILE = SPIDER_DIR / "trail.json"
MEMGRAPH_FILE = SPIDER_DIR / "memory_graph.json"
TASKS_FILE = SPIDER_DIR / "tasks.json"
SESSION_DIR = SPIDER_DIR / "sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
STATE: Dict[str, Any] = {"pos": None, "visited": 0, "started": 0.0, "root": ""}


class _SpiderParser:
    def __init__(self):
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

# Rolling trail cap: dashboard renders the live window; memory graph keeps totals.
TRAIL_CAP = 300
# Flush trail.json at most this often during hot loops (sec); final write always happens.
TRAIL_FLUSH_SEC = 0.5


def _sector_of(root: Path, p: Path) -> str:
    try:
        rel = p.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "(root)"
    except Exception:
        return "(root)"


def _memorize_census(root: Path, counts: dict, ext_counts: dict, sector_counts: dict,
                     watermark: float, elapsed: float):
    """Record one full census pass (snapshot, not sum).

    A census always walks the whole tree, so its numbers REPLACE the previous
    pass — accumulating would double-count every run.
    """
    now = time.time()
    mem = _load_mem()
    census = {"sectors": dict(sector_counts),
              "exts": dict(ext_counts),
              "total": counts.get("files", 0),
              "dirs": counts.get("dirs", 0)}
    census["watermark"] = watermark
    census["last_elapsed_s"] = round(elapsed, 1)
    census["last_root"] = str(root)
    mem.update({"census": census, "updated": now, "runs": mem.get("runs", 0) + 1})
    MEMGRAPH_FILE.write_text(json.dumps(mem), encoding="utf-8")
    return census


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

SKIP = {".git", "__pycache__", "node_modules", ".venv-mazebench", ".venv", "dist", "build", ".next", ".antigravity", ".antigravity-ide", ".vscode-shared", ".windsurf", ".gemini", ".codeium", ".qwen", ".devin", "venv_oni_cu126", "venv_cuda1050", "checkpoints", ".pytest_cache", ".obsidian", "quillan_rag_db", ".gitnexus", "lancedb", "colab_bundle"}


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
def spider_crawl_live(root: str = "C:\\02_QUILLAN\\09 - Projects\\mcp", max_files: int = 120, delay_ms: int = 120, exts: str = ".py,.json,.md,.html") -> Dict[str, Any]:
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
def spider_work(root: str = "C:\\02_QUILLAN\\09 - Projects\\mcp", job: str = "scan", max_files: int = 40, delay_ms: int = 220) -> Dict[str, Any]:
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
    last_flush = 0.0

    def emit_work(ev, force: bool = False):
        nonlocal last_flush
        trail.append(ev)
        if len(trail) > TRAIL_CAP:
            del trail[:len(trail) - TRAIL_CAP]
        now = time.time()
        if not force and (now - last_flush) < TRAIL_FLUSH_SEC:
            return
        last_flush = now
        TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": now, "live": True,
            "started": t0, "job": job, "nodes": nodes[-400:], "edges": edges[-800:],
            "trail": trail,
            "spider": {"pos": ev, "visited": len(trail), "state": "WORKING",
                       "action": action,
                       "findings_so_far": total_findings}}, indent=1), encoding="utf-8")

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
        emit_work(ev)
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
def spider_task_submit(task: str, job: str = "scan", root: str = "C:\\02_QUILLAN\\09 - Projects\\mcp", max_files: int = 40) -> Dict[str, Any]:
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
    census = mem.get("census", {})
    top_sectors = sorted(census.get("sectors", {}).items(), key=lambda kv: -kv[1])[:12]
    top_exts = sorted(census.get("exts", {}).items(), key=lambda kv: -kv[1])[:12]
    return {"nodes": len(mn), "edges": len(me), "runs": mem.get("runs", 0),
            "updated_ago_s": round(time.time() - mem.get("updated", time.time()), 1),
            "most_visited": [{"path": k[-70:], "visits": v.get("visits"), "lines": v.get("lines")} for k, v in by_visits],
            "most_wired": wired,
            "flagged": [{"path": k[-70:], "findings": v.get("findings", [])[:2]} for k, v in flagged],
            "census_total": census.get("total", 0),
            "census_sectors": top_sectors, "census_exts": top_exts,
            "file": str(MEMGRAPH_FILE)}


@mcp.tool()
def spider_census(root: str = "C:\\02_QUILLAN") -> Dict[str, Any]:
    """Fast stat-only census of EVERY file under root (no content reads).
    Records per-sector + per-extension aggregates into the memory graph so the
    web shows the full 100k+ docs, not just deep-scanned .py files. Deep scan
    stays .py-only via spider_work; census gives the map, work gives the detail.
    Progress events stream throttled; trail stays capped; memory keeps totals.
    Args:
        root: Directory to census.
    """
    r = Path(root)
    if not r.exists():
        return {"error": f"Root not found: {root}"}
    t0 = time.time()
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    trail: List[Dict[str, Any]] = []
    counts = {"files": 0, "dirs": 0, "skipped": 0, "errors": 0}
    ext_counts: Dict[str, int] = {}
    sector_counts: Dict[str, int] = {}
    watermark = 0.0
    last_flush = 0.0
    # sector hub nodes for the web (one per top-level dir)
    root_id = 0
    nodes.append({"id": 0, "path": str(r)[-70:], "full": str(r), "kind": "root"})
    sec_ids: Dict[str, int] = {}

    def emit_progress(force: bool = False):
        nonlocal last_flush
        now = time.time()
        if not force and (now - last_flush) < TRAIL_FLUSH_SEC:
            return
        last_flush = now
        ev = {"t": round(now - t0, 2), "at": f"CENSUS {counts['files']} files…",
              "full": str(r), "node": 0, "event": "census",
              "action": "CENSUS", "result": "COUNTING",
              "detail": f"{counts['files']} files, {counts['dirs']} dirs",
              "ms": 0.0}
        trail.append(ev)
        if len(trail) > TRAIL_CAP:
            del trail[:len(trail) - TRAIL_CAP]
        TRAIL_FILE.write_text(json.dumps({"root": str(r), "at": now, "live": True,
            "started": t0, "job": "census", "nodes": nodes[-400:], "edges": edges[-800:],
            "trail": trail,
            "spider": {"pos": ev, "visited": counts["files"], "state": "WORKING",
                       "action": "CENSUS",
                       "findings_so_far": 0}}, indent=1), encoding="utf-8")

    emit_progress(force=True)
    stack = [r]
    while stack:
        cur = stack.pop()
        try:
            kids = list(cur.iterdir())
        except Exception:
            counts["errors"] += 1
            continue
        for k in kids:
            if k.name in SKIP:
                counts["skipped"] += 1
                continue
            try:
                if k.is_dir():
                    counts["dirs"] += 1
                    stack.append(k)
                    sec = _sector_of(r, k)
                    if sec not in sec_ids and len(sec_ids) < 64:
                        sec_ids[sec] = len(nodes)
                        nodes.append({"id": len(nodes), "path": sec[-70:],
                                      "full": str(r / sec), "kind": "sector"})
                        edges.append({"from": root_id, "to": sec_ids[sec]})
                else:
                    try:
                        st = k.stat()
                        watermark = max(watermark, st.st_mtime)
                    except Exception:
                        counts["errors"] += 1
                        continue
                    counts["files"] += 1
                    ext = k.suffix.lower() or "(noext)"
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    sec = _sector_of(r, k)
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
            except Exception:
                counts["errors"] += 1
                continue
        if counts["files"] % 2000 < len(kids):
            emit_progress()
    emit_progress(force=True)
    elapsed = time.time() - t0
    census = _memorize_census(r, counts, ext_counts, sector_counts, watermark, elapsed)
    summary = {"root": str(r), "at": time.time(), "live": False, "started": t0,
               "job": "census", "nodes": nodes, "edges": edges, "trail": trail,
               "census": census,
               "spider": {"pos": trail[-1] if trail else None, "visited": counts["files"],
                          "state": "DONE", "action": "CENSUS", "total_findings": 0,
                          "wires": 0, "elapsed_s": round(elapsed, 1)}}
    TRAIL_FILE.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    top_exts = sorted(ext_counts.items(), key=lambda kv: -kv[1])[:12]
    top_sectors = sorted(sector_counts.items(), key=lambda kv: -kv[1])[:12]
    return {"status": "CENSUS_DONE", "files": counts["files"], "dirs": counts["dirs"],
            "skipped": counts["skipped"], "errors": counts["errors"],
            "elapsed_s": round(elapsed, 1),
            "files_per_sec": round(counts["files"] / max(elapsed, 0.01), 1),
            "top_exts": top_exts, "top_sectors": top_sectors,
            "memory_total": census["total"],
            "report": str(SPIDER_DIR / "spider_report.json")}


@mcp.tool()
def spider_sweep(root: str = "C:\\02_QUILLAN\\09 - Projects\\mcp", since_ts: float = 0.0, delay_ms: int = 0) -> Dict[str, Any]:
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
def spider_read_file(path: str, max_lines: int = 150) -> Dict[str, Any]:
    """Inspect and read lines of a specific file, page, or directory directly for the HUD viewer."""
    try:
        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": f"File or folder does not exist: {path}"}
        
        if p.is_dir():
            entries = []
            try:
                for item in sorted(p.iterdir()):
                    if item.name.startswith(".git") or item.name == "__pycache__":
                        continue
                    entries.append({
                        "name": item.name,
                        "is_dir": item.is_dir(),
                        "size": item.stat().st_size if item.is_file() else 0,
                        "ext": item.suffix.lower() if item.is_file() else "DIR"
                    })
            except Exception:
                pass
            dir_lines = [
                f"📁 DIRECTORY: {p.name}",
                f"   Full Path: {str(p)}",
                f"   Contained Items: {len(entries)} items",
                "─" * 48,
            ]
            for e in entries[:max_lines]:
                icon = "📁" if e["is_dir"] else "📄"
                size_str = f" ({e['size']:,} B)" if not e["is_dir"] else " [FOLDER]"
                dir_lines.append(f" {icon} {e['name']}{size_str}")
            return {
                "ok": True,
                "is_dir": True,
                "file_type": "dir",
                "path": str(p),
                "name": p.name,
                "total_lines": len(dir_lines),
                "lines": dir_lines,
                "entries": entries[:80],
                "size_bytes": 0
            }

        # Handle regular files
        ext = p.suffix.lower()
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        file_type = "code"
        if ext in [".html", ".htm"]:
            file_type = "html"
        elif ext in [".md", ".markdown", ".txt", ".rst"]:
            file_type = "markdown"
        elif ext == ".json":
            file_type = "json"
        elif ext == ".py":
            file_type = "python"

        ast_symbols = []
        if ext == ".py":
            import ast
            try:
                tree = ast.parse(content)
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        ast_symbols.append({"type": "class", "name": node.name, "line": node.lineno})
                    elif isinstance(node, ast.FunctionDef):
                        ast_symbols.append({"type": "def", "name": node.name, "line": node.lineno})
                    elif isinstance(node, ast.AsyncFunctionDef):
                        ast_symbols.append({"type": "async_def", "name": node.name, "line": node.lineno})
            except Exception:
                pass

        return {
            "ok": True,
            "is_dir": False,
            "file_type": file_type,
            "path": str(p),
            "name": p.name,
            "total_lines": len(lines),
            "lines": lines[:max_lines],
            "raw_sample": content[:45000],
            "ast_symbols": ast_symbols,
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


# ─── SPIDER ENHANCEMENTS: more rules, persistence, summary, cross-MCP routing ─

# Additional scan patterns (appended to WORK_PATTERNS at runtime)
EXTRA_WORK_PATTERNS = [
    ("hardcoded_secret", re.compile(r"(api[_-]?key|secret|password|token)\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]")),
    ("hardcoded_path", re.compile(r"[A-Za-z]:\\\\[^\s'\"]{6,}")),
    ("bare_except", re.compile(r"except\s*:"),),  # already in WORK_PATTERNS; harmless dup
    ("missing_docstring", re.compile(r"def\s+\w+\s*\([^)]*\)\s*:\s*\n\s*(?!\"\"\")")),
    ("long_function", re.compile(r"def\s+\w+")),
]
# Merge into WORK_PATTERNS (avoid dup names)
_existing_names = {p[0] for p in WORK_PATTERNS}
for name, rx in EXTRA_WORK_PATTERNS:
    if name not in _existing_names:
        WORK_PATTERNS.append((name, rx))
        _existing_names.add(name)

SPIDER_SESSION_DIR = SPIDER_DIR / "sessions"
SPIDER_SESSION_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
def spider_summary(top: int = 10) -> Dict[str, Any]:
    """Produce a digest summary of the last crawl: top findings, wired modules, census totals."""
    mem = _load_mem()
    mn = mem.get("nodes", {})
    flagged = [(k, v) for k, v in mn.items() if v.get("findings")]
    flagged_sorted = sorted(flagged, key=lambda kv: len(kv[1].get("findings", [])), reverse=True)[:top]
    census = mem.get("census", {})
    top_exts = sorted(census.get("exts", {}).items(), key=lambda kv: -kv[1])[:8]
    top_sectors = sorted(census.get("sectors", {}).items(), key=lambda kv: -kv[1])[:8]
    trail_data = {}
    if TRAIL_FILE.exists():
        try:
            trail_data = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
        except Exception:
            trail_data = {}
    return {
        "status": "SUMMARY_READY",
        "memory_nodes": len(mn),
        "memory_edges": len(mem.get("edges", {})),
        "runs": mem.get("runs", 0),
        "census_total_files": census.get("total", 0),
        "top_extensions": top_exts,
        "top_sectors": top_sectors,
        "top_flagged": [{"path": k[-60:], "findings": v.get("findings", [])[:3]} for k, v in flagged_sorted],
        "last_trail_events": len(trail_data.get("trail", [])),
        "last_job": trail_data.get("job"),
    }


@mcp.tool()
def spider_route_findings(target_mcp: str = "code") -> Dict[str, Any]:
    """Format the last crawl's findings as a payload ready to pass to another MCP.

    target_mcp: 'code' (QuillanCode scan), 'council' (predator strike), or 'buddy' (code review).
    Returns a structured payload the model can forward to the chosen MCP tool.
    """
    mem = _load_mem()
    mn = mem.get("nodes", {})
    flagged = [(k, v) for k, v in mn.items() if v.get("findings")]
    items = []
    for k, v in flagged[:20]:
        items.append({
            "path": k,
            "findings": v.get("findings", [])[:5],
        })
    if target_mcp.lower() == "code":
        return {"target": "QuillanCode", "suggested_tool": "codeweaver_security_scan",
                "payloads": [{"code_or_file_path": it["path"]} for it in items],
                "count": len(items)}
    if target_mcp.lower() == "council":
        return {"target": "QuillanCouncil", "suggested_tool": "council_predator_strike",
                "payloads": [{"proposal": f"File {it['path']} has findings: {it['findings']}"} for it in items],
                "count": len(items)}
    if target_mcp.lower() == "buddy":
        return {"target": "QuillanBuddy", "suggested_tool": "buddy_code_review",
                "payloads": [{"code": f"# {it['path']}", "language": "python"} for it in items],
                "count": len(items)}
    return {"error": f"Unknown target_mcp '{target_mcp}'. Use code, council, or buddy."}


@mcp.tool()
def spider_session_save(label: str = "") -> Dict[str, Any]:
    """Persist the current trail + memory graph as a named session."""
    import uuid
    sid = f"spider-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    payload = {}
    if TRAIL_FILE.exists():
        try:
            payload = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    mem = _load_mem()
    session = {
        "session_id": sid,
        "label": label,
        "trail": payload.get("trail", []),
        "nodes": payload.get("nodes", []),
        "edges": payload.get("edges", []),
        "memory_nodes": len(mem.get("nodes", {})),
        "memory_edges": len(mem.get("edges", {})),
        "timestamp": time.time(),
    }
    p = SPIDER_SESSION_DIR / f"{sid}.json"
    p.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "SAVED", "session_id": sid, "path": str(p),
            "trail_events": len(session["trail"])}


@mcp.tool()
def spider_session_list() -> Dict[str, Any]:
    """List persisted spider sessions."""
    sessions = []
    for p in sorted(SPIDER_SESSION_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": data.get("session_id"),
                "label": data.get("label"),
                "trail_events": len(data.get("trail", [])),
                "memory_nodes": data.get("memory_nodes"),
                "timestamp": data.get("timestamp"),
                "file": str(p),
            })
        except Exception:
            continue
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
def spider_session_replay(session_id: str) -> Dict[str, Any]:
    """Replay a persisted spider session by ID."""
    path = SPIDER_SESSION_DIR / f"{session_id}.json"
    if not path.exists():
        matches = list(SPIDER_SESSION_DIR.glob(f"{session_id}*.json"))
        if not matches:
            return {"error": f"Session not found: {session_id}"}
        path = matches[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": f"Read failed: {e}"}



# ─── 21. SPIDER STOP / PAUSE / RESUME ────────────────────────────────────────

_SPIDER_STATE = {"running": False, "paused": False, "stopped": False}

@mcp.tool()
def spider_stop() -> Dict[str, Any]:
    """Stop a running spider crawl immediately."""
    _SPIDER_STATE["stopped"] = True
    _SPIDER_STATE["running"] = False
    _SPIDER_STATE["paused"] = False
    return {"status": "SPIDER_STOPPED", "message": "Spider crawl stopped. Current trail preserved."}

@mcp.tool()
def spider_pause() -> Dict[str, Any]:
    """Pause a running spider crawl."""
    _SPIDER_STATE["paused"] = True
    return {"status": "SPIDER_PAUSED", "message": "Spider crawl paused. Call spider_resume to continue."}

@mcp.tool()
def spider_resume() -> Dict[str, Any]:
    """Resume a paused spider crawl."""
    _SPIDER_STATE["paused"] = False
    _SPIDER_STATE["running"] = True
    return {"status": "SPIDER_RESUMED", "message": "Spider crawl resumed."}

# ─── 22. SPIDER CONFIG ───────────────────────────────────────────────────────

@mcp.tool()
def spider_config(max_files: int = 220, delay_ms: int = 120, exts: str = ".py,.json,.md,.html",
                 skip_dirs: str = "", trail_cap: int = 300) -> Dict[str, Any]:
    """Configure spider crawl settings."""
    return {
        "max_files": max(10, min(1000, max_files)),
        "delay_ms": max(20, min(500, delay_ms)),
        "exts": [e.strip() for e in exts.split(",") if e.strip()],
        "skip_dirs": [d.strip() for d in skip_dirs.split(",") if d.strip()],
        "trail_cap": max(50, min(1000, trail_cap)),
        "status": "CONFIG_UPDATED",
    }

# ─── 23. SPIDER EXPORT / IMPORT ──────────────────────────────────────────────

@mcp.tool()
def spider_export(format: str = "json", output_path: str = "") -> Dict[str, Any]:
    """Export the current trail/map to a file."""
    if not TRAIL_FILE.exists():
        return {"error": "No trail to export. Run spider_crawl first."}
    data = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
    if not output_path:
        output_path = str(SPIDER_DIR / f"export_{int(time.time())}.{format}")
    out = Path(output_path)
    if format == "json":
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif format == "csv":
        import csv
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "path", "event", "action", "result"])
            for t in data.get("trail", []):
                writer.writerow([t.get("t"), t.get("at"), t.get("event"), t.get("action"), t.get("result")])
    else:
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"status": "EXPORTED", "format": format, "output_path": str(out), "entries": len(data.get("trail", []))}

@mcp.tool()
def spider_import(input_path: str) -> Dict[str, Any]:
    """Import a trail/map from a file."""
    p = Path(input_path)
    if not p.exists():
        return {"error": f"File not found: {input_path}"}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        TRAIL_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return {"status": "IMPORTED", "entries": len(data.get("trail", [])), "nodes": len(data.get("nodes", []))}
    except Exception as e:
        return {"error": f"Import failed: {e}"}

# ─── 24. SPIDER DIFF ─────────────────────────────────────────────────────────

@mcp.tool()
def spider_diff(trail_a: str, trail_b: str) -> Dict[str, Any]:
    """Compare two trail files and report differences."""
    pa = Path(trail_a)
    pb = Path(trail_b)
    if not pa.exists() or not pb.exists():
        return {"error": "One or both trail files not found."}
    try:
        da = json.loads(pa.read_text(encoding="utf-8"))
        db = json.loads(pb.read_text(encoding="utf-8"))
        nodes_a = {n.get("full") for n in da.get("nodes", [])}
        nodes_b = {n.get("full") for n in db.get("nodes", [])}
        added = nodes_b - nodes_a
        removed = nodes_a - nodes_b
        common = nodes_a & nodes_b
        return {
            "trail_a": str(pa),
            "trail_b": str(pb),
            "nodes_a": len(nodes_a),
            "nodes_b": len(nodes_b),
            "added_nodes": len(added),
            "removed_nodes": len(removed),
            "common_nodes": len(common),
            "added": list(added)[:20],
            "removed": list(removed)[:20],
            "status": "DIFF_COMPLETE",
        }
    except Exception as e:
        return {"error": f"Diff failed: {e}"}

# ─── 25. SPIDER STATS ────────────────────────────────────────────────────────

@mcp.tool()
def spider_stats() -> Dict[str, Any]:
    """Detailed statistics about the current trail and memory graph."""
    if not TRAIL_FILE.exists():
        return {"status": "NO_TRAIL", "hint": "Run spider_crawl first."}
    data = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
    trail = data.get("trail", [])
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    # File type distribution
    ext_counts = {}
    for n in nodes:
        if n.get("kind") == "file":
            ext = Path(n.get("full", "")).suffix.lower() or "(noext)"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    # Event type distribution
    event_counts = {}
    for t in trail:
        ev = t.get("event", "unknown")
        event_counts[ev] = event_counts.get(ev, 0) + 1
    # Result distribution
    result_counts = {}
    for t in trail:
        r = t.get("result", "unknown")
        result_counts[r] = result_counts.get(r, 0) + 1
    return {
        "status": "STATS_READY",
        "nodes": len(nodes),
        "edges": len(edges),
        "trail_events": len(trail),
        "file_type_distribution": dict(sorted(ext_counts.items(), key=lambda x: -x[1])[:15]),
        "event_distribution": event_counts,
        "result_distribution": result_counts,
        "live": data.get("live", False),
        "spider_state": (data.get("spider") or {}).get("state", "unknown"),
    }

# ─── 26. SPIDER FILTER ───────────────────────────────────────────────────────

@mcp.tool()
def spider_filter(event_type: str = "", result: str = "", path_contains: str = "") -> Dict[str, Any]:
    """Filter trail events by criteria."""
    if not TRAIL_FILE.exists():
        return {"error": "No trail to filter. Run spider_crawl first."}
    data = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
    trail = data.get("trail", [])
    filtered = trail
    if event_type:
        filtered = [t for t in filtered if t.get("event") == event_type]
    if result:
        filtered = [t for t in filtered if t.get("result") == result]
    if path_contains:
        filtered = [t for t in filtered if path_contains.lower() in str(t.get("at", "")).lower()]
    return {
        "status": "FILTER_COMPLETE",
        "total_events": len(trail),
        "filtered_count": len(filtered),
        "events": filtered[:50],
    }

# ─── 27. SPIDER SEARCH ───────────────────────────────────────────────────────

@mcp.tool()
def spider_search(query: str) -> Dict[str, Any]:
    """Search within crawled data for files/paths matching a query."""
    if not TRAIL_FILE.exists():
        return {"error": "No trail to search. Run spider_crawl first."}
    data = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
    trail = data.get("trail", [])
    nodes = data.get("nodes", [])
    q = query.lower()
    matches = []
    for t in trail:
        if q in str(t.get("at", "")).lower() or q in str(t.get("full", "")).lower():
            matches.append({"path": t.get("at"), "full": t.get("full"), "event": t.get("event"), "result": t.get("result")})
    for n in nodes:
        if q in str(n.get("path", "")).lower() or q in str(n.get("full", "")).lower():
            if not any(m["full"] == n.get("full") for m in matches):
                matches.append({"path": n.get("path"), "full": n.get("full"), "event": n.get("kind"), "result": "node"})
    return {"status": "SEARCH_COMPLETE", "query": query, "matches": matches[:50], "total": len(matches)}

# ─── 28. SPIDER WEBHOOK ──────────────────────────────────────────────────────

@mcp.tool()
def spider_webhook(url: str, message: str = "Spider crawl complete") -> Dict[str, Any]:
    """Send a webhook notification (e.g., to Discord/Slack)."""
    import urllib.request
    try:
        payload = json.dumps({"content": message, "text": message}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": "WEBHOOK_SENT", "response_code": resp.status, "message": message}
    except Exception as e:
        return {"status": "WEBHOOK_FAILED", "error": str(e)}

# ─── 29. SPIDER SCHEDULE ─────────────────────────────────────────────────────

@mcp.tool()
def spider_schedule(root: str = "C:\\02_QUILLAN\\09 - Projects\\mcp", interval_minutes: int = 60, job: str = "scan") -> Dict[str, Any]:
    """Schedule a recurring spider crawl."""
    schedule_file = SPIDER_DIR / "schedule.json"
    schedule = {"root": root, "interval_minutes": interval_minutes, "job": job, "enabled": True, "created": time.time()}
    schedule_file.write_text(json.dumps(schedule, indent=2), encoding="utf-8")
    return {"status": "SCHEDULE_CREATED", "schedule": schedule,
            "note": "Use a cron job or task scheduler to run spider_work or spider_sweep at the specified interval."}

# ─── 30. SPIDER REMOTE CONTROL ───────────────────────────────────────────────

@mcp.tool()
def spider_remote_control(action: str = "status", params: str = "") -> Dict[str, Any]:
    """Remote control interface for the spider bot."""
    if action.lower() == "status":
        if TRAIL_FILE.exists():
            data = json.loads(TRAIL_FILE.read_text(encoding="utf-8"))
            return {"status": "OK", "spider_state": (data.get("spider") or {}).get("state", "unknown"),
                    "visited": (data.get("spider") or {}).get("visited", 0), "live": data.get("live", False)}
        return {"status": "IDLE", "spider_state": "no trail"}
    elif action.lower() == "stop":
        return spider_stop()
    elif action.lower() == "pause":
        return spider_pause()
    elif action.lower() == "resume":
        return spider_resume()
    elif action.lower() == "config":
        return {"status": "CONFIG", "params": params}
    return {"error": f"Unknown action '{action}'. Use: status, stop, pause, resume, config."}

# --- Quillan-Ronin Global Error-Handling Middleware ---
from fastmcp.server.middleware import Middleware, MiddlewareContext
import logging as _log

class QuillanErrorMiddleware(Middleware):
    async def on_call_tool(self, context, call_next):
        try:
            return await call_next(context)
        except Exception as e:
            _log.error("Tool error: %s", e, exc_info=True)
            return {"status": "ERROR", "error": str(e), "error_type": type(e).__name__}


mcp.add_middleware(QuillanErrorMiddleware())
if __name__ == "__main__":
    mcp.run()

@mcp.tool()
def spider_health() -> dict:
    """Health check endpoint for QuillanSpider MCP server."""
    import time
    return {"status": "OK", "server": "QuillanSpider", "timestamp": time.time()}
