#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin v5.4.0-ONI -- Deterministic 6-Stage MCP Chain Bus
=====================================================================
Chains the Quillan MCP servers into one ordered pipeline instead of leaving
them as isolated tools. Every run carries a ``correlation_id`` and appends to a
durable state ledger so a run can be reconstructed after the fact.

    STAGE 1  Intake & Sharding      Prism.shard_9_vectors + RAG.retrieve_context
    STAGE 2  Thermodynamic Pacing   Governor.compute_venturi_pid
                                     -> fast-path (conf>=0.80, dP<=0.15)
                                     -> deep-path -> SequentialThinking
    STAGE 3  Council Deliberation   Council.deliberate_34 + ThinkingEngine gates
    STAGE 4  Actuation              route by task class to the actuation group
    STAGE 5  RCI Self-Healing       SamuraiEngine.audit_and_rci_loop (max 5)
    STAGE 6  Persistence            RAG.store_trace + ledger flush

DESIGN NOTES
------------
* The bus is transport-agnostic. Servers may be attached in-process (the
  default, used by the local MCP servers) or reached over stdio JSON-RPC.
  ``register()`` accepts either.
* A stage failure is recorded and, per the C4-PRAXIS completion rule, the run
  continues where it safely can -- unless ``strict=True``. Silent success on a
  broken stage is exactly what this bus exists to prevent, so every stage
  outcome lands in the ledger either way.
* No network calls, no shell-outs, no credential access. Warden-gated.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# --------------------------------------------------------------------------
# stdio hygiene MUST happen before anything else touches stdout. Importing an
# MCP module that prints an ASCII banner would otherwise corrupt the JSON-RPC
# stream for any server launched in-process alongside this one.
# --------------------------------------------------------------------------
import logging

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s | chain-bus | %(levelname)s | %(message)s",
)

ONION_VERSION = "5.4.0-oni"
CHAIN_VERSION = "5.4.0-oni"

REPO_ROOT = Path(r"C:\02_QUILLAN")
LEDGER_PATH = REPO_ROOT / "07 - Memory & LanceDB" / "chain_state.json"

# Canonical PYTHONPATH for every Quillan MCP child process.
CANONICAL_PYTHONPATH = os.pathsep.join([
    str(REPO_ROOT / "09 - Projects" / "mcp"),
    str(REPO_ROOT / "09 - Projects" / "projects" / "oni"),
    str(REPO_ROOT / "03 - Training & Model" / "scripts"),
    str(REPO_ROOT),
])

WINDOWS_PYTHON = r"C:\Users\Admin\AppData\Local\Programs\Python\Python314\python.exe"

# Stage 2 pacing thresholds (directive spec).
FAST_PATH_CONF = 0.80
FAST_PATH_DELTA_P = 0.15
RCI_MAX_RETRIES = 5


# ==========================================================================
# 1. CROSS-PLATFORM RESOLVER (Quirk: Devin runs Linux/WSL)
# ==========================================================================
def wsl_path(p: str | Path) -> str:
    r"""/mnt/c/... form for a Windows path (Devin/WSL)."""
    s = str(p).replace("\\", "/")
    if len(s) > 1 and s[1] == ":":
        drive = s[0].lower()
        return "/mnt/" + drive + s[2:]
    return s


def resolve_python() -> str:
    """Windows python.exe, else python3, else WSL shim."""
    if platform.system() == "Windows":
        if os.path.exists(WINDOWS_PYTHON):
            return WINDOWS_PYTHON
        for exe in ("python", "python3", "py"):
            from shutil import which
            hit = which(exe)
            if hit:
                return hit
    for exe in ("python3", "python"):
        from shutil import which
        hit = which(exe)
        if hit:
            return hit
    return WINDOWS_PYTHON if platform.system() == "Windows" else "python3"


def pythonpath_for(platform_name: str | None = None) -> str:
    """Windows-native or WSL-translated PYTHONPATH, matching the host."""
    if platform_name == "wsl":
        return os.pathsep.join([wsl_path(p) for p in CANONICAL_PYTHONPATH.split(os.pathsep)])
    return CANONICAL_PYTHONPATH


# ==========================================================================
# 2. STATE LEDGER
# ==========================================================================
@dataclass
class StageRecord:
    stage: int
    name: str
    status: str                      # ok | failed | skipped
    started_at: str
    finished_at: str = ""
    duration_ms: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ChainLedger:
    """Append-only JSON ledger keyed by correlation_id.

    Writes are atomic (tmp file + os.replace) so a crash mid-write cannot
    corrupt the whole ledger.
    """

    def __init__(self, path: Path = LEDGER_PATH):
        self.path = Path(path)
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._runs = raw.get("runs", {}) or {}
        except Exception as exc:                                  # pragma: no cover
            logging.warning("ledger load failed (%s); starting empty", exc)
            self._runs = {}

    def start(self, correlation_id: str, query: str) -> Dict[str, Any]:
        run = {
            "correlation_id": correlation_id,
            "query": query,
            "version": CHAIN_VERSION,
            "started_at": _now(),
            "finished_at": None,
            "governor_mode": None,
            "stages": [],
            "result": None,
        }
        self._runs[correlation_id] = run
        self._dirty = True
        return run

    def append(self, correlation_id: str, rec: StageRecord) -> None:
        run = self._runs.get(correlation_id)
        if run is None:
            return
        run["stages"].append(asdict(rec))
        self._dirty = True

    def finish(self, correlation_id: str, result: Any = None) -> None:
        run = self._runs.get(correlation_id)
        if run is None:
            return
        run["finished_at"] = _now()
        run["result"] = result
        self._dirty = True
        self.flush()

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"version": CHAIN_VERSION, "runs": self._runs},
                           indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception as exc:                                  # pragma: no cover
            logging.error("ledger flush failed: %s", exc)

    def get(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        return self._runs.get(correlation_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_correlation_id(query: str = "") -> str:
    """Deterministic prefix + unique suffix. Same query -> traceable prefix."""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8] if query else "00000000"
    return f"q-{digest}-{uuid.uuid4().hex[:12]}"


# ==========================================================================
# 3. REGISTRY
# ==========================================================================
class ChainBus:
    """Ordered 6-stage pipeline over attached MCP handlers.

    Handlers are plain callables ``fn(payload: dict) -> dict``. That keeps the
    bus decoupled from any particular transport: wrap an MCP stdio client, an
    in-process server module, or a stub in one line.
    """

    def __init__(self, ledger: Optional[ChainLedger] = None, strict: bool = False):
        self.ledger = ledger or ChainLedger()
        self.strict = strict
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    # -- registration -----------------------------------------------------
    def register(self, name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self._handlers[name] = fn
        return self

    def has(self, name: str) -> bool:
        return name in self._handlers

    @property
    def registered(self) -> List[str]:
        return sorted(self._handlers)

    # -- stage runner -----------------------------------------------------
    def _run_stage(self, cid: str, num: int, name: str,
                   fn_name: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
        rec = StageRecord(stage=num, name=name, status="skipped", started_at=_now())
        t0 = time.perf_counter()
        if fn_name is None or fn_name not in self._handlers:
            rec.status = "skipped"
            rec.detail = {"reason": f"handler '{fn_name}' not registered"}
            rec.finished_at = _now()
            rec.duration_ms = (time.perf_counter() - t0) * 1000
            self.ledger.append(cid, rec)
            return {"_stage": num, "_name": name, "_status": "skipped"}

        try:
            out = self._handlers[fn_name](payload) or {}
            rec.status = "ok"
            rec.detail = out if isinstance(out, dict) else {"result": out}
        except Exception as exc:
            rec.status = "failed"
            rec.error = f"{type(exc).__name__}: {exc}"
            rec.detail = {"traceback": traceback.format_exc(limit=3)}
            logging.error("stage %d (%s) failed: %s", num, name, exc)
            self.ledger.append(cid, rec)
            if self.strict:
                raise
            return {"_stage": num, "_name": name, "_status": "failed", "_error": rec.error}

        rec.finished_at = _now()
        rec.duration_ms = (time.perf_counter() - t0) * 1000
        self.ledger.append(cid, rec)
        out = dict(out) if isinstance(out, dict) else {"result": out}
        out.update({"_stage": num, "_name": name, "_status": "ok"})
        return out

    # ======================================================================
    # THE PIPELINE
    # ======================================================================
    def run(self, query: str, task_class: str = "general",
            correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute all 6 stages. Returns the trace + final payload."""
        cid = correlation_id or new_correlation_id(query)
        self.ledger.start(cid, query)
        run = self.ledger.get(cid) or {}

        ctx: Dict[str, Any] = {
            "correlation_id": cid,
            "query": query,
            "task_class": task_class,
            "version": CHAIN_VERSION,
        }
        trace: List[Dict[str, Any]] = []

        # ---- STAGE 1 : intake & sharding -------------------------------
        s1 = self._run_stage(cid, 1, "intake_and_sharding", "QuillanPrism.shard_9_vectors", ctx)
        ctx["prism_state"] = s1.get("prism_state", s1)
        ctx["rag_context"] = self._run_stage(
            cid, 1, "rag_retrieval", "QuillanRAG.retrieve_context", ctx)
        trace.append(s1)

        # ---- STAGE 2 : pacing & risk gate -------------------------------
        s2 = self._run_stage(cid, 2, "thermodynamic_pacing", "QuillanGovernor.compute_venturi_pid", ctx)
        conf = float(s2.get("confidence", 0.0) or 0.0)
        dP = float(s2.get("venturi_delta_p", 0.0) or 0.0)
        fast_path = conf >= FAST_PATH_CONF and dP <= FAST_PATH_DELTA_P
        mode = "fast" if fast_path else "deep"
        ctx["governor_mode"] = mode
        ctx["venturi_delta_p"] = dP
        run["governor_mode"] = mode
        s2.update({"_mode": mode, "_conf": conf, "_delta_p": dP})
        trace.append(s2)

        if not fast_path:
            st = self._run_stage(cid, 2, "deep_path_sequential_thinking",
                                 "SequentialThinking.reason", ctx)
            ctx["deep_reasoning"] = st
            trace.append(st)

        # ---- STAGE 3 : council deliberation -----------------------------
        s3 = self._run_stage(cid, 3, "council_deliberation_34",
                             "QuillanCouncil.deliberate_34", {**ctx, "governor_mode": mode})
        ctx["council"] = s3
        trace.append(s3)

        gates = self._run_stage(cid, 3, "gates_wot",
                                "ThinkingEngine.run_wot_and_gates",
                                {"correlation_id": cid, "query": query,
                                 "governor_mode": mode, "council": s3})
        ctx["gates"] = gates
        trace.append(gates)

        # ---- STAGE 4 : actuation ----------------------------------------
        s4 = self._run_stage(cid, 4, "actuation", _ACTUATOR.get(task_class), {**ctx, "task_class": task_class})
        ctx["actuation"] = s4
        trace.append(s4)

        # ---- STAGE 5 : RCI self-healing ---------------------------------
        s5 = self._run_stage(cid, 5, "rci_self_healing",
                             "SamuraiEngine.audit_and_rci_loop",
                             {"correlation_id": cid, "draft_result": s4,
                              "max_retries": RCI_MAX_RETRIES})
        ctx["audit"] = s5
        trace.append(s5)

        # ---- STAGE 6 : persistence & telemetry --------------------------
        s6 = self._run_stage(cid, 6, "persistence_telemetry",
                             "QuillanRAG.store_trace",
                             {"correlation_id": cid, "trace": trace, "query": query})
        trace.append(s6)

        self.ledger.finish(cid, {"governor_mode": mode, "trace_len": len(trace)})
        return {
            "correlation_id": cid,
            "version": CHAIN_VERSION,
            "governor_mode": mode,
            "stages": trace,
            "ledger_path": str(self.ledger.path),
            "context": ctx,
        }


# Stage-4 task-class -> handler name.
_ACTUATOR = {
    "code": "QuillanCode.execute_task",
    "repo": "QuillanCode.execute_task",
    "ml": "QuillanModelLab.execute_task",
    "training": "QuillanModelLab.execute_task",
    "web": "WebSearch.search",
    "research": "Fetch.fetch_url",
    "desktop": "ComputerUse.act",
    "os": "ComputerUse.act",
    "general": None,
}


# ==========================================================================
# 4. DEFAULT BUS (in-process handlers over the local MCP modules)
# ==========================================================================
def build_default_bus(strict: bool = False) -> ChainBus:
    """Attach the real MCP servers where importable; degrade gracefully."""
    bus = ChainBus(strict=strict)

    def _maybe(module_path: str, attr: str, tool: str, wrap=None):
        try:
            mod = __import__(module_path, fromlist=[attr])
            fn = getattr(mod, attr)
            bus.register(tool, wrap(fn) if wrap else fn)
            return True
        except Exception as exc:
            logging.warning("MCP '%s' unavailable (%s) -- stage will skip", tool, exc)
            return False

    _maybe("quillan_prism.server", "shard_9_vectors", "QuillanPrism.shard_9_vectors")
    _maybe("quillan_governor.server", "compute_venturi_pid", "QuillanGovernor.compute_venturi_pid")
    _maybe("quillan_council.server", "deliberate_34", "QuillanCouncil.deliberate_34")
    _maybe("quillan_thinking_engine.server", "run_wot_and_gates", "ThinkingEngine.run_wot_and_gates")
    _maybe("quillan_samurai_engine.server", "audit_and_rci_loop", "SamuraiEngine.audit_and_rci_loop")
    _maybe("quillan_code.server", "execute_task", "QuillanCode.execute_task")
    _maybe("quillan_model_lab.server", "execute_task", "QuillanModelLab.execute_task")
    _maybe("quillan_rag.server", "retrieve_context", "QuillanRAG.retrieve_context")
    _maybe("quillan_rag.server", "store_trace", "QuillanRAG.store_trace")

    return bus


def run_quillan_pipeline(query: str, task_class: str = "general",
                         correlation_id: Optional[str] = None,
                         strict: bool = False) -> Dict[str, Any]:
    """Master entry point -- also exposed as a SamuraiEngine MCP tool."""
    bus = build_default_bus(strict=strict)
    out = bus.run(query, task_class=task_class, correlation_id=correlation_id)
    out["registered_handlers"] = bus.registered
    return out


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    q = sys.argv[1] if len(sys.argv) > 1 else "verify quillan chain bus"
    tc = sys.argv[2] if len(sys.argv) > 2 else "general"
    res = run_quillan_pipeline(q, task_class=tc)
    print(json.dumps({
        "correlation_id": res["correlation_id"],
        "governor_mode": res["governor_mode"],
        "stages": [
            {"n": s.get("_stage"), "name": s.get("_name"), "status": s.get("_status")}
            for s in res["stages"]
        ],
        "ledger": res["ledger_path"],
    }, indent=2))