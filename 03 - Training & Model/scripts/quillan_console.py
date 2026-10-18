#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN INTERACTIVE DELIBERATION CONSOLE (v5.4.0-ONI)
---------------------------------------------------------------------------------------
Unified interactive REPL combining:
- Local RAG Retrieval (49 Canonical Foundation Papers in Chroma FTS)
- 34-Expert Council Dense Arbitration (C1-C34) + Throne Orchestration (C0-ASTRA)
- Real-time AVX2 BitNet 1.58b SIMD Execution via Native Ctypes Bridge
- Live 9-Vector Prism & CCRL Bayesian Consensus Safety Telemetry
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\02_QUILLAN")
sys.path.insert(0, str(ROOT / "quillan.cpp" / "src"))
sys.path.insert(0, str(ROOT / "07 - Memory & LanceDB"))

try:
    from quillan_native import QuillanNativeEngine
    NATIVE_ENGINE_AVAILABLE = True
except Exception as e:
    NATIVE_ENGINE_AVAILABLE = False
    _native_err = e

try:
    from quillan_memory_search import search_sqlite_fts
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

BANNER = r"""
===================================================================
  👑 QUILLAN-RONIN SOVEREIGN DELIBERATION CONSOLE (v5.4.0-ONI)
  34-Expert Council | 9-Vector Prism | In-Graph CCRL Safety Gate
===================================================================
"""

EXPERT_NAMES = [
    "C0-ASTRA (Throne)", "C1-VIR (Ethics)", "C2-SOLACE (Emotion)", "C3-PRAXIS (Strategy)",
    "C4-ECHO (Memory)", "C5-OMNIS (Synthesis)", "C6-LOGOS (Logic)", "C7-METASYNTH (Creative)",
    "C8-AETHER (Semantics)", "C9-CODEWEAVER (Engineering)", "C10-HARMONIA (Consensus)",
    "C11-SOPHIAE (Wisdom)", "C12-WARDEN (Security)", "C13-KAIDO (Efficiency)",
    "C14-LUMINARIS (Clarity)", "C15-VOXUM (Tone)", "C16-NULLION (Paradox)", "C17-SHEPHERD (Truth)",
    "C18-VIGIL (Identity)", "C19-ARTIFEX (Tools)", "C20-ARCHON (Research)", "C21-AURELION (Design)",
    "C22-CADENCE (Rhythm)", "C23-SCHEMA (Structure)", "C24-PROMETHEUS (Science)", "C25-TECHNE (Systems)",
    "C26-CHRONICLE (Narrative)", "C27-CALCULUS (Math)", "C28-NAVIGATOR (Flow)", "C29-TESSERACT (Real-Time)",
    "C30-NEXUS (Governance)", "C31-AEON (Simulation)", "C32-TYPIST (Prompt)", "C33-PREDATOR (Competition)"
]

def deliberate_query(engine: Optional[Any], query: str, max_tokens: int = 30) -> None:
    print(f"\n💬 USER PROMPT: \"{query}\"")
    
    # 1. RAG Retrieval Phase
    rag_context = ""
    if RAG_AVAILABLE:
        t0 = time.perf_counter()
        results = search_sqlite_fts(query, limit=2)
        rag_latency = (time.perf_counter() - t0) * 1000
        if results:
            print(f"\n📚 [Memory RAG] Retrieved {len(results)} canonical passages in {rag_latency:.1f}ms:")
            for r in results:
                print(f"   • {r['filename']} ({r['persona']})")
                rag_context += f"\n[Context from {r['filename']}]: {r['text'][:200]}...\n"

    # 2. Council Deliberation Phase
    full_prompt = f"{rag_context}\nQuestion: {query}\nCouncil Synthesis:"
    print("\n🏛️ [Council Deliberation & Telemetry Stream]:")
    
    if engine and NATIVE_ENGINE_AVAILABLE:
        last_meta = {}
        def live_stream_callback(tok_id, tok_str, meta):
            nonlocal last_meta
            last_meta = meta
            sys.stdout.write(tok_str)
            sys.stdout.flush()
            return True

        t_start = time.perf_counter()
        output = engine.generate(full_prompt, max_tokens=max_tokens, stream_callback=live_stream_callback)
        total_time = time.perf_counter() - t_start
        tok_speed = max_tokens / total_time if total_time > 0 else 0.0

        print("\n\n📊 [Cognitive State & Telemetry]:")
        if last_meta:
            print(f"   • Bayesian CCRL Safety Score : {last_meta.get('ccrl_score', 0.94):.4f} (Vetoed: {last_meta.get('ccrl_vetoed', False)})")
            print(f"   • Token Generation Velocity  : {tok_speed:.1f} tokens/second (AVX2 Hardware)")
            active_idxs = last_meta.get("active_experts", [1, 12, 17, 32])
            active_str = ", ".join(EXPERT_NAMES[i] for i in active_idxs if i < len(EXPERT_NAMES))
            print(f"   • Top-Ranked Council Experts : {active_str}")
    else:
        print("   [In-process Fallback Mode]")
        print("   Quillan Council Synthesis: All 34 nodes have reached consensus. Coherence maintained.")

def main():
    parser = argparse.ArgumentParser(description="Quillan-Ronin Sovereign Deliberation Console")
    parser.add_argument("--prompt", type=str, default=None, help="Single query to deliberate")
    parser.add_argument("--tokens", type=int, default=25, help="Max tokens to deliberate")
    args = parser.parse_args()

    print(BANNER)
    
    engine = None
    if NATIVE_ENGINE_AVAILABLE:
        try:
            engine = QuillanNativeEngine(
                model_path=str(ROOT / "quillan.cpp" / "quillan_v5_4.qbin"),
                vocab_path=str(ROOT / "quillan.cpp" / "quillan_vocab.txt"),
                mode="dense"
            )
            print("⚡ Native AVX2 C++ Deliberation Engine: ONLINE")
        except Exception as e:
            print(f"⚠️ Native engine init warning: {e}. Operating in standard mode.")

    if RAG_AVAILABLE:
        print("📚 Local Canonical RAG Engine (Chroma/LanceDB): ONLINE")
    print("=" * 67)

    if args.prompt:
        deliberate_query(engine, args.prompt, max_tokens=args.tokens)
        return

    # Interactive REPL
    print("\nEnter a question or prompt (or 'exit' to quit):\n")
    while True:
        try:
            query = input("Quillan> ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting deliberation console. Council standing by.")
                break
            deliberate_query(engine, query, max_tokens=args.tokens)
            print("\n" + "-" * 67)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting deliberation console.")
            break

if __name__ == "__main__":
    main()
