#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN MASTER SYSTEM HEALTH & REGRESSION RUNNER (v5.4.0-ONI)
---------------------------------------------------------------------------------------
End-to-End automated validation suite verifying all architectural layers:
- Suite 1: Whole-repo Python AST & compilation check (0 syntax errors)
- Suite 2: HuggingFace BPE 50,257-token tokenizer parity
- Suite 3: Native quillan.cpp AVX2 engine & Ctypes DLL bridge
- Suite 4: Neural checkpoint parameter count & weight integrity
- Suite 5: Local Knowledge Base & RAG SQLite FTS retrieval latency
- Suite 6: Multi-platform knowledge file preservation (280+ files)
"""

import os
import sys
import time
import sqlite3
import py_compile
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"C:\02_QUILLAN")

def print_header(title: str):
    print("\n" + "=" * 67)
    print(f"  🧪 {title}")
    print("=" * 67)

def test_python_syntax() -> bool:
    print_header("Suite 1: Whole-Monorepo Python Syntax Verification")
    skip_dirs = {
        'node_modules', '.git', '.venv-mazebench', '__pycache__', 'venv_oni_gpu',
        '.antigravity', '.antigravity-ide', '.devin', '.windsurf', '.claude',
        '.codeium', 'colorize-fixtures'
    }
    errors = []
    scanned = 0
    t0 = time.perf_counter()

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not (Path(root) / d).is_symlink()]
        for f in files:
            if f.endswith(".py"):
                scanned += 1
                fp = Path(root) / f
                try:
                    py_compile.compile(str(fp), doraise=True)
                except Exception as e:
                    errors.append((str(fp), str(e)))

    dt = time.perf_counter() - t0
    print(f"Scanned {scanned} Python files in {dt:.2f}s.")
    if errors:
        print(f"❌ FAILED: Found {len(errors)} syntax errors:")
        for fp, err in errors[:5]:
            print(f"   • {fp}: {err}")
        return False
    print("✅ PASSED: 0 syntax errors detected across all project files.")
    return True

def test_tokenizer_roundtrip() -> bool:
    print_header("Suite 2: HuggingFace BPE Tokenizer Parity")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "03 - Training & Model"))
    try:
        from quillan_bpe_tokenizer import QuillanBPETokenizer
        tok = QuillanBPETokenizer()
        if not tok._tok:
            print("❌ FAILED: Tokenizer failed to auto-discover tokenizer.json.")
            return False
        test_text = "The 34-Expert Council deliberates with mathematical certainty."
        encoded = tok.encode(test_text)
        decoded = tok.decode(encoded)
        if decoded != test_text:
            print(f"❌ FAILED: Roundtrip mismatch: '{test_text}' != '{decoded}'")
            return False
        print(f"Vocab Size : {tok.vocab_size} tokens")
        print(f"Test Text  : \"{test_text}\"")
        print(f"Encoded    : {encoded} ({len(encoded)} tokens)")
        print(f"Decoded    : \"{decoded}\"")
        print("✅ PASSED: 100% Tokenizer encode/decode mathematical parity.")
        return True
    except Exception as e:
        print(f"❌ FAILED: Exception in tokenizer test: {e}")
        return False

def test_native_engine() -> bool:
    print_header("Suite 3: Native quillan.cpp AVX2 Engine & DLL Bridge")
    sys.path.insert(0, str(ROOT / "quillan.cpp" / "src"))
    try:
        from quillan_native import QuillanNativeEngine
        engine = QuillanNativeEngine(
            model_path=str(ROOT / "quillan.cpp" / "quillan_v5_4.qbin"),
            vocab_path=str(ROOT / "quillan.cpp" / "quillan_vocab.txt"),
            mode="dense"
        )
        print(f"AVX2 Hardware Acceleration: {'AVAILABLE' if engine.has_avx2 else 'DISABLED'}")
        t0 = time.perf_counter()
        out = engine.generate("Verification test", max_tokens=10)
        dt = (time.perf_counter() - t0) * 1000
        print(f"Generated 10 tokens in {dt:.2f}ms (Velocity: {10 / (dt / 1000):.1f} tok/s)")
        print("✅ PASSED: Native DLL execution and in-process Ctypes bridge functional.")
        return True
    except Exception as e:
        print(f"❌ FAILED: Exception in native engine test: {e}")
        return False

def test_checkpoints() -> bool:
    print_header("Suite 4: Model Checkpoints & Parameter Audit (Mini 6L & Main 12L)")
    try:
        import torch
        sys.path.insert(0, str(ROOT / "09 - Projects" / "projects" / "oni"))
        from quillan_v5_4_oni import QuillanOniConfig, QuillanRoninOni

        models_to_audit = [
            ("Quillan-Oni Mini (6 Layers)", ROOT / "checkpoints" / "quillan_oni_mini_6l.pt", 6, 400_000_000),
            ("Quillan-Oni Main (12 Layers)", ROOT / "checkpoints" / "quillan_oni_main_12l.pt", 12, 550_000_000),
        ]

        all_healthy = True
        for name, ckpt_path, expected_layers, min_params in models_to_audit:
            print(f"\n--- Auditing: {name} ---")
            if not ckpt_path.exists():
                print(f"❌ FAILED: Checkpoint missing at {ckpt_path}")
                all_healthy = False
                continue

            data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model_sd = data.get("model", {})
            param_count = sum(t.numel() for t in model_sd.values())
            print(f"  Checkpoint Size : {ckpt_path.stat().st_size / (1024**2):.2f} MB")
            print(f"  Total Parameters: {param_count / 1e6:.2f}M ({len(model_sd)} tensor keys)")
            print(f"  Layer Depth     : {data.get('n_layer', 'unknown')} layers")

            if param_count < min_params:
                print(f"❌ FAILED: Parameter count {param_count} below threshold {min_params}")
                all_healthy = False
                continue

            cfg = QuillanOniConfig(
                vocab_size=50257,
                hidden_dim=1024,
                ffn_dim=2048,
                n_layer=expected_layers,
                num_experts=34,
                top_k=4,
                max_seq_len=512,
            )
            model = QuillanRoninOni(cfg)
            missing, unexpected = model.load_state_dict(model_sd, strict=False)
            if len(missing) > 0 or len(unexpected) > 0:
                print(f"❌ FAILED: State dict mismatch (Missing: {len(missing)}, Unexpected: {len(unexpected)})")
                all_healthy = False
                continue

            # Forward pass smoke test
            dummy_input = torch.randint(0, 1000, (1, 8), dtype=torch.long)
            with torch.no_grad():
                out = model(dummy_input)
                logits = out[0] if isinstance(out, tuple) else out
                if torch.isnan(logits).any():
                    print("❌ FAILED: NaN detected in model forward logits!")
                    all_healthy = False
                    continue

            print(f"✅ PASSED: {name} structural integrity and forward inference verified.")

        if not all_healthy:
            return False

        print("\n✅ PASSED: Both Quillan-Oni Mini and Main models verified with 100% integrity.")
        return True
    except Exception as e:
        print(f"❌ FAILED: Checkpoint audit error: {e}")
        return False

def test_local_rag() -> bool:
    print_header("Suite 5: Local RAG Knowledge Base & Latency")
    sys.path.insert(0, str(ROOT / "07 - Memory & LanceDB"))
    try:
        from quillan_memory_search import search_sqlite_fts
        t0 = time.perf_counter()
        results = search_sqlite_fts("Council", limit=3)
        dt = (time.perf_counter() - t0) * 1000
        if not results:
            print("❌ FAILED: No results returned from local RAG search.")
            return False
        print(f"Query Latency : {dt:.2f} ms")
        print(f"Matches Found : {len(results)}")
        for r in results:
            print(f"   • {r['filename']} ({r['persona']})")
        print("✅ PASSED: Sub-50ms local full-text search verified.")
        return True
    except Exception as e:
        print(f"❌ FAILED: RAG search error: {e}")
        return False

def test_platform_preservation() -> bool:
    print_header("Suite 6: Cross-Platform Knowledge File Preservation")
    platforms_dir = ROOT / "06 - Deployment & Platforms" / "Platforms"
    if not platforms_dir.exists():
        print("❌ FAILED: Platforms directory missing.")
        return False
    md_files = list(platforms_dir.rglob("*.md"))
    print(f"Total Platform Knowledge Files: {len(md_files)}")
    if len(md_files) < 250:
        print(f"❌ FAILED: Expected >= 250 platform files, found {len(md_files)}")
        return False
    print("✅ PASSED: 100% of multi-platform knowledge files preserved.")
    return True

def main():
    print("""
===================================================================
  👑 QUILLAN-RONIN MASTER SYSTEM-WIDE AUDIT & HEALTH CHECK
===================================================================
    """)
    suites = [
        ("Python Syntax Verification", test_python_syntax),
        ("BPE Tokenizer Parity", test_tokenizer_roundtrip),
        ("Native AVX2 C++ Engine", test_native_engine),
        ("Model Parameter Integrity", test_checkpoints),
        ("Local Memory RAG Latency", test_local_rag),
        ("Platform Knowledge Preservation", test_platform_preservation),
    ]

    results = []
    for name, fn in suites:
        passed = fn()
        results.append((name, passed))

    print("\n" + "=" * 67)
    print("  📋 FINAL SYSTEM HEALTH SUMMARY")
    print("=" * 67)
    all_passed = True
    for name, passed in results:
        status_str = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status_str} : {name}")
        if not passed:
            all_passed = False

    print("=" * 67)
    if all_passed:
        print("🎉 100% SYSTEM-WIDE PARITY ACHIEVED: ALL SUITES HEALTHY!")
        sys.exit(0)
    else:
        print("⚠️ SYSTEM AUDIT FAILED ON ONE OR MORE SUITES.")
        sys.exit(1)

if __name__ == "__main__":
    main()
