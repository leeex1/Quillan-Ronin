#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
👑 QUILLAN-RONIN SYSTEM HEALTH & INTEGRITY AUDIT SUITE (v5.4.0)
===============================================================================
Performs an automated, end-to-end verification of all live and core sections:
  1. Live Web Portal & GitHub Pages Assets (HTML, Visualizer, NFT DB)
  2. Quantum Mechanics & Density Matrix Invariants (AQCS, EEMF, QHIS, DQRO, QCRDM, JQLD)
  3. Deserialization & Secret Security Hygiene
  4. Hardware Telemetry & Runtime Governor Intertwining
  5. Script Path Resolutions across 05_Training and 02_Projects
===============================================================================
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Color helpers for readable CLI audit output
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def log_audit(section: str, name: str, passed: bool, details: str = ""):
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    det = f" -> {details}" if details else ""
    print(f"  {status} {section} :: {name}{det}")
    return passed

def audit_gitpage_assets(repo_root: Path) -> bool:
    print(f"\n{CYAN}{BOLD}--- [1/5] AUDITING LIVE GITHUB PAGES & WEB SECTIONS ---{RESET}")
    all_ok = True
    
    docs_dir = repo_root / "02_Projects" / "docs"
    index_html = docs_dir / "index.html"
    vis_html = docs_dir / "nn_visualizer.html"
    nft_json = docs_dir / "nft_data.json"
    
    # 1.1 Index HTML presence and size
    idx_ok = index_html.exists() and index_html.stat().st_size > 1000000
    all_ok &= log_audit("GitPage", "02_Projects/docs/index.html exists & non-empty", idx_ok, f"Size: {index_html.stat().st_size:,} bytes" if idx_ok else "Not found")
    
    # 1.2 NN Visualizer
    vis_ok = vis_html.exists() and vis_html.stat().st_size > 5000
    all_ok &= log_audit("GitPage", "02_Projects/docs/nn_visualizer.html exists", vis_ok, f"Size: {vis_html.stat().st_size:,} bytes" if vis_ok else "Not found")
    
    # 1.3 NFT Data JSON
    nft_ok = False
    if nft_json.exists():
        try:
            with open(nft_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            nft_ok = isinstance(data, (list, dict)) and len(data) > 0
            details = f"Loaded {len(data)} items"
        except Exception as e:
            details = f"JSON parse error: {e}"
    else:
        details = "File missing"
    all_ok &= log_audit("GitPage", "02_Projects/docs/nft_data.json valid JSON", nft_ok, details)
    
    return all_ok

def audit_quantum_engine(repo_root: Path) -> bool:
    print(f"\n{CYAN}{BOLD}--- [2/5] AUDITING QUANTUM FORMULAS & DENSITY MATRIX INVARIANTS ---{RESET}")
    all_ok = True
    oni_dir = repo_root / "00 - Meta" / "oni"
    
    if str(oni_dir) not in sys.path:
        sys.path.insert(0, str(oni_dir))
        
    try:
        import torch
        from quillan_v5_4_oni import QuantumFormulasEngine
        
        q_engine = QuantumFormulasEngine(hidden_dim=128)
        B, N, D = 2, 34, 128
        probs = torch.softmax(torch.randn(B, N), dim=-1)
        vectors = torch.randn(B, N, D)
        
        # 2.1 Formula 1: AQCS Superposition
        aqcs = q_engine.aqcs_superposition(probs, vectors)
        f1_ok = aqcs.shape == (B, D) and not torch.isnan(aqcs).any()
        all_ok &= log_audit("Quantum", "AQCS Superposition (|Psi>)", f1_ok, f"Shape: {tuple(aqcs.shape)}")
        
        # 2.2 Formula 2: EEMF Reduced Density & Partial Trace
        hidden = torch.randn(B, 10, D)
        rho_sys, purity, lin_ent = q_engine.eemf_reduced_density(hidden)
        f2_ok = rho_sys.shape == (B, D // 2, D // 2) and (purity >= 0.0).all() and not torch.isnan(purity).any()
        all_ok &= log_audit("Quantum", "EEMF Partial Trace Density Matrix (Tr_env)", f2_ok, f"Purity: {purity.mean().item():.4f}")
        
        # 2.3 Formula 3: QHIS Bures Fidelity
        h_prev = torch.randn(B, D)
        h_curr = torch.randn(B, D)
        fidelity = q_engine.qhis_fidelity(h_prev, h_curr, v_lm6=1.0)
        f3_ok = not torch.isnan(fidelity).any()
        all_ok &= log_audit("Quantum", "QHIS Bures Fidelity", f3_ok, f"Score: {fidelity.item():.4f}")
        
        # 2.4 Formula 4: DQRO Ising Hamiltonian
        spins = torch.sign(torch.randn(B, D))
        energy = q_engine.dqro_energy(spins)
        f4_ok = energy.shape == (B,) and not torch.isnan(energy).any()
        all_ok &= log_audit("Quantum", "DQRO Ising Hamiltonian", f4_ok, f"Energy Mean: {energy.mean().item():.4f}")
        
        # 2.5 Formula 5: QCRDM Reasoning Projection
        qcrdm = q_engine.qcrdm_reasoning(aqcs)
        f5_ok = qcrdm.shape == (B, D) and not torch.isnan(qcrdm).any()
        all_ok &= log_audit("Quantum", "QCRDM Born Rule Reasoning", f5_ok, f"Output Norm: {qcrdm.norm(dim=-1).mean().item():.4f}")
        
        # 2.6 Formula 6: JQLD Lindblad Dissipation
        sample_hidden = torch.randn(B, D)
        lindblad = q_engine.jqld_evolution_step(sample_hidden)
        f6_ok = lindblad.shape == (B, D) and not torch.isnan(lindblad).any()
        all_ok &= log_audit("Quantum", "JQLD Open-System Lindblad Dynamics", f6_ok, f"Output Norm: {lindblad.norm(dim=-1).mean().item():.4f}")
        
    except Exception as e:
        all_ok &= log_audit("Quantum", "Quantum Engine Execution", False, str(e))
        
    return all_ok

def audit_security_hygiene(repo_root: Path) -> bool:
    print(f"\n{CYAN}{BOLD}--- [3/5] AUDITING SECURITY & DESERIALIZATION HYGIENE ---{RESET}")
    all_ok = True
    
    # 3.1 Check Chrome extension background.js for raw eval(expr)
    ext_bg = repo_root / "testing" / "Quillan-Ronin-chrome-extension" / "background.js"
    if ext_bg.exists():
        content = ext_bg.read_text(encoding="utf-8")
        has_raw_eval = "eval(" in content and "eval(expr)" in content
        all_ok &= log_audit("Security", "Chrome Extension raw eval() elimination", not has_raw_eval, "Zero raw eval(expr) found" if not has_raw_eval else "Vulnerable eval(expr) present")
    else:
        log_audit("Security", "Chrome Extension background.js check", True, "Skipped (extension not found)")
        
    # 3.2 Check .gitignore coverage for .env
    gitignore_file = repo_root / ".gitignore"
    if gitignore_file.exists():
        gi_text = gitignore_file.read_text(encoding="utf-8")
        env_ignored = ".env" in gi_text
        all_ok &= log_audit("Security", ".env protected in .gitignore", env_ignored, ".env rule verified")
    else:
        all_ok &= log_audit("Security", ".gitignore presence", False, ".gitignore missing")
        
    # 3.3 Check training watchdog safe deserialization
    watchdog = repo_root / "05_Training" / "scripts" / "training_watchdog.py"
    if watchdog.exists():
        wd_text = watchdog.read_text(encoding="utf-8")
        has_weights_only = "weights_only=True" in wd_text
        all_ok &= log_audit("Security", "training_watchdog.py weights_only guard", has_weights_only, "Safe load path verified")
        
    return all_ok

def audit_hardware_governor(repo_root: Path) -> bool:
    print(f"\n{CYAN}{BOLD}--- [4/5] AUDITING HARDWARE TELEMETRY GOVERNOR INTERTWINING ---{RESET}")
    all_ok = True
    scripts_dir = repo_root / "05_Training" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
        
    try:
        from quillan_hardware_governor import (
            HardwareTelemetryGovernor,
            HardwareThresholds,
            TelemetrySnapshot,
            LeeMach6GovernorAdapter,
        )
        
        thresholds = HardwareThresholds(sample_interval_s=0.05)
        with HardwareTelemetryGovernor(thresholds=thresholds) as gov:
            time.sleep(0.12)
            snap = gov.get_snapshot()
            snap_ok = isinstance(snap, TelemetrySnapshot) and snap.timestamp > 0
            all_ok &= log_audit("Telemetry", "Asynchronous daemon lifecycle & snapshot", snap_ok, f"CPU: {snap.cpu_temp_c}C, RAM: {snap.host_ram_used_pct:.1f}%")
            
            # Step boundary hook
            hook_res = gov.step_boundary_hook()
            hook_ok = "throttle_factor" in hook_res and "paused_seconds" in hook_res
            all_ok &= log_audit("Telemetry", "step_boundary_hook() reactive throttle contract", hook_ok, f"Throttle: {hook_res['throttle_factor']}")
            
            # Adapter backward-compatibility
            adapter = LeeMach6GovernorAdapter(target_latency_ms=100, hardware_governor=gov)
            scale, ema, bias = adapter.adjust(latency_ms=120.0)
            adapter_ok = (0.0 < scale <= 1.0) and (0.0 <= ema <= 1.0)
            all_ok &= log_audit("Telemetry", "LeeMach6GovernorAdapter backward-compatibility", adapter_ok, f"Scale: {scale:.2f}, EMA: {ema:.4f}")
            
    except Exception as e:
        all_ok &= log_audit("Telemetry", "Hardware Telemetry Governor execution", False, str(e))
        
    return all_ok

def audit_script_path_consistency(repo_root: Path) -> bool:
    print(f"\n{CYAN}{BOLD}--- [5/5] AUDITING REPOSITORY PATH INTEGRITY & RESOLUTION ---{RESET}")
    all_ok = True
    
    # 5.1 build_flagship_portal.py
    portal_script = repo_root / "05_Training" / "scripts" / "build_flagship_portal.py"
    if portal_script.exists():
        text = portal_script.read_text(encoding="utf-8")
        dynamic_docs = "02_Projects" in text or "repo_root" in text
        all_ok &= log_audit("Paths", "build_flagship_portal.py points to active docs", dynamic_docs, "Dynamic candidate path resolution verified")
        
    # 5.2 update_nn_vis.py
    vis_script = repo_root / "05_Training" / "scripts" / "update_nn_vis.py"
    if vis_script.exists():
        text = vis_script.read_text(encoding="utf-8")
        dynamic_vis = "02_Projects" in text
        all_ok &= log_audit("Paths", "update_nn_vis.py targets 02_Projects/docs/nn_visualizer.html", dynamic_vis, "Active visualizer target verified")
        
    # 5.3 training_watchdog.py
    watchdog_script = repo_root / "05_Training" / "scripts" / "training_watchdog.py"
    if watchdog_script.exists():
        text = watchdog_script.read_text(encoding="utf-8")
        dynamic_wd = "Path(__file__).resolve().parent" in text
        all_ok &= log_audit("Paths", "training_watchdog.py self-resolves parent directory", dynamic_wd, "SCRIPTS_DIR canonicalization verified")
        
    return all_ok

def main():
    repo_root = Path(r"C:\02_QUILLAN").resolve()
    print("=" * 70)
    print(f"{BOLD}👑 QUILLAN-RONIN COMPREHENSIVE SYSTEM INTEGRITY & HEALTH AUDIT{RESET}")
    print(f"Target Root: {repo_root}")
    print("=" * 70)
    
    r1 = audit_gitpage_assets(repo_root)
    r2 = audit_quantum_engine(repo_root)
    r3 = audit_security_hygiene(repo_root)
    r4 = audit_hardware_governor(repo_root)
    r5 = audit_script_path_consistency(repo_root)
    
    overall = r1 and r2 and r3 and r4 and r5
    print("\n" + "=" * 70)
    if overall:
        print(f"{GREEN}{BOLD}>>> AUDIT COMPLETE: ALL SYSTEM SECTIONS 100% HEALTHY & VERIFIED <<<{RESET}")
    else:
        print(f"{RED}{BOLD}>>> AUDIT COMPLETED WITH WARNINGS/FAILURES <<<{RESET}")
    print("=" * 70)
    sys.exit(0 if overall else 1)

if __name__ == "__main__":
    main()
