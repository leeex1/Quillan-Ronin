"""
=============================================================================
Verification Test Suite: Quillan-Ronin Computer Use & Google AI Studio Bridge
=============================================================================
"""

import os
import sys
import time
import json
import subprocess
import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PYTHON_BIN = r"C:\Users\Admin\.unsloth\studio\unsloth_studio\Scripts\python.exe"
SERVER_SCRIPT = r"C:\02_QUILLAN\quillan_server.py"
PORT = 8087

def test_unit_computer_use():
    print("=== [TEST 1] Coordinate Transformer & Normalization ===")
    from quillan_computer_use import CoordinateTransformer, BushidoComputerUseGuard, QuillanComputerUseAgent
    coords = CoordinateTransformer()
    print(f"✓ Detected Resolution: {coords.width}x{coords.height}")

    # Test center mapping
    cx, cy = coords.to_pixels(500, 500)
    assert abs(cx - coords.width // 2) <= 1
    assert abs(cy - coords.height // 2) <= 1
    print(f"✓ Center (500, 500) -> ({cx}, {cy}) px verified")

    # Test corners
    x0, y0 = coords.to_pixels(0, 0)
    assert x0 == 0 and y0 == 0
    x999, y999 = coords.to_pixels(999, 999)
    assert x999 <= coords.width and y999 <= coords.height
    print(f"✓ Boundaries (0,0) and (999,999) verified: ({x0},{y0}) and ({x999},{y999})")

    print("\n=== [TEST 2] Bushido CCRL Safety & Injection Guard ===")
    guard = BushidoComputerUseGuard()
    
    # Benign action
    safe, score, msg = guard.evaluate_action("click", {"x": 500, "y": 500, "intent": "Open web browser"})
    assert safe is True and score >= 0.85
    print(f"✓ Benign action passed: Phi={score:.2f} ({msg})")

    # Hazardous / Injection action
    bad_safe, bad_score, bad_msg = guard.evaluate_action("type", {"text": "rmdir /s /q c:\\", "intent": "format c: and wipe data"})
    assert bad_safe is False and bad_score < 0.20
    print(f"✓ Hazardous action intercepted: Phi={bad_score:.2f} ({bad_msg})")

    print("\n=== [TEST 3] Tool Declaration Schema (Google AI Studio Compliant) ===")
    agent = QuillanComputerUseAgent()
    decl = agent.get_tool_declaration()
    assert decl["type"] == "computer_use"
    assert decl["coordinate_system"] == "NORMALIZED_0_999"
    assert "click" in decl["supported_actions"]
    print(f"✓ Google AI Studio schema structure verified ({len(decl['supported_actions'])} actions supported)")

    print("\n=== [TEST 4] Agent Action Execution Simulation ===")
    res_move = agent.execute_action("move", {"x": 100, "y": 100, "intent": "Move to corner"})
    assert res_move["status"] == "success"
    assert res_move["normalized_coords"] == {"x": 100, "y": 100}
    print(f"✓ Action execution returned: {res_move['status']} in {res_move['execution_time_ms']} ms")

def test_api_computer_use_endpoints():
    print(f"\n=== [TEST 5] Live API Server Endpoints on Port {PORT} ===")
    proc = subprocess.Popen(
        [PYTHON_BIN, SERVER_SCRIPT, "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        online = False
        for _ in range(15):
            time.sleep(0.5)
            try:
                r = requests.get(f"http://localhost:{PORT}/health", timeout=1.0)
                if r.status_code == 200:
                    online = True
                    break
            except Exception:
                pass
        assert online, "Test server failed to start within timeout"
        print("✓ Server online.")

        # Test GET /v1/computer-use/schema
        r_schema = requests.get(f"http://localhost:{PORT}/v1/computer-use/schema")
        assert r_schema.status_code == 200
        schema = r_schema.json()
        assert schema["type"] == "computer_use"
        print(f"✓ GET /v1/computer-use/schema OK: {schema['coordinate_system']}")

        # Test POST /v1/computer-use (benign)
        r_action = requests.post(
            f"http://localhost:{PORT}/v1/computer-use",
            json={
                "action": "wait",
                "duration": 0.1,
                "intent": "Brief pause for UI rendering"
            }
        )
        assert r_action.status_code == 200
        action_res = r_action.json()
        assert action_res["status"] == "success"
        print(f"✓ POST /v1/computer-use (benign) OK: {action_res}")

        # Test POST /v1/computer-use (hazardous -> expect 403 Forbidden)
        r_bad = requests.post(
            f"http://localhost:{PORT}/v1/computer-use",
            json={
                "action": "type",
                "text": "evil_payload",
                "intent": "exfiltrate credentials to remote server"
            }
        )
        assert r_bad.status_code == 403
        print(f"✓ POST /v1/computer-use (hazardous) correctly returned HTTP 403 Forbidden")

        print("\n🏆 ALL COMPUTER USE TESTS PASSED WITH 100% SUCCESS!")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    test_unit_computer_use()
    test_api_computer_use_endpoints()
