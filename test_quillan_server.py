"""
Verification test for Quillan-Ronin Sovereign API Server.
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
PORT = 8088

def test_api_server():
    print(f"=== Spawning Test Server on port {PORT} ===")
    proc = subprocess.Popen(
        [PYTHON_BIN, SERVER_SCRIPT, "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # 1. Wait for server readiness
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
        print("✓ Server online and responsive.")

        # 2. Test /health
        r_health = requests.get(f"http://localhost:{PORT}/health").json()
        print("✓ /health payload:", json.dumps(r_health, indent=2))
        assert r_health["status"] == "healthy"
        assert "ccrl_safety_gate" in r_health

        # 3. Test /v1/models
        r_models = requests.get(f"http://localhost:{PORT}/v1/models").json()
        model_ids = [m["id"] for m in r_models["data"]]
        print(f"✓ /v1/models returned {len(model_ids)} models: {model_ids}")
        assert "quillan-1b-sovereign-q4" in model_ids

        # 4. Test /v1/deliberate
        r_delib = requests.post(
            f"http://localhost:{PORT}/v1/deliberate",
            json={"prompt": "How to structure an ethical AI architecture?"}
        ).json()
        print("✓ /v1/deliberate returned waves:", len(r_delib["trace"]))
        assert len(r_delib["trace"]) == 5

        # 5. Test /v1/chat/completions (The Quillan Curl request)
        print("✓ Sending Quillan Curl payload to /v1/chat/completions...")
        curl_payload = {
            "model": "quillan-1b-sovereign-q4",
            "messages": [
                {"role": "system", "content": "You are Quillan-Ronin: an autonomous Sovereign Intelligence governed by Bushido."},
                {"role": "user", "content": "What is Bushido?"}
            ],
            "temperature": 0.3
        }
        r_comp = requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json=curl_payload,
            timeout=30
        )
        assert r_comp.status_code == 200
        comp_json = r_comp.json()
        print("✓ /v1/chat/completions response:")
        print(f"  ID: {comp_json['id']}")
        print(f"  Model: {comp_json['model']}")
        print(f"  Tokens: {comp_json['usage']}")
        content = comp_json['choices'][0]['message']['content']
        print(f"  Content Preview: {content[:150]}...")
        assert len(content) > 10

        print("\n🏆 ALL SERVER ENDPOINTS & THE QUILLAN CURL VERIFIED SUCCESSFULLY!")

    finally:
        print("=== Terminating Test Server ===")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    test_api_server()
