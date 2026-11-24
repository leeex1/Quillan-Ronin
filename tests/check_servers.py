import subprocess
import sys
from pathlib import Path

mcp_dir = Path(r"C:\02_QUILLAN\09 - Projects\mcp")
servers = [d for d in mcp_dir.iterdir() if d.is_dir() and (d / "server.py").exists()]

print(f"Checking {len(servers)} MCP servers...\n")

for s in sorted(servers, key=lambda x: x.name):
    server_path = s / "server.py"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(server_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            print(f"OK {s.name}")
        else:
            print(f"FAIL {s.name}: {result.stderr.strip()[:300]}")
    except subprocess.TimeoutExpired:
        print(f"FAIL {s.name}: TIMEOUT")
    except Exception as e:
        print(f"FAIL {s.name}: {e}")
