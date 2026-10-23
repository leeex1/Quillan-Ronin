import subprocess
import sys
import os
from pathlib import Path

mcp_dir = Path(r"C:\02_QUILLAN\09 - Projects\mcp")
servers = [d for d in mcp_dir.iterdir() if d.is_dir() and (d / "server.py").exists()]

os.environ["QUILLAN_BASE"] = r"C:\02_QUILLAN"
os.environ["QUILLAN_MCP_BASE"] = r"C:\02_QUILLAN\09 - Projects\mcp"

print(f"Boot testing {len(servers)} MCP servers...\n")

for s in sorted(servers, key=lambda x: x.name):
    server_path = s / "server.py"
    test_code = f"""
import sys
sys.path.insert(0, r"{mcp_dir}")
sys.path.insert(0, r"C:\\02_QUILLAN\\09 - Projects\\projects\\oni")
sys.path.insert(0, r"C:\\02_QUILLAN\\03 - Training & Model\\scripts")
sys.path.insert(0, r"C:\\02_QUILLAN")

import os
os.environ["QUILLAN_BASE"] = r"C:\\02_QUILLAN"
os.environ["QUILLAN_MCP_BASE"] = r"C:\\02_QUILLAN\\09 - Projects\\mcp"

spec = __import__("importlib.util").util.spec_from_file_location("{s.name}", r"{server_path}")
mod = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(mod)

mcp = getattr(mod, "mcp", None)
if mcp is None:
    raise RuntimeError("No mcp object found")
print("BOOT_OK")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(mcp_dir),
        )
        if result.returncode == 0:
            print(f"[OK] {s.name} (exit={result.returncode})")
        else:
            print(f"[FAIL] {s.name} (exit={result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr.strip()[:300]}")
            if result.stdout:
                print(f"    stdout: {result.stdout.strip()[:300]}")
    except subprocess.TimeoutExpired:
        print(f"✗ FAIL {s.name}: TIMEOUT (15s)")
    except Exception as e:
        print(f"✗ FAIL {s.name}: {e}")
