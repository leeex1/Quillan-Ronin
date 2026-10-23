import subprocess
import sys
import os
from pathlib import Path

mcp_dir = Path(r"C:\02_QUILLAN\09 - Projects\mcp")
servers = [d for d in mcp_dir.iterdir() if d.is_dir() and (d / "server.py").exists()]

# Set required env vars
os.environ["QUILLAN_BASE"] = r"C:\02_QUILLAN"
os.environ["QUILLAN_MCP_BASE"] = r"C:\02_QUILLAN\09 - Projects\mcp"

print(f"Boot testing {len(servers)} MCP servers...\n")

for s in sorted(servers, key=lambda x: x.name):
    server_path = s / "server.py"
    # Test import + mcp instantiation (3 second timeout)
    test_code = f"""
import sys
sys.path.insert(0, r"{mcp_dir}")
sys.path.insert(0, r"C:\\02_QUILLAN\\09 - Projects\\projects\\oni")
sys.path.insert(0, r"C:\\02_QUILLAN\\03 - Training & Model\\scripts")
sys.path.insert(0, r"C:\\02_QUILLAN")

import os
os.environ["QUILLAN_BASE"] = r"C:\\02_QUILLAN"
os.environ["QUILLAN_MCP_BASE"] = r"C:\\02_QUILLAN\\09 - Projects\\mcp"

# Import the server module
spec = __import__("importlib.util").util.spec_from_file_location("{s.name}", r"{server_path}")
mod = __import__("importlib.util").util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Check mcp object exists
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
            timeout=10,
            cwd=str(mcp_dir),
        )
        if result.returncode == 0 and "BOOT_OK" in result.stdout:
            print(f"OK {s.name}")
        else:
            err = (
                result.stderr.strip()[:400]
                if result.stderr
                else result.stdout.strip()[:400]
            )
            print(f"FAIL {s.name}: {err}")
    except subprocess.TimeoutExpired:
        print(f"FAIL {s.name}: TIMEOUT (10s)")
    except Exception as e:
        print(f"FAIL {s.name}: {e}")
