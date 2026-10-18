import sys
import importlib.util
spec = importlib.util.spec_from_file_location("0-Quillan_loader_manifest", "0-Quillan_loader_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('Import successful')