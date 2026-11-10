"""
Sibling Modelfile Generator for Quillan Tripartite Hub Registry
Syncs Modelfile, Modelfile.mini, and Modelfile.dragon with verified GGUFs and native templates.
"""
import sys
from pathlib import Path

# Re-route to sync_production_modelfiles
scripts_dir = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(scripts_dir))
import sync_production_modelfiles

if __name__ == "__main__":
    sync_production_modelfiles.build_flagship_modelfile()
    sync_production_modelfiles.build_mini_modelfile()
    sync_production_modelfiles.build_dragon_modelfile()
    print("[SUCCESS] All three tripartite Modelfiles synchronized successfully.")
