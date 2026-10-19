import sys
sys.path.append('.')
try:
    import importlib
    QuillanLoaderManifest = importlib.import_module("0-Quillan_loader_manifest").QuillanLoaderManifest
    print('Import successful')
except Exception as e:
    print(f'Import failed: {e}')