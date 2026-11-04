import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("TESTING QUILLAN-RONIN MODULE IMPORTS & INTEGRATIONS")
print("=" * 60)

try:
    import quillan_bridge
    print("[-] quillan_bridge: IMPORTED OK")
    summary = quillan_bridge.bridge.get_status_summary()
    print(f"    OpenCode keys: {summary.get('opencode', {}).get('key_count')}")
    print(f"    Antigravity: {summary.get('antigravity', {}).get('linked')}")
    print(f"    Ollama models discovered ({summary.get('ollama', {}).get('model_count')}):")
    for m in summary.get('ollama', {}).get('models_discovered', [])[:5]:
        print(f"      - {m}")
except Exception as e:
    print(f"[!] quillan_bridge ERROR: {e}")

try:
    import quillan_engine
    print("[-] quillan_engine: IMPORTED OK")
    engine = quillan_engine.SovereignInferenceEngine()
    models = engine.get_available_models()
    print(f"    Available models in engine ({len(models)}):")
    for k, v in list(models.items())[:6]:
        print(f"      * {k}: {v.get('name')}")
except Exception as e:
    print(f"[!] quillan_engine ERROR: {e}")

try:
    import quillan_agent
    print("[-] quillan_agent: IMPORTED OK")
    tools = quillan_agent.QuillanToolRegistry()
    print("    Registry initialized successfully.")
except Exception as e:
    print(f"[!] quillan_agent ERROR: {e}")

try:
    import quillan_server
    print("[-] quillan_server: IMPORTED OK")
except Exception as e:
    print(f"[!] quillan_server ERROR: {e}")

try:
    import quillan_tui
    print("[-] quillan_tui: IMPORTED OK")
except Exception as e:
    print(f"[!] quillan_tui ERROR: {e}")

print("=" * 60)
print("ALL CORE MODULE AUDIT COMPLETE")
print("=" * 60)
