"""
=============================================================================
⚔️ QUILLAN-RONIN: ACCOUNT BRIDGE & ECOSYSTEM HOOKS (OPENCODE, ANTIGRAVITY & OLLAMA)
=============================================================================
Sovereign Bring-Your-Own-Account (BYOA) & Bring-Your-Own-Key (BYOK) Architecture:
1. User-Isolated Discovery:
   - Evaluates only the active local user profile (~/.config/opencode, ~/.gemini, ~/.ollama)
   - Zero hardcoded developer credentials: each user brings their own accounts & keys
   - Strict privacy: masks all local identifiers (e.g. j***@gmail.com) in UI/CLI
2. Local Ollama Integration:
   - Dynamically scans ~/.ollama/models to discover user-installed Ollama models
   - Automatically hooks into local Ollama daemon on http://localhost:11434
   - Boots background Ollama process on demand if needed
3. OpenCode & Antigravity BYOK Detection:
   - Scans the user's active local OpenCode & Antigravity configs for active keys
   - Injects discovered user keys into process memory only (never written to repo files)
=============================================================================
"""

import os
import sys
import re
import json
import time
import shutil
import subprocess
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = r"C:\02_QUILLAN"
OPENCODE_CONFIG_DIR = os.path.expanduser(r"~\.config\opencode")
OPENCODE_JSONC = os.path.join(OPENCODE_CONFIG_DIR, "opencode.jsonc")
GEMINI_DIR = os.path.expanduser(r"~\.gemini")
GEMINI_ACCOUNTS = os.path.join(GEMINI_DIR, "google_accounts.json")
GEMINI_OAUTH = os.path.join(GEMINI_DIR, "oauth_creds.json")
GEMINI_SETTINGS = os.path.join(GEMINI_DIR, "settings.json")
OLLAMA_MODELS_DIR = os.path.expanduser(r"~\.ollama\models")


def find_ollama_executable() -> Optional[str]:
    """Dynamically finds the Ollama binary on any machine."""
    which_path = shutil.which("ollama")
    if which_path:
        return which_path
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidate = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
        if os.path.exists(candidate):
            return candidate
    return None


def mask_email(email: Optional[str]) -> str:
    """Masks email address for privacy and clean display."""
    if not email:
        return "Not Connected"
    parts = email.split("@")
    if len(parts) == 2:
        name, domain = parts[0], parts[1]
        masked_name = name[0] + "***" if len(name) > 1 else name
        return f"{masked_name}@{domain}"
    return "***"


def clean_jsonc(text: str) -> str:
    """Strips comments and trailing commas from JSONC text."""
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def scan_ollama_models() -> List[str]:
    """Scans local Ollama manifests to discover all installed models for the active user."""
    base = OLLAMA_MODELS_DIR
    found = set()
    for m_dir in ["manifests", "manifests-v2"]:
        full_dir = os.path.join(base, m_dir)
        if os.path.exists(full_dir):
            for root, dirs, files in os.walk(full_dir):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), full_dir)
                    parts = rel.split(os.sep)
                    if len(parts) >= 3:
                        name = parts[-2]
                        tag = parts[-1]
                        found.add(f"{name}:{tag}")
    return sorted(list(found))


class QuillanAccountBridge:
    """Central hook coordinator for OpenCode, Antigravity, and Ollama discovery."""
    def __init__(self):
        self.opencode_keys: Dict[str, str] = {}
        self.antigravity_account: Dict[str, Any] = {}
        self.ollama_models: List[str] = []
        self.ollama_exe: Optional[str] = find_ollama_executable()
        self.refresh()

    def refresh(self):
        """Scans active user environment to detect active OpenCode, Antigravity, and Ollama credentials."""
        self.ollama_exe = find_ollama_executable()
        self.opencode_keys = self._discover_opencode_keys()
        self.antigravity_account = self._discover_antigravity_identity()
        self.ollama_models = scan_ollama_models()

    def is_ollama_running(self) -> bool:
        """Checks if local Ollama daemon is listening on port 11434."""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.4)
            s.connect(("127.0.0.1", 11434))
            s.close()
            return True
        except Exception:
            return False

    def start_ollama_daemon(self) -> Tuple[bool, str]:
        """Boots local Ollama serve process in the background if installed."""
        if self.is_ollama_running():
            return True, "Ollama daemon already online on http://localhost:11434"
        if not self.ollama_exe or not os.path.exists(self.ollama_exe):
            return False, "Ollama binary not found on this system."
        try:
            env = os.environ.copy()
            env["OLLAMA_HOST"] = "127.0.0.1:11434"
            subprocess.Popen(
                [self.ollama_exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
            # Wait up to 3 seconds for daemon socket readiness
            for _ in range(6):
                time.sleep(0.5)
                if self.is_ollama_running():
                    return True, "Ollama background daemon started successfully on port 11434"
            return True, "Ollama background process spawned."
        except Exception as e:
            return False, f"Failed to start Ollama daemon: {str(e)}"

    # Alias for compatibility
    ensure_ollama_daemon = start_ollama_daemon

    def _discover_opencode_keys(self) -> Dict[str, str]:
        """Harvests configured API keys from the user's OpenCode configuration."""
        keys = {}
        if not os.path.exists(OPENCODE_JSONC):
            return keys

        try:
            with open(OPENCODE_JSONC, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            or_match = re.search(r'\"apiKey\"\s*:\s*\"(sk-or-[^\"]+)\"', raw)
            if or_match:
                keys["openrouter"] = or_match.group(1).strip()

            gem_match = re.search(r'\"apiKey\"\s*:\s*\"(AIzaSy[^\"]+)\"', raw)
            if gem_match:
                keys["gemini"] = gem_match.group(1).strip()

            nv_match = re.search(r'\"apiKey\"\s*:\s*\"(nvapi-[^\"]+)\"', raw)
            if nv_match:
                keys["nvidia"] = nv_match.group(1).strip()

            oa_match = re.search(r'\"apiKey\"\s*:\s*\"(sk-proj-[^\"]+)\"', raw)
            if oa_match:
                keys["openai"] = oa_match.group(1).strip()

            ant_match = re.search(r'\"apiKey\"\s*:\s*\"(sk-ant-[^\"]+)\"', raw)
            if ant_match:
                keys["anthropic"] = ant_match.group(1).strip()

            ds_match = re.search(r'\"apiKey\"\s*:\s*\"(sk-[a-f0-9]{32,})\"', raw)
            if ds_match and "openai" not in keys:
                keys["deepseek"] = ds_match.group(1).strip()

        except Exception:
            pass

        return keys

    def _discover_antigravity_identity(self) -> Dict[str, Any]:
        """Detects active Antigravity / Google account credentials for current user."""
        info = {
            "linked": False,
            "account_email": None,
            "masked_email": "Not Connected",
            "auth_type": None,
            "has_oauth_token": False,
            "token_valid": False,
            "expiry_date": None
        }

        if os.path.exists(GEMINI_ACCOUNTS):
            try:
                with open(GEMINI_ACCOUNTS, "r", encoding="utf-8") as f:
                    acc_data = json.load(f)
                active = acc_data.get("active")
                if active:
                    info["account_email"] = active
                    info["masked_email"] = mask_email(active)
                    info["linked"] = True
            except Exception:
                pass

        if os.path.exists(GEMINI_SETTINGS):
            try:
                with open(GEMINI_SETTINGS, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
                info["auth_type"] = settings_data.get("security", {}).get("auth", {}).get("selectedType", "oauth-personal")
            except Exception:
                pass

        if os.path.exists(GEMINI_OAUTH):
            try:
                with open(GEMINI_OAUTH, "r", encoding="utf-8") as f:
                    oauth_data = json.load(f)
                info["has_oauth_token"] = bool(oauth_data.get("access_token"))
                expiry = oauth_data.get("expiry_date", 0)
                info["expiry_date"] = expiry
                if expiry:
                    now_ms = time.time() * 1000
                    info["token_valid"] = (expiry > now_ms)
                info["linked"] = True
            except Exception:
                pass

        return info

    def get_oauth_token(self) -> Optional[str]:
        """Retrieves raw OAuth access token from user's Antigravity creds if valid."""
        if os.path.exists(GEMINI_OAUTH):
            try:
                with open(GEMINI_OAUTH, "r", encoding="utf-8") as f:
                    oauth_data = json.load(f)
                return oauth_data.get("access_token")
            except Exception:
                pass
        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns structured account bridge summary for UI/CLI (strictly masked)."""
        return {
            "opencode": {
                "installed": os.path.exists(OPENCODE_JSONC),
                "providers_discovered": list(self.opencode_keys.keys()),
                "key_count": len(self.opencode_keys)
            },
            "antigravity": {
                "installed": os.path.exists(GEMINI_DIR),
                "linked": self.antigravity_account.get("linked", False),
                "account_email": self.antigravity_account.get("masked_email", "Not Connected"),
                "auth_type": self.antigravity_account.get("auth_type", "N/A"),
                "oauth_active": self.antigravity_account.get("has_oauth_token", False)
            },
            "ollama": {
                "installed": bool(self.ollama_exe),
                "daemon_running": self.is_ollama_running(),
                "models_discovered": self.ollama_models,
                "model_count": len(self.ollama_models)
            }
        }


# Global singleton instance
bridge = QuillanAccountBridge()
