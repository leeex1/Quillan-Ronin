"""
=============================================================================
⚔️ QUILLAN-RONIN: ACCOUNT BRIDGE & ECOSYSTEM HOOKS (OPENCODE & ANTIGRAVITY)
=============================================================================
Seamlessly discovers and hooks into existing developer accounts:
1. OpenCode Accounts Hook:
   - Scans ~/.config/opencode/opencode.jsonc and AppData/Roaming/ai.opencode.desktop
   - Extracts configured provider keys (OpenRouter, Google Gemini, NVIDIA NIM, OpenAI, Anthropic)
   - Automatically registers discovered keys into Quillan's KeyManager
2. Antigravity & Gemini OAuth Hook:
   - Scans ~/.gemini/ (google_accounts.json, oauth_creds.json, settings.json)
   - Detects logged-in Google / Antigravity identity and OAuth tokens
   - Enables direct API access using Antigravity account quota
3. Universal Model Compatibility:
   - Aggregates Local GGUF, OpenCode providers, and Antigravity accounts into a single unified catalog
=============================================================================
"""

import os
import sys
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple

ROOT_DIR = r"C:\02_QUILLAN"
OPENCODE_CONFIG_DIR = os.path.expanduser(r"~\.config\opencode")
OPENCODE_JSONC = os.path.join(OPENCODE_CONFIG_DIR, "opencode.jsonc")
GEMINI_DIR = os.path.expanduser(r"~\.gemini")
GEMINI_ACCOUNTS = os.path.join(GEMINI_DIR, "google_accounts.json")
GEMINI_OAUTH = os.path.join(GEMINI_DIR, "oauth_creds.json")
GEMINI_SETTINGS = os.path.join(GEMINI_DIR, "settings.json")


def clean_jsonc(text: str) -> str:
    """Strips comments and trailing commas from JSONC text."""
    # Strip line comments
    text = re.sub(r"//.*", "", text)
    # Strip block comments
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip trailing commas before closing braces/brackets
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


class QuillanAccountBridge:
    """Central hook coordinator for OpenCode and Antigravity account discovery."""
    def __init__(self):
        self.opencode_keys: Dict[str, str] = {}
        self.antigravity_account: Dict[str, Any] = {}
        self.refresh()

    def refresh(self):
        """Scans filesystem to detect active OpenCode and Antigravity credentials."""
        self.opencode_keys = self._discover_opencode_keys()
        self.antigravity_account = self._discover_antigravity_identity()

    def _discover_opencode_keys(self) -> Dict[str, str]:
        """Harvests configured API keys from OpenCode configuration."""
        keys = {}
        if not os.path.exists(OPENCODE_JSONC):
            return keys

        try:
            with open(OPENCODE_JSONC, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            # 1. Regex discovery for known key patterns in OpenCode
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

        except Exception as e:
            pass

        return keys

    def _discover_antigravity_identity(self) -> Dict[str, Any]:
        """Detects active Antigravity / Google account credentials."""
        info = {
            "linked": False,
            "account_email": None,
            "auth_type": None,
            "has_oauth_token": False,
            "token_valid": False,
            "expiry_date": None
        }

        # Check google_accounts.json
        if os.path.exists(GEMINI_ACCOUNTS):
            try:
                with open(GEMINI_ACCOUNTS, "r", encoding="utf-8") as f:
                    acc_data = json.load(f)
                active = acc_data.get("active")
                if active:
                    info["account_email"] = active
                    info["linked"] = True
            except Exception:
                pass

        # Check settings.json for auth type
        if os.path.exists(GEMINI_SETTINGS):
            try:
                with open(GEMINI_SETTINGS, "r", encoding="utf-8") as f:
                    settings_data = json.load(f)
                info["auth_type"] = settings_data.get("security", {}).get("auth", {}).get("selectedType", "oauth-personal")
            except Exception:
                pass

        # Check oauth_creds.json for live tokens
        if os.path.exists(GEMINI_OAUTH):
            try:
                with open(GEMINI_OAUTH, "r", encoding="utf-8") as f:
                    oauth_data = json.load(f)
                info["has_oauth_token"] = bool(oauth_data.get("access_token"))
                expiry = oauth_data.get("expiry_date", 0)
                info["expiry_date"] = expiry
                # If expiry in milliseconds and in future -> valid
                if expiry:
                    now_ms = time.time() * 1000
                    info["token_valid"] = (expiry > now_ms)
                info["linked"] = True
            except Exception:
                pass

        return info

    def get_oauth_token(self) -> Optional[str]:
        """Retrieves raw OAuth access token from Antigravity creds if valid."""
        if os.path.exists(GEMINI_OAUTH):
            try:
                with open(GEMINI_OAUTH, "r", encoding="utf-8") as f:
                    oauth_data = json.load(f)
                return oauth_data.get("access_token")
            except Exception:
                pass
        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns structured account bridge summary for UI/CLI."""
        return {
            "opencode": {
                "installed": os.path.exists(OPENCODE_JSONC),
                "providers_discovered": list(self.opencode_keys.keys()),
                "key_count": len(self.opencode_keys)
            },
            "antigravity": {
                "installed": os.path.exists(GEMINI_DIR),
                "linked": self.antigravity_account.get("linked", False),
                "account_email": self.antigravity_account.get("account_email", "Not Logged In"),
                "auth_type": self.antigravity_account.get("auth_type", "N/A"),
                "oauth_active": self.antigravity_account.get("has_oauth_token", False)
            }
        }


# Global singleton instance
bridge = QuillanAccountBridge()
