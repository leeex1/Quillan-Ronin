#!/usr/bin/env python3
"""Server-side entitlement and item-issuance authority for NextVerse.

The client never decides what a user owns. Tiers and item definitions are
server-side; issuance is append-only with a hash chain so provenance can be
verified and exported. State persists to JSON across restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

TIERS = ("FREE", "EXPLORER", "CREATOR", "SOVEREIGN")
TIER_RANK = {name: index for index, name in enumerate(TIERS)}

# Cosmetic/expressive only - never combat power or inventory advantage.
ITEMS: dict[str, dict[str, Any]] = {
    "skin_neon_ronin": {
        "name": "Neon Ronin Skin",
        "category": "avatar_skin",
        "min_tier": "EXPLORER",
    },
    "emote_salute": {"name": "Ronin Salute", "category": "emote", "min_tier": "FREE"},
    "hub_theme_void": {
        "name": "Void Hub Theme",
        "category": "hub_theme",
        "min_tier": "CREATOR",
    },
    "stage_theme_aurora": {
        "name": "Aurora Stage",
        "category": "stage_theme",
        "min_tier": "CREATOR",
    },
    "hall_golden": {
        "name": "Golden Guild Hall",
        "category": "guild_hall",
        "min_tier": "SOVEREIGN",
    },
}

STATE_FILE = Path(
    os.environ.get(
        "NEXTVERSE_ENTITLEMENT_STATE",
        Path(__file__).resolve().parent / "entitlement_state.json",
    )
)


class EntitlementService:
    def __init__(self, state_file: Path | None = None) -> None:
        self.state_file = Path(state_file) if state_file else STATE_FILE
        self._lock = threading.Lock()
        self._accounts: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                with self._lock:
                    self._accounts = data
        except (OSError, ValueError):
            return

    def _persist(self) -> None:
        with self._lock:
            snapshot = json.dumps(self._accounts, indent=2, sort_keys=True)
        self.state_file.write_text(snapshot, encoding="utf-8")

    def ensure_account(self, account_id: str) -> dict[str, Any]:
        if not account_id:
            raise ValueError("account_id required")
        with self._lock:
            account = self._accounts.setdefault(
                account_id,
                {
                    "account_id": account_id,
                    "tier": "FREE",
                    "items": [],
                    "provenance": [],
                    "chain_head": "0" * 64,
                },
            )
        return dict(account)

    def set_tier(self, account_id: str, tier: str) -> bool:
        if tier not in TIERS:
            return False
        account = self.ensure_account(account_id)
        with self._lock:
            self._accounts[account_id]["tier"] = tier
        self._persist()
        return True

    def tier_of(self, account_id: str) -> str:
        return str(self.ensure_account(account_id)["tier"])

    def may_issue(self, account_id: str, item_id: str) -> tuple[bool, str]:
        item = ITEMS.get(item_id)
        if item is None:
            return False, "unknown item"
        account = self.ensure_account(account_id)
        if TIER_RANK[account["tier"]] < TIER_RANK[item["min_tier"]]:
            return False, f"requires {item['min_tier']}"
        if item_id in account["items"]:
            return False, "already issued"
        return True, "ok"

    def issue_item(self, account_id: str, item_id: str) -> dict[str, Any]:
        allowed, reason = self.may_issue(account_id, item_id)
        if not allowed:
            return {"status": "denied", "reason": reason, "item_id": item_id}
        item = ITEMS[item_id]
        with self._lock:
            account = self._accounts[account_id]
            payload = json.dumps(
                {
                    "account_id": account_id,
                    "item_id": item_id,
                    "tier": account["tier"],
                    "previous": account["chain_head"],
                },
                sort_keys=True,
            ).encode("utf-8")
            record_hash = hashlib.sha256(payload).hexdigest()
            record = {
                "item_id": item_id,
                "name": item["name"],
                "category": item["category"],
                "tier_at_issue": account["tier"],
                "previous": account["chain_head"],
                "hash": record_hash,
            }
            account["items"].append(item_id)
            account["provenance"].append(record)
            account["chain_head"] = record_hash
        self._persist()
        return {"status": "issued", "item_id": item_id, "record": record}

    def verify_provenance(self, account_id: str) -> bool:
        account = self.ensure_account(account_id)
        previous = "0" * 64
        for record in account["provenance"]:
            payload = json.dumps(
                {
                    "account_id": account_id,
                    "item_id": record["item_id"],
                    "tier": record["tier_at_issue"],
                    "previous": previous,
                },
                sort_keys=True,
            ).encode("utf-8")
            if hashlib.sha256(payload).hexdigest() != record["hash"]:
                return False
            previous = record["hash"]
        return previous == account["chain_head"]

    def export_account(self, account_id: str) -> dict[str, Any]:
        account = self.ensure_account(account_id)
        return {
            "account": dict(account),
            "provenance_valid": self.verify_provenance(account_id),
            "item_definitions": {key: dict(value) for key, value in ITEMS.items()},
        }
