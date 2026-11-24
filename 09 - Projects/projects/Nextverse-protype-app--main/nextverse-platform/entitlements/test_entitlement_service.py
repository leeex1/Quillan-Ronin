"""Entitlement authority verification: issuance, tiers, provenance, persistence."""

from __future__ import annotations

import pytest

from entitlement_service import EntitlementService, ITEMS, TIERS


@pytest.fixture
def service(tmp_path):
    return EntitlementService(tmp_path / "state.json")


def test_tier_gating(service: EntitlementService) -> None:
    assert service.tier_of("alice") == "FREE"
    denied = service.issue_item("alice", "skin_neon_ronin")
    assert denied["status"] == "denied" and "EXPLORER" in denied["reason"]
    service.set_tier("alice", "EXPLORER")
    issued = service.issue_item("alice", "skin_neon_ronin")
    assert issued["status"] == "issued"
    print("[PASS] server-side tier gating for exclusive item")


def test_free_item_and_duplicate_rejection(service: EntitlementService) -> None:
    assert service.issue_item("bob", "emote_salute")["status"] == "issued"
    duplicate = service.issue_item("bob", "emote_salute")
    assert duplicate["status"] == "denied" and duplicate["reason"] == "already issued"
    print("[PASS] free item issuance + duplicate rejection")


def test_provenance_chain_valid_and_tamper_detected(
    service: EntitlementService,
) -> None:
    service.set_tier("carol", "SOVEREIGN")
    for item_id in ("emote_salute", "hall_golden"):
        assert service.issue_item("carol", item_id)["status"] == "issued"
    assert service.verify_provenance("carol") is True
    service._accounts["carol"]["provenance"][0]["item_id"] = "hall_golden"
    assert service.verify_provenance("carol") is False
    print("[PASS] append-only provenance verifies and detects tampering")


def test_persistence_across_reload(tmp_path) -> None:
    path = tmp_path / "state.json"
    first = EntitlementService(path)
    first.set_tier("dave", "CREATOR")
    assert first.issue_item("dave", "stage_theme_aurora")["status"] == "issued"
    second = EntitlementService(path)
    assert second.tier_of("dave") == "CREATOR"
    assert "stage_theme_aurora" in second.export_account("dave")["account"]["items"]
    assert second.verify_provenance("dave") is True
    print("[PASS] entitlements + provenance survive restart")


def test_catalog_is_cosmetic_only() -> None:
    categories = {item["category"] for item in ITEMS.values()}
    assert all(
        "weapon" not in category and "power" not in category for category in categories
    )
    assert len(TIERS) == 4
    print("[PASS] catalog contains no power-granting categories")
