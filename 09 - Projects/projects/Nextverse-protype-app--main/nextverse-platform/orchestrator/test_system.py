"""Integration tests for the NextVerse system orchestrator."""

from __future__ import annotations

import pytest

from system import SystemOrchestrator


@pytest.fixture(scope="module")
def orchestrator() -> SystemOrchestrator:
    return SystemOrchestrator()


def test_health_reports_seven_real_services(orchestrator: SystemOrchestrator) -> None:
    health = orchestrator.health()
    assert health["total_count"] == 7
    assert health["available_count"] == 7
    assert health["status"] == "healthy"
    print(
        "[PASS] orchestrator health:",
        health["available_count"],
        "/",
        health["total_count"],
    )


def test_jqld_routing(orchestrator: SystemOrchestrator) -> None:
    result = orchestrator.execute("jqld", {"C": 1000.0, "exponent": 10.0, "omega": 1.0})
    assert result["value"] == pytest.approx(1024000.0, rel=1e-9)
    print("[PASS] jqld route value:", result["value"])


def test_real_cpu_batch_routing(orchestrator: SystemOrchestrator) -> None:
    result = orchestrator.execute("real_cpu_batch", {"data": [100] * 12, "workers": 4})
    assert len(result["result"]) == 12
    assert result["report"]["correctness"] == "verified"
    print("[PASS] real CPU batch route:", result["report"])


def test_daw_automation_routing(orchestrator: SystemOrchestrator) -> None:
    result = orchestrator.execute(
        "daw_automation",
        {
            "points": [{"time": 0.0, "value": 0.0}, {"time": 1.0, "value": 1.0}],
            "sample_rate": 100.0,
            "samples": 100,
        },
    )
    assert result["samples"] == 100
    assert result["curve"][0] == pytest.approx(0.0)
    assert result["curve"][-1] == pytest.approx(1.0)
    print("[PASS] DAW automation route samples:", result["samples"])


def test_world_routes(orchestrator: SystemOrchestrator) -> None:
    zones = orchestrator.execute("world_zones", {})
    assert len(zones["zones"]) == 9
    entry = orchestrator.execute(
        "enter_zone", {"character_id": "hero", "zone_id": "nexus_atrium"}
    )
    assert entry["asset"] == "structure.glb"
    print("[PASS] world routes: zones + zone entry with asset binding")


def test_entitlement_routes(orchestrator: SystemOrchestrator) -> None:
    denied = orchestrator.execute(
        "entitlement_issue", {"account_id": "api-user", "item_id": "skin_neon_ronin"}
    )
    assert denied["status"] == "denied"
    orchestrator.execute(
        "entitlement_issue", {"account_id": "api-user", "item_id": "emote_salute"}
    )
    exported = orchestrator.execute("entitlement_export", {"account_id": "api-user"})
    assert exported["provenance_valid"] is True
    print("[PASS] entitlement API routes with provenance verification")


def test_benchmark_route_reports_measured_speedups(
    orchestrator: SystemOrchestrator,
) -> None:
    report = orchestrator.execute("benchmark", {})
    assert report["parallel"]["measured_speedup"] > 0
    assert report["vectorised"]["correctness"] == "verified"
    print(
        "[PASS] benchmark route:",
        report["parallel"]["measured_speedup"],
        report["vectorised"]["measured_speedup"],
    )
