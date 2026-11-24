"""World-state service verification."""

from __future__ import annotations

import pytest

from world import WorldService, ZONES, NPC_SEEDS, ASSET_BINDINGS


@pytest.fixture
def world() -> WorldService:
    return WorldService()


def test_zone_registry_matches_design_docs() -> None:
    assert len(ZONES) == 9
    assert ZONES["starting_zone_town_square"]["name"] == "Town Square"
    assert ZONES["nexus_atrium"]["tier"] == "civic"
    assert len(NPC_SEEDS) == 4
    print("[PASS] world registry: 9 zones, 4 NPC seeds")


def test_zone_asset_bindings_exist_on_disk() -> None:
    from pathlib import Path

    assets = (
        Path(__file__).resolve().parent.parent
        / "web-ui"
        / "public"
        / "assets"
        / "tripio"
    )
    for filename in ASSET_BINDINGS.values():
        assert (assets / filename).exists(), filename
    print("[PASS] all zone asset bindings point to real files")


def test_zone_entry_and_movement(world: WorldService) -> None:
    state = world.enter_zone("hero-1", "dungeon_level_1")
    assert state.zone_id == "dungeon_level_1"
    assert state.position == ZONES["dungeon_level_1"]["spawn"]
    moved = world.move("hero-1", 1.5, -0.5, 2.5)
    assert moved.position == [1.5, -0.5, 2.5]
    assert world.snapshot()["active_players"] == 1
    print("[PASS] zone entry + movement + snapshot")


def test_zone_transition_repositions_player(world: WorldService) -> None:
    world.enter_zone("hero-2", "starting_zone_market")
    world.move("hero-2", 9.0, 0.0, 9.0)
    state = world.enter_zone("hero-2", "nexus_atrium")
    assert state.position == ZONES["nexus_atrium"]["spawn"]
    print("[PASS] zone transition resets spawn position")


def test_invalid_zone_rejected(world: WorldService) -> None:
    with pytest.raises(KeyError):
        world.enter_zone("hero-3", "not_a_zone")
    with pytest.raises(KeyError):
        world.move("nobody", 0, 0, 0)
    print("[PASS] invalid zone/character rejected")
