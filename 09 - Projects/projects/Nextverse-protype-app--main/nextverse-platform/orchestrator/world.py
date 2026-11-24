#!/usr/bin/env python3
"""NextVerse world state service.

Single source of truth for zones, spawn points, NPC seeds and traversal rules.
Grounded in docs/MMORPG_UNIVERSE_DESIGN_2026.md section 3 and mirrored by
web-ui/src/data/zones.ts and godot/worlds/ZoneRegistry.gd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ZONES: dict[str, dict[str, Any]] = {
    "starting_zone_town_square": {
        "name": "Town Square",
        "tier": "starter",
        "spawn": [0.0, -0.5, 3.0],
    },
    "starting_zone_market": {
        "name": "Market District",
        "tier": "starter",
        "spawn": [2.0, -0.5, 3.0],
    },
    "starting_zone_forest_edge": {
        "name": "Forest Edge",
        "tier": "starter",
        "spawn": [-4.0, -0.5, 4.0],
    },
    "dungeon_level_1": {
        "name": "Abandoned Mines - Level 1",
        "tier": "dungeon",
        "spawn": [0.0, -0.5, 5.0],
    },
    "unknown_zone": {"name": "The Void", "tier": "special", "spawn": [0.0, -0.5, 0.0]},
    "dungeon_level_2": {
        "name": "Abandoned Mines - Level 2",
        "tier": "dungeon",
        "spawn": [0.0, -0.5, 5.0],
    },
    "nexus_atrium": {
        "name": "Nexus Atrium",
        "tier": "civic",
        "spawn": [0.0, -0.5, 4.0],
    },
    "guild_temple_1": {
        "name": "Guild Temple 1",
        "tier": "guild",
        "spawn": [0.0, -0.5, 4.0],
    },
    "live_event_arena": {
        "name": "Live Event Arena",
        "tier": "event",
        "spawn": [0.0, -0.5, 6.0],
    },
}

NPC_SEEDS: list[dict[str, Any]] = [
    {
        "name": "Guard Thompson",
        "definition_id": 1001,
        "zone_id": "starting_zone_town_square",
    },
    {
        "name": "Mysterious Merchant",
        "definition_id": 1002,
        "zone_id": "starting_zone_market",
    },
    {
        "name": "Forest Goblin Alpha",
        "definition_id": 2001,
        "zone_id": "starting_zone_forest_edge",
    },
    {
        "name": "Forest Goblin Bravo",
        "definition_id": 2001,
        "zone_id": "starting_zone_forest_edge",
    },
]

ASSET_BINDINGS: dict[str, str] = {
    "starting_zone_town_square": "robot.glb",
    "starting_zone_market": "gem.glb",
    "dungeon_level_1": "pillar.glb",
    "nexus_atrium": "structure.glb",
}


@dataclass
class PlayerState:
    character_id: str
    zone_id: str = "starting_zone_town_square"
    position: list[float] = field(default_factory=lambda: [0.0, -0.5, 0.0])


class WorldService:
    def __init__(self) -> None:
        self.players: dict[str, PlayerState] = {}

    def zones(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in ZONES.items()}

    def npcs(self, zone_id: str) -> list[dict[str, Any]]:
        return [dict(npc) for npc in NPC_SEEDS if npc["zone_id"] == zone_id]

    def asset_for(self, zone_id: str) -> str | None:
        return ASSET_BINDINGS.get(zone_id)

    def enter_zone(self, character_id: str, zone_id: str) -> PlayerState:
        if zone_id not in ZONES:
            raise KeyError(f"unknown zone: {zone_id}")
        state = self.players.get(character_id) or PlayerState(character_id=character_id)
        state.zone_id = zone_id
        state.position = list(ZONES[zone_id]["spawn"])
        self.players[character_id] = state
        return state

    def move(self, character_id: str, x: float, y: float, z: float) -> PlayerState:
        state = self.players.get(character_id)
        if state is None:
            raise KeyError(f"unknown character: {character_id}")
        state.position = [x, y, z]
        return state

    def snapshot(self) -> dict[str, Any]:
        return {
            "zone_count": len(ZONES),
            "npc_seeds": len(NPC_SEEDS),
            "asset_bindings": len(ASSET_BINDINGS),
            "active_players": len(self.players),
        }
