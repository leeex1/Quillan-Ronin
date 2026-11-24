#!/usr/bin/env python3
"""Live in-process SocketIO tests for the multiplayer hub service."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hub_server import app, socketio, _HUBS  # noqa: E402


def _events(client) -> list[tuple[str, dict]]:
    return [
        (packet["name"], packet["args"][0] if packet["args"] else {})
        for packet in client.get_received()
    ]


def main() -> int:
    _HUBS.clear()
    client_a = socketio.test_client(app)
    client_b = socketio.test_client(app)

    client_a.emit(
        "join_hub",
        {"hub_id": "nexus_atrium", "zone_id": "nexus_atrium", "character_id": "Alice"},
    )
    assert any(
        name == "presence_update" and len(args["members"]) == 1
        for name, args in _events(client_a)
    )

    client_b.emit(
        "join_hub",
        {"hub_id": "nexus_atrium", "zone_id": "nexus_atrium", "character_id": "Bob"},
    )
    b_events = _events(client_b)
    assert any(
        name == "presence_update" and len(args["members"]) == 2
        for name, args in b_events
    )
    a_events = _events(client_a)
    assert any(
        name == "presence_update" and len(args["members"]) == 2
        for name, args in a_events
    )
    print("[PASS] two clients join same hub; presence count = 2 for both")

    client_a.emit("hub_chat", {"hub_id": "nexus_atrium", "body": "hello Bob"})
    b_events = _events(client_b)
    chat = [(name, args) for name, args in b_events if name == "hub_message"]
    assert (
        chat
        and chat[0][1]["body"] == "hello Bob"
        and chat[0][1]["character_id"] == "Alice"
    )
    print("[PASS] room-scoped chat delivered to peer with author")

    _events(client_a)  # drain the earlier room broadcast to Alice
    client_b.emit("hub_chat", {"hub_id": "other_hub", "body": "should not leak"})
    assert not [e for e in _events(client_a) if e[0] == "hub_message"]
    print("[PASS] chat does not leak across hubs")

    # In-game movement is broadcast to peers but not echoed to the sender.
    _events(client_a)
    _events(client_b)
    client_a.emit(
        "player_move", {"hub_id": "nexus_atrium", "x": 1.5, "y": 0, "z": -2.0}
    )
    moved = [args for name, args in _events(client_b) if name == "player_moved"]
    assert moved and moved[0]["x"] == 1.5 and moved[0]["character_id"] == "Alice"
    assert not [name for name, _ in _events(client_a) if name == "player_moved"]
    print("[PASS] in-game player movement broadcast to peers, not echoed to sender")

    client_b.disconnect()
    a_events = _events(client_a)
    assert any(
        name == "presence_update" and len(args["members"]) == 1
        for name, args in a_events
    )
    print("[PASS] disconnect removes presence and broadcasts update")

    # REST bridge used by the native Godot/three.js clients.
    http = app.test_client()
    response = http.post(
        "/api/hub/nexus_atrium/move",
        json={
            "character_id": "godot-hero",
            "x": 1.25,
            "y": -0.5,
            "z": 3.0,
        },
    )
    assert response.status_code == 200
    response = http.get("/api/hub/nexus_atrium/presence")
    assert response.status_code == 200
    members = response.get_json()["members"]
    hero = next(member for member in members if member["character_id"] == "godot-hero")
    assert hero["position"] == {"x": 1.25, "y": -0.5, "z": 3.0}
    print("[PASS] REST presence/move bridge for native clients")

    # Persistent multiplayer state survives a simulated restart.
    import tempfile
    from pathlib import Path
    import hub_server as hub

    with tempfile.TemporaryDirectory() as temporary:
        hub.STATE_FILE = Path(temporary) / "hub_state.json"
        hub._HUBS.clear()
        hub._HUBS["persist_test"] = {
            "rest:hero": {
                "sid": "rest:hero",
                "hub_id": "persist_test",
                "zone_id": "nexus_atrium",
                "character_id": "hero",
                "presence_state": "online",
                "last_seen": 1.0,
                "position": {"x": 2.0, "y": 0.0, "z": 4.0},
            }
        }
        hub.save_state()
        assert hub.STATE_FILE.exists()
        hub._HUBS.clear()
        hub.load_state()
        assert hub._HUBS["persist_test"]["rest:hero"]["position"]["x"] == 2.0
    print("[PASS] persistent multiplayer state survives restart")

    print("[PASS] multiplayer hub verification complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
