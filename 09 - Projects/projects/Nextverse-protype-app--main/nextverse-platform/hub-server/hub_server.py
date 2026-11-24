#!/usr/bin/env python3
"""Local multiplayer hub service for NextVerse social spaces.

Uses Flask-SocketIO, already installed on this host. No AI models are loaded;
this is a standalone presence + text service that the web client can talk to.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request as flask_request
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.config["SECRET_KEY"] = __import__("os").environ.get("FLASK_SECRET_KEY", "")
socketio = SocketIO(
    app, cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"]
)

# hub_id -> {session_id: presence record}
_HUBS: dict[str, dict[str, dict[str, Any]]] = {}
_LOCK = threading.Lock()
STATE_FILE = Path(
    os.environ.get(
        "NEXTVERSE_HUB_STATE", Path(__file__).resolve().parent / "hub_state.json"
    )
)


def save_state() -> None:
    with _LOCK:
        snapshot = json.dumps(_HUBS, indent=2)
    STATE_FILE.write_text(snapshot, encoding="utf-8")


def load_state() -> None:
    global _HUBS
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            with _LOCK:
                _HUBS = data
    except (OSError, ValueError):
        return


load_state()


def _serialize(hub_id: str) -> list[dict[str, Any]]:
    return list(_HUBS.get(hub_id, {}).values())


# REST bridge for native clients (Godot/Three.js) that cannot use SocketIO directly.
@app.get("/api/hub/<hub_id>/presence")
def rest_presence(hub_id: str):
    with _LOCK:
        members = _serialize(hub_id)
    return jsonify({"hub_id": hub_id, "members": members})


@app.post("/api/hub/<hub_id>/move")
def rest_move(hub_id: str):
    payload = flask_request.get_json(silent=True) or {}
    character = str(payload.get("character_id", "guest"))
    try:
        x, y, z = float(payload["x"]), float(payload["y"]), float(payload["z"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "invalid position"}), 400
    sid = f"rest:{character}"
    with _LOCK:
        record = _HUBS.setdefault(hub_id, {}).setdefault(
            sid,
            {
                "sid": sid,
                "hub_id": hub_id,
                "zone_id": str(payload.get("zone_id", hub_id)),
                "character_id": character,
                "presence_state": "online",
                "last_seen": time.time(),
            },
        )
        record["position"] = {"x": x, "y": y, "z": z}
        record["last_seen"] = time.time()
    save_state()
    return jsonify({"status": "ok", "character_id": character})


@app.post("/api/hub/<hub_id>/enter")
def rest_enter_zone(hub_id: str):
    payload = flask_request.get_json(silent=True) or {}
    character = str(payload.get("character_id", "guest"))
    zone = str(payload.get("zone_id", hub_id))
    sid = f"rest:{character}"
    with _LOCK:
        record = _HUBS.setdefault(hub_id, {}).setdefault(
            sid,
            {
                "sid": sid,
                "hub_id": hub_id,
                "zone_id": zone,
                "character_id": character,
                "presence_state": "online",
                "last_seen": time.time(),
            },
        )
        record["zone_id"] = zone
        record["last_seen"] = time.time()
    save_state()
    return jsonify({"status": "ok", "character_id": character, "zone_id": zone})


@socketio.on("join_hub")
def on_join(payload: dict[str, Any]) -> None:
    hub_id = str(payload.get("hub_id", "lobby"))
    zone_id = str(payload.get("zone_id", "starting_zone_town_square"))
    character_id = str(payload.get("character_id", "guest"))
    sid = __import__("flask").request.sid
    with _LOCK:
        _HUBS.setdefault(hub_id, {})[sid] = {
            "sid": sid,
            "hub_id": hub_id,
            "zone_id": zone_id,
            "character_id": character_id,
            "presence_state": "online",
            "last_seen": time.time(),
        }
        members = _serialize(hub_id)
    join_room(hub_id)
    save_state()
    emit("presence_update", {"hub_id": hub_id, "members": members}, to=hub_id)
    print(f"[hub] {character_id} joined {hub_id} ({len(members)} present)")


@socketio.on("hub_chat")
def on_chat(payload: dict[str, Any]) -> None:
    hub_id = str(payload.get("hub_id", "lobby"))
    body = str(payload.get("body", ""))[:500]
    sid = __import__("flask").request.sid
    if not body.strip():
        return
    with _LOCK:
        record = _HUBS.get(hub_id, {}).get(sid)
    if record is None:
        emit("hub_error", {"message": "join a hub before chatting"})
        return
    emit(
        "hub_message",
        {
            "hub_id": hub_id,
            "character_id": record["character_id"],
            "body": body,
            "timestamp": time.time(),
        },
        to=hub_id,
    )


@socketio.on("player_move")
def on_player_move(payload: dict[str, Any]) -> None:
    """Broadcast an avatar position to everyone else in the same hub."""
    hub_id = str(payload.get("hub_id", "lobby"))
    sid = __import__("flask").request.sid
    with _LOCK:
        record = _HUBS.get(hub_id, {}).get(sid)
        if record is None:
            return
        try:
            x, y, z = float(payload["x"]), float(payload["y"]), float(payload["z"])
        except (KeyError, TypeError, ValueError):
            return
        record["position"] = {"x": x, "y": y, "z": z}
    save_state()
    emit(
        "player_moved",
        {
            "sid": sid,
            "hub_id": hub_id,
            "character_id": record["character_id"],
            "x": x,
            "y": y,
            "z": z,
        },
        to=hub_id,
        include_self=False,
    )


@socketio.on("leave_hub")
def on_leave(payload: dict[str, Any]) -> None:
    hub_id = str(payload.get("hub_id", "lobby"))
    sid = __import__("flask").request.sid
    with _LOCK:
        _HUBS.get(hub_id, {}).pop(sid, None)
        members = _serialize(hub_id)
    leave_room(hub_id)
    save_state()
    emit("presence_update", {"hub_id": hub_id, "members": members}, to=hub_id)


@socketio.on("disconnect")
def on_disconnect() -> None:
    sid = __import__("flask").request.sid
    affected: list[str] = []
    with _LOCK:
        for hub_id, members in list(_HUBS.items()):
            if sid in members:
                members.pop(sid)
                affected.append(hub_id)
    for hub_id in affected:
        emit(
            "presence_update",
            {"hub_id": hub_id, "members": _serialize(hub_id)},
            to=hub_id,
        )


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=8085, debug=False)
