extends Node
## Local loopback session + peer command bridge.
## This is NOT a Tripio integration: Tripio is an image-to-3D asset generator.
## A real multiplayer backend has not been selected yet; these are the contracts
## the rest of the runtime programs against.

var connected: bool = false
var session_id: String = ""
var peer_count: int = 0
var _inboxes: Dictionary = {}

func connect_session(requested_id: String = "") -> bool:
	session_id = requested_id if requested_id != "" else "local-offline-session"
	connected = true
	peer_count = 0
	_inboxes.clear()
	session_event_emit("connected", {"session_id": session_id})
	return connected

func disconnect_session() -> void:
	connected = false
	peer_count = 0
	_inboxes.clear()
	session_event_emit("disconnected", {})

func register_peer(peer_id: String) -> bool:
	if peer_id.is_empty():
		return false
	_inboxes[peer_id] = []
	peer_count = _inboxes.size()
	return true

func send_command(target_peer: String, command: Dictionary) -> bool:
	if not _inboxes.has(target_peer):
		return false
	(_inboxes[target_peer] as Array).append(command.duplicate(true))
	return true

func poll_inbox(peer_id: String) -> Array:
	if not _inboxes.has(peer_id):
		return []
	var messages: Array = (_inboxes[peer_id] as Array).duplicate(true)
	_inboxes[peer_id] = []
	return messages

func session_event_emit(event: String, payload: Dictionary) -> void:
	if Engine.has_singleton("CommandBus"):
		Engine.get_singleton("CommandBus").session_event.emit(event, payload)
