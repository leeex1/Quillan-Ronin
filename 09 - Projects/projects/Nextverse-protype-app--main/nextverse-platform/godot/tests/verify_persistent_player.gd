extends SceneTree
## Verifies persistent player state and zone transitions.

func _initialize() -> void:
	var player: Node = Node.new()
	player.set_script(load("res://worlds/persistent_player.gd"))
	root.add_child(player)

	player.quest_snapshot = {"1": 4}
	player.entitlement_snapshot = {"tier": "EXPLORER", "items": ["skin_neon_ronin"]}
	if not player.transition_to("nexus_atrium"):
		push_error("[FAIL] zone transition failed")
		quit(1)
		return
	if player.zone_id != "nexus_atrium":
		push_error("[FAIL] zone id not updated")
		quit(1)
		return
	var second: Node = Node.new()
	second.set_script(load("res://worlds/persistent_player.gd"))
	root.add_child(second)
	if not second.load_saved():
		push_error("[FAIL] saved player state could not be loaded")
		quit(1)
		return
	if second.zone_id != "nexus_atrium" or second.quest_snapshot.get("1", 0) != 4:
		push_error("[FAIL] persisted state mismatch")
		quit(1)
		return
	if second.entitlement_snapshot.get("tier", "") != "EXPLORER":
		push_error("[FAIL] entitlement snapshot not persisted")
		quit(1)
		return
	print("[PASS] player state persists across reload")
	print("[PASS] zone transition updates and saves state")
	print("[PASS] persistent multiplayer player verification complete")
	quit(0)
