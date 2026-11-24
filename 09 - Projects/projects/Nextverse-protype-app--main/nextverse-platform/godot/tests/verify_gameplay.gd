extends SceneTree
## Verifies Godot gameplay: quests, NPCs, zone transitions.

func _initialize() -> void:
	var manager: Node = Node.new()
	manager.set_script(load("res://worlds/world_manager.gd"))
	manager.enable_hub_network = false  # offline harness: no hub server
	root.add_child(manager)
	manager._ready()
	manager.enter_zone("starting_zone_town_square")  # deterministic starting state

	if manager.npc_count() != 1:
		push_error("[FAIL] town square should have 1 NPC, got " + str(manager.npc_count()))
		quit(1)
		return
	var guard: Node = manager.get_node_or_null("NPC_Guard_Thompson")
	if guard != null:
		guard._ready()
	if guard == null or not String(guard.interact()).contains("Guard Thompson"):
		var names: Array = []
		for child in manager.get_children():
			names.append(child.name)
		print("[DEBUG] children=", names, " guard=", guard)
		push_error("[FAIL] NPC identity/dialogue")
		quit(1)
		return
	print("[PASS] NPC spawn + dialogue for starting zone")

	var quests: Node = manager.get_node_or_null("QuestSystem")
	if quests == null:
		push_error("[FAIL] quest system missing")
		quit(1)
		return
	if quests.QUESTS.size() != 3:
		push_error("[FAIL] three design-doc quests expected")
		quit(1)
		return
	if not quests.transition(1, 1) or not quests.accept(1):
		push_error("[FAIL] quest eligibility/accept failed")
		quit(1)
		return
	if quests.transition(1, 4):
		push_error("[FAIL] forward-only guard failed")
		quit(1)
		return
	if not quests.transition(1, 3) or not quests.complete(1) or quests.completed_count() != 1:
		push_error("[FAIL] quest completion flow")
		quit(1)
		return
	print("[PASS] 3 quests, forward-only transitions, completion persistence")

	manager.enter_zone("starting_zone_forest_edge")
	var active_npcs: int = 0
	for child in manager.get_children():
		if child.name.begins_with("NPC_") and not child.is_queued_for_deletion():
			active_npcs += 1
	if active_npcs != 2:
		push_error("[FAIL] forest edge should have 2 active NPCs, got " + str(active_npcs))
		quit(1)
		return
	print("[PASS] zone transition respawns correct NPCs")
	if manager.zone_transitions < 2:
		push_error("[FAIL] zone transitions not tracked")
		quit(1)
		return
	if manager.player_state == null or manager.player_state.zone_id != "starting_zone_forest_edge":
		push_error("[FAIL] persistent player not tied to transitions")
		quit(1)
		return
	print("[PASS] zone transitions persist player (hub reporting disabled offline)")
	print("[PASS] gameplay verification complete")
	quit(0)
