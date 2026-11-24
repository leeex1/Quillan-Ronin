extends SceneTree
## Verifies the connected nine-zone world layout.

func _initialize() -> void:
	var world: Node3D = Node3D.new()
	world.set_script(load("res://worlds/world_stream.gd"))
	root.add_child(world)
	world._ready()
	if world.zone_count() != 9 or world.portal_count() != 9:
		push_error("[FAIL] expected 9 zones and 9 portals, got " + str(world.zone_count()) + "/" + str(world.portal_count()))
		quit(1)
		return
	var first: Node = world.get_node_or_null("Zone_starting_zone_town_square")
	var last: Node = world.get_node_or_null("Zone_live_event_arena")
	if first == null or last == null:
		push_error("[FAIL] named zone nodes missing")
		quit(1)
		return
	print("[PASS] connected world: 9 zones + 9 portals instantiated")
	print("[PASS] world stream verification complete")
	quit(0)
