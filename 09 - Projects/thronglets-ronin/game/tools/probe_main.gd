extends SceneTree

func _initialize() -> void:
	var scene = load("res://scenes/Main.tscn")
	var inst = scene.instantiate()
	root.add_child(inst)
	print("PROBE: Main children =", inst.get_child_count())
	for c in inst.get_children():
		var info: String = c.name + " <" + c.get_class() + ">"
		if c is MeshInstance3D:
			info += " mesh=" + str(c.mesh != null)
		if c is Camera3D:
			info += " pos=" + str(c.position) + " current=" + str(c.current)
		print("PROBE:   ", info)
	var gs = root.get_node("/root/GameState")
	var h0: float = gs.terrain_height(0, 0)
	print("PROBE: terrain_height(0,0) = ", h0)
	for z in [-60.0, 0.0, 60.0]:
		print("PROBE: terrain_height(0, %d) = %s" % [z, gs.terrain_height(0, z)])
	quit()
