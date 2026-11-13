extends SceneTree
## Probe: does the ronin GLB contain an AnimationPlayer and our clips?

func _init():
	var scene = load("res://assets/ronin.glb")
	if scene == null:
		print("PROBE: FAILED to load ronin.glb")
		quit()
		return
	var inst = scene.instantiate()
	print("PROBE: root type = ", inst.get_class(), " name = ", inst.name)
	var ap = inst.find_child("AnimationPlayer", true, false) as AnimationPlayer
	if ap:
		print("PROBE: AnimationPlayer found, animations = ", ap.get_animation_list())
	else:
		print("PROBE: NO AnimationPlayer found. Children:")
		_dump(inst, 0)
	quit()

func _dump(n: Node, depth: int) -> void:
	if depth > 3:
		return
	print("  ".repeat(depth), n.name, " <", n.get_class(), ">")
	for c in n.get_children():
		_dump(c, depth + 1)
