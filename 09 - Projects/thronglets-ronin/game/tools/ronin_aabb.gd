extends SceneTree

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	var mi: MeshInstance3D
	_find(inst)
	func_dummy()
	quit()

func func_dummy() -> void:
	var scene2 = load("res://assets/ronin.glb")
	var inst2 = scene2.instantiate()
	var min_p := Vector3.INF
	var max_p := -Vector3.INF
	_accumulate(inst2, Transform3D.IDENTITY, min_p, max_p)
	print("AABB_MIN: ", min_p)
	print("AABB_MAX: ", max_p)
	print("AABB_SIZE: ", max_p - min_p)
	inst2.queue_free()

func _find(node: Node) -> void:
	pass

func _accumulate(node: Node, xform: Transform3D, min_p: Vector3, max_p: Vector3) -> void:
	for child in node.get_children():
		var next := xform
		if child is Node3D:
			next = xform * child.transform
		if child is MeshInstance3D:
			var aabb: AABB = child.mesh.get_aabb()
			for corner in _corners(aabb):
				var wp: Vector3 = next * corner
				min_p = Vector3(min(min_p.x, wp.x), min(min_p.y, wp.y), min(min_p.z, wp.z))
				max_p = Vector3(max(max_p.x, wp.x), max(max_p.y, wp.y), max(max_p.z, wp.z))
		_accumulate(child, next, min_p, max_p)

func _corners(aabb: AABB) -> Array:
	var p: Array[Vector3] = []
	for x in [aabb.position.x, aabb.position.x + aabb.size.x]:
		for y in [aabb.position.y, aabb.position.y + aabb.size.y]:
			for z in [aabb.position.z, aabb.position.z + aabb.size.z]:
				p.append(Vector3(x, y, z))
	return p
