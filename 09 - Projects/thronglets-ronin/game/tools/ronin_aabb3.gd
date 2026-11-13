extends SceneTree

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	var res := {"min": Vector3(1e9, 1e9, 1e9), "max": Vector3(-1e9, -1e9, -1e9)}
	_walk(inst, Transform3D.IDENTITY, res)
	print("AABB_MIN_BODY: ", res["min"])
	print("AABB_MAX_BODY: ", res["max"])
	print("SIZE_BODY: ", res["max"] - res["min"])
	quit()

func _walk(node: Node, xf: Transform3D, res: Dictionary) -> void:
	for child in node.get_children():
		var nx := xf
		if child is Node3D:
			nx = xf * child.transform
		if child is MeshInstance3D:
			if child.name == "Icosphere":
				continue
			var aabb: AABB = child.mesh.get_aabb()
			for ax in [aabb.position.x, aabb.position.x + aabb.size.x]:
				for ay in [aabb.position.y, aabb.position.y + aabb.size.y]:
					for az in [aabb.position.z, aabb.position.z + aabb.size.z]:
						var wp = nx * Vector3(ax, ay, az)
						res["min"] = (res["min"] as Vector3).min(wp)
						res["max"] = (res["max"] as Vector3).max(wp)
		_walk(child, nx, res)
