extends SceneTree

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	var out := []
	_walk(inst, out)
	for line in out:
		print("RONINMAT: ", line)
	quit()

func _walk(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child is MeshInstance3D:
			var mesh: Mesh = child.mesh
			for s in range(mesh.get_surface_count()):
				var mat = mesh.surface_get_material(s)
				var desc := child.name + " surface=" + str(s) + " mat="
				if mat == null:
					desc += "<null>"
				elif mat is StandardMaterial3D:
					var sm: StandardMaterial3D = mat
					desc += "Std albedo=" + str(sm.albedo_color) + " emissive=" + str(sm.emission_enabled) + " ecolor=" + str(sm.emission) + " energy=" + str(sm.emission_energy_multiplier) + " metal=" + str(sm.metallic) + " rough=" + str(sm.roughness)
				else:
					desc += mat.get_class()
				out.append(desc)
		_walk(child, out)
