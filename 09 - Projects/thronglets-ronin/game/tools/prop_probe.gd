extends SceneTree

func _initialize() -> void:
	var m := ProceduralSkyMaterial.new()
	for p in m.get_property_list():
		var n: String = p.name
		if n.contains("sun") or n.contains("curve") or n.contains("color") or n.contains("ground") or n.contains("sky"):
			print("SKYPROP: ", n)
	var e := Environment.new()
	for p in e.get_property_list():
		var n: String = p.name
		if n.begins_with("glow") or n.begins_with("ssao") or n.begins_with("volumetric") or n.begins_with("adjustment") or n.begins_with("tonemap") or n.begins_with("fog_"):
			print("ENVPROP: ", n)
	quit()
