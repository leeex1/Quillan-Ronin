extends SceneTree

func _initialize() -> void:
	var d := EditorDock.new()
	for p in d.get_property_list():
		var n: String = p.name
		if n.contains("slot") or n.contains("title") or n.contains("default") or n.contains("layout"):
			print("DOCKPROP: ", n, " type=", p.type)
	quit()
