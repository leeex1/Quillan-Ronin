extends SceneTree

var _anim: AnimationPlayer = null

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	_scan(inst)
	for anim_name in ["Walk", "Attack", "Celebrate"]:
		var a: Animation = _anim.get_animation(anim_name)
		print(anim_name, " tracks=", a.get_track_count(), " length=", a.length)
		for t in range(a.get_track_count()):
			var p: String = str(a.track_get_path(t))
			if p.contains("Hips") and p.contains("rotation"):
				print("   HIPS_ROT key0=", a.track_get_key_value(t, 0))
	quit()

func _scan(node: Node) -> void:
	if node is AnimationPlayer:
		_anim = node
	for c in node.get_children():
		_scan(c)
