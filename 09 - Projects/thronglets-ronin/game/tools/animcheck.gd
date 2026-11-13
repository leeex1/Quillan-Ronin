extends SceneTree

var _anim: AnimationPlayer = null
var _skel: Skeleton3D = null

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	_scan(inst)
	if _anim:
		for anim_name in ["Walk", "Attack"]:
`tvar a: Animation = _anim.get_animation(anim_name)
		print(anim_name, " tracks=", a.get_track_count(), " length=", a.length)
		for t in range(a.get_track_count()):
			var path: String = str(a.track_get_path(t))
			var rot = a.track_get_key_value(t, 0)
			print("T: ", path, " key0=", rot)
	quit()

func _scan(node: Node) -> void:
	if node is Skeleton3D:
		_skel = node
	if node is AnimationPlayer:
		_anim = node
	for c in node.get_children():
		_scan(c)
