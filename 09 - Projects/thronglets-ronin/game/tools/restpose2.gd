extends SceneTree

var _skel: Skeleton3D = null
var _anim: AnimationPlayer = null

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	_scan(inst)
	if _skel:
		var n := _skel.get_bone_count()
		for i in range(n):
			var rk := _skel.get_bone_rest(i)
			var rq := rk.basis.get_rotation_quaternion()
			print("BONE: ", _skel.get_bone_name(i), " rot=(", rq.x, ",", rq.y, ",", rq.z, ",", rq.w, ") origin=(", rk.origin.x, ",", rk.origin.y, ",", rk.origin.z, ")")
	if _anim:
		print("ANIMS: ", _anim.get_animation_list())
	quit()

func _scan(node: Node) -> void:
	if node is Skeleton3D:
		_skel = node
	if node is AnimationPlayer:
		_anim = node
	for c in node.get_children():
		_scan(c)
