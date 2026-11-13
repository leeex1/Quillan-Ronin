extends SceneTree

func _initialize() -> void:
	var scene = load("res://assets/ronin.glb")
	var inst = scene.instantiate()
	var anim: AnimationPlayer
	var skel: Skeleton3D
	_scan(inst)
	if skel:
		var n := skel.get_bone_count()
		for i in range(n):
			var rq := skel.get_bone_rest(i).basis.get_rotation_quaternion()
			var rk := skel.get_bone_rest(i).origin
			print("BONE: ", skel.get_bone_name(i), " rest_rot=", rq, " origin=", rk)
	quit()

func _scan(node: Node) -> void:
	if node is AnimationPlayer:
		var ap = node
		print("ANIMS: ", ap.get_animation_list())
	if node is Skeleton3D:
		skel_global(node)
	for c in node.get_children():
		_scan(c)

func skel_global(s) -> void:
	pass
