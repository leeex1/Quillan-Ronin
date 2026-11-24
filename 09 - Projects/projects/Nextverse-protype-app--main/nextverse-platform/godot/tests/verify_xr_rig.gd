extends SceneTree
## Verifies the OpenXR-capable rig instantiates and produces the same
## movement contract as desktop input, without requiring a headset.

func _initialize() -> void:
	var rig: Node = XROrigin3D.new()
	rig.set_script(load("res://xr/xr_rig.gd"))
	root.add_child(rig)
	rig._ready()

	if rig.get_node_or_null("XRCamera") == null:
		push_error("[FAIL] XR camera missing")
		quit(1)
		return
	if rig.get_node_or_null("LeftController") == null or rig.get_node_or_null("RightController") == null:
		push_error("[FAIL] XR controllers missing")
		quit(1)
		return
	print("[PASS] OpenXR-capable rig instantiates with camera + two controllers")

	var parity: Node = Node.new()
	parity.set_script(load("res://worlds/parity_input.gd"))
	root.add_child(parity)
	var xr_vector: Vector3 = rig.movement_from_axes(Vector2(0.707, -0.707), Vector2.ZERO)
	var desktop_vector: Vector3 = parity.from_keyboard({"w": true, "d": true})
	if parity.parity_delta(desktop_vector, xr_vector, 0.016) > 0.001:
		push_error("[FAIL] XR rig movement differs from desktop contract")
		quit(1)
		return
	print("[PASS] XR rig movement matches desktop parity contract")
	print("[PASS] OpenXR-capable parity path verification complete (no headset)")
	quit(0)
