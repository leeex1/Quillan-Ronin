extends SceneTree
## Verifies entitlement client and VR/desktop input parity contract.

func _initialize() -> void:
	var client: Node = Node.new()
	client.set_script(load("res://worlds/entitlement_client.gd"))
	root.add_child(client)
	var response := "{\"status\":\"success\",\"result\":{\"provenance_valid\":true,\"account\":{\"tier\":\"EXPLORER\",\"items\":[\"skin_neon_ronin\"]}}}"
	if not client.parse_export(response):
		push_error("[FAIL] entitlement export parse: " + client.last_error)
		quit(1)
		return
	if client.tier() != "EXPLORER" or not client.owns_item("skin_neon_ronin"):
		push_error("[FAIL] entitlement state")
		quit(1)
		return
	if client.owns_item("hall_golden"):
		push_error("[FAIL] client must not grant unissued items")
		quit(1)
		return
	print("[PASS] entitlement client consumes server authority only")

	var parity: Node = Node.new()
	parity.set_script(load("res://worlds/parity_input.gd"))
	root.add_child(parity)
	var desktop: Vector3 = parity.from_keyboard({"w": true, "d": true})
	var xr: Vector3 = parity.from_xr_axes(Vector2(0.707, -0.707), Vector2.ZERO)
	if parity.parity_delta(desktop, xr, 0.016) > 0.001:
		push_error("[FAIL] desktop/XR movement not equivalent")
		quit(1)
		return
	print("[PASS] desktop and XR paths produce equivalent movement vectors")
	print("[PASS] entitlement + parity verification complete")
	quit(0)
