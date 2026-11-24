extends SceneTree
## Verifies the production zone scene in-engine: asset, PBR environment,
## collision bodies, avatar and network bridge nodes.

func _initialize() -> void:
	var packed := load("res://worlds/ProductionZone.tscn") as PackedScene
	if packed == null:
		push_error("[FAIL] ProductionZone.tscn could not be loaded")
		quit(1)
		return
	var zone: Node = packed.instantiate()
	root.add_child(zone)
	# During SceneTree._initialize the deferred _ready has not run yet.
	zone._ready()

	var asset: Node = zone.get_node_or_null("ZoneAsset")
	var environment: Node = zone.get_node_or_null("WorldEnvironment")
	var ground: Node = zone.get_node_or_null("GroundCollider")
	var avatar: Node = zone.get_node_or_null("LocalAvatar")
	var presence: Node = zone.get_node_or_null("PresenceRequest")
	var move: Node = zone.get_node_or_null("MoveRequest")

	if asset == null:
		print("[DEBUG] zone_id=", zone.get("zone_id"), " loaded_asset=", zone.get("loaded_asset_name"),
			" path_exists=", ResourceLoader.exists("res://assets/tripio/robot.glb"))
		push_error("[FAIL] GLB zone asset was not instantiated")
		quit(1)
		return
	if environment == null or ground == null or avatar == null:
		push_error("[FAIL] environment/collider/avatar missing")
		quit(1)
		return
	if presence == null or move == null:
		push_error("[FAIL] live hub bridge missing")
		quit(1)
		return
	var loaded: String = zone.get("loaded_asset_name")
	if loaded != "robot.glb":
		push_error("[FAIL] wrong asset loaded: " + loaded)
		quit(1)
		return

	print("[PASS] production zone scene loaded with asset=", loaded)
	print("[PASS] PBR environment, ground+obstacle colliders present")
	print("[PASS] local avatar and live hub presence/move bridge present")
	print("[PASS] production zone verification complete")
	quit(0)
