extends SceneTree
## Headless verification of Godot migration Phases 0-5.
## Run: godot --headless --path <project> --script res://tests/verify_runtime.gd

func _fail(message: String) -> void:
	push_error("[FAIL] " + message)
	quit(1)

func _initialize() -> void:
	# --- Phase 0: skeleton + DVVE port -------------------------------------
	var director = load("res://autoload/PerformanceDirector.gd").new()
	var rp: float = director.calculate_performance()
	if absf(rp - 952380952.38) > 1.0:
		_fail("DVVE mismatch: " + str(rp))
		return
	print("[PASS] P0 DVVE performance=", rp, " tier=", director.quality_tier(rp))

	var bus = load("res://autoload/CommandBus.gd").new()
	var received: Array = []
	bus.avatar_command.connect(func(id: String, action: String, x: float, y: float, z: float) -> void:
		received.append([id, action, x, y, z])
	)
	bus.publish_avatar_command("avatar-1", "move_to", 1.0, 2.0, 3.0)
	if received.size() != 1 or received[0][0] != "avatar-1" or received[0][2] != 1.0:
		_fail("CommandBus signal contract")
		return
	if bus.history.size() != 1:
		_fail("CommandBus history contract")
		return
	print("[PASS] P0 CommandBus signal + history contract")
	bus.free()

	# --- Phase 1: quality tiers drive the viewport -------------------------
	var viewport: SubViewport = SubViewport.new()
	var tier: int = director.update_from_performance(viewport)
	if tier != 2 or not is_equal_approx(viewport.scaling_3d_scale, 0.85):
		_fail("quality tier did not drive viewport scale")
		return
	if not director.frame_budget_ok(16.0) or director.frame_budget_ok(16.1):
		_fail("frame budget gate")
		return
	print("[PASS] P1 tier=", tier, " scale=", viewport.scaling_3d_scale, " budget gate ok")
	viewport.free()
	director.free()

	# --- Phase 2: physics + particle budget --------------------------------
	var physics = load("res://systems/physics/PhysicsDirector.gd").new()
	var body: Node3D = Node3D.new()
	if not physics.register_body(&"avatar-1", body) or physics.body_count() != 1:
		_fail("physics body registration")
		return
	var task: Dictionary = physics.collision_task(1000)
	if String(task.type) != "ADVANCED_PHYSICS_COLLISION" or int(task.particle_count) != 1000:
		_fail("collision task contract")
		return
	var integrated: Vector3 = physics.integrate(Vector3.ZERO, Vector3(1.0, 0.0, 0.0), 0.5)
	if not is_equal_approx(integrated.x, 0.5):
		_fail("physics integration")
		return
	print("[PASS] P2 physics registration + integration")
	body.free()
	physics.free()

	var budget = load("res://systems/particles/ParticleBudget.gd").new()
	var amount: int = budget.compute_amount(952380952.38, 60.0)
	if amount != 100000:
		_fail("particle clamp expected 100000 got " + str(amount))
		return
	var particles: GPUParticles3D = GPUParticles3D.new()
	budget.configure(particles, 952380952.38, 60.0)
	if particles.amount != 100000:
		_fail("GPUParticles3D amount not applied")
		return
	print("[PASS] P2 particle budget amount=", amount)
	particles.free()
	budget.free()

	# --- Phase 3: local session contract with two loopback peers -----------
	var bridge = load("res://autoload/LocalSessionBridge.gd").new()
	if not bridge.connect_session("verify-session") or bridge.session_id != "verify-session":
		_fail("LocalSessionBridge session contract")
		return
	if not bridge.register_peer("client-a") or not bridge.register_peer("client-b"):
		_fail("LocalSessionBridge peer registration")
		return
	if not bridge.send_command("client-b", {"avatar_id": "a1", "action": "move_to", "x": 4.0}):
		_fail("LocalSessionBridge send")
		return
	var inbox: Array = bridge.poll_inbox("client-b")
	if inbox.size() != 1 or float(inbox[0].x) != 4.0:
		_fail("LocalSessionBridge inbox")
		return
	if bridge.poll_inbox("client-a").size() != 0:
		_fail("LocalSessionBridge cross-peer leak")
		return
	print("[PASS] P3 two-peer command exchange + local session contract")
	bridge.free()

	# --- Tripio asset pipeline (image -> 3D renders) -----------------------
	var assets = load("res://autoload/AssetPipeline.gd").new()
	if not assets.validate_path("res://assets/tripio/nexus_pillar.glb"):
		_fail("Tripio GLB validation")
		return
	if assets.validate_path("res://assets/tripio/notes.txt"):
		_fail("Tripio rejected-extension contract")
		return
	if not assets.register_asset("nexus_pillar", "res://assets/tripio/nexus_pillar.glb") or assets.asset_count() != 1:
		_fail("Tripio asset registration")
		return
	if not assets.has_asset("nexus_pillar") or assets.manifest().size() != 1:
		_fail("Tripio manifest")
		return
	print("[PASS] Tripio render import contract: glb accepted, txt rejected, manifest=1")
	assets.free()

	# --- Phase 4: doc-grounded zones and hubs ------------------------------
	var zones = load("res://worlds/ZoneRegistry.gd").new()
	if zones.legacy_zone_count() != 5 or not zones.has_zone("starting_zone_town_square") or not zones.has_zone("nexus_atrium"):
		_fail("zone registry")
		return
	var zone_scene: PackedScene = load("res://worlds/Zone.tscn")
	var zone = zone_scene.instantiate()
	if not zone.initialize("starting_zone_market"):
		_fail("zone scene init")
		return
	print("[PASS] P4 zones=", zones.all_ids().size(), " legacy=", zones.legacy_zone_count(),
		" scene=", String(zone.name))
	zone.free()
	zones.free()

	var hubs = load("res://worlds/HubRegistry.gd").new()
	if hubs.count() != 6 or not hubs.has_type("DAW jam rooms"):
		_fail("hub registry")
		return
	var hub_scene: PackedScene = load("res://worlds/Hub.tscn")
	var hub = hub_scene.instantiate()
	if not hub.initialize("Creator stages", "live_event_arena"):
		_fail("hub scene init")
		return
	print("[PASS] P4 hub types=", hubs.count(), " scene=", String(hub.name))
	hub.free()
	hubs.free()

	# --- Phase 4b: production scene with PBR + lighting ---------------------
	var builder = load("res://worlds/ZoneBuilder.gd").new()
	if not builder.build("dungeon_level_1"):
		_fail("ZoneBuilder rejected a valid zone")
		return
	if builder.get_child_count() != 1 + 6 + 3:
		_fail("ZoneBuilder child count: " + str(builder.get_child_count()))
		return
	var material: StandardMaterial3D = builder.pillar_material()
	if material == null or not is_equal_approx(material.metallic, 0.8) or not is_equal_approx(material.roughness, 0.25):
		_fail("PBR material contract")
		return
	var sun: DirectionalLight3D = builder.get_node_or_null("Sun") as DirectionalLight3D
	var environment: WorldEnvironment = builder.get_node_or_null("Environment") as WorldEnvironment
	if sun == null or environment == null or environment.environment == null:
		_fail("lighting/environment contract")
		return
	if not environment.environment.fog_enabled:
		_fail("environment fog contract")
		return
	print("[PASS] P4b production scene: PBR metallic=", material.metallic,
		" roughness=", material.roughness, " children=", builder.get_child_count())
	builder.free()

	# --- Phase 5: legacy is reference-only ---------------------------------
	var legacy_cmake: String = "res://../../game_engine/CMakeLists.txt"
	if not FileAccess.file_exists(legacy_cmake):
		_fail("legacy reference CMake missing")
		return
	print("[PASS] P5 legacy engine retained as reference")
	print("[PASS] Godot Phases 0-5 verified")
	quit(0)
