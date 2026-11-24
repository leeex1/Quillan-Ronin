extends Node3D
## Production zone: PBR environment, real GLB assets, collision, and live hub presence.

@export var zone_id: String = "starting_zone_town_square"
@export var hub_url: String = "http://127.0.0.1:8085"

const ASSET_DIR := "res://assets/tripio"
const ASSET_BY_ZONE := {
	"starting_zone_town_square": "robot.glb",
	"starting_zone_market": "gem.glb",
	"dungeon_level_1": "pillar.glb",
	"nexus_atrium": "structure.glb",
}

var loaded_asset_name: String = ""
var remote_count: int = 0
var local_avatar: CharacterBody3D
var _presence_http: HTTPRequest
var _move_http: HTTPRequest
var _poll_timer: Timer
var _move_elapsed: float = 0.0

func _ready() -> void:
	_setup_environment()
	_load_zone_asset()
	_setup_collision()
	_setup_avatar()
	_setup_network()

func _setup_environment() -> void:
	var world_environment := WorldEnvironment.new()
	world_environment.name = "WorldEnvironment"
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.02, 0.03, 0.06, 1.0)
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color(0.35, 0.42, 0.55, 1.0)
	environment.ambient_light_energy = 0.45
	environment.tonemap_mode = Environment.TONE_MAPPER_ACES
	environment.fog_enabled = true
	environment.fog_density = 0.015
	world_environment.environment = environment
	add_child(world_environment)

	var sun := DirectionalLight3D.new()
	sun.name = "Sun"
	sun.light_energy = 1.3
	sun.rotation_degrees = Vector3(-45.0, 30.0, 0.0)
	add_child(sun)

func _load_zone_asset() -> void:
	var file_name: String = ASSET_BY_ZONE.get(zone_id, "")
	var instance: Node = null
	if not file_name.is_empty():
		var path: String = ASSET_DIR + "/" + file_name
		if ResourceLoader.exists(path):
			var loaded: Resource = load(path)
			if loaded is PackedScene:
				instance = (loaded as PackedScene).instantiate()
			if instance != null:
				loaded_asset_name = file_name
	if instance == null:
		# Procedural landmark for zones without a generated GLB yet.
		var mesh := MeshInstance3D.new()
		var sphere := SphereMesh.new()
		sphere.radius = 1.2
		mesh.mesh = sphere
		var material := StandardMaterial3D.new()
		material.metallic = 0.8
		material.roughness = 0.25
		material.emission_enabled = true
		material.emission = Color(0.05, 0.15, 0.3)
		mesh.material_override = material
		instance = mesh
		loaded_asset_name = "procedural"
	instance.name = "ZoneAsset"
	instance.position = Vector3(0.0, 0.0, -2.0)
	add_child(instance)

func _setup_collision() -> void:
	var ground := StaticBody3D.new()
	ground.name = "GroundCollider"
	var ground_shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(40.0, 0.5, 40.0)
	ground_shape.shape = box
	ground.add_child(ground_shape)
	ground.position.y = -0.25
	add_child(ground)

	for index in range(3):
		var obstacle := StaticBody3D.new()
		obstacle.name = "Obstacle_%d" % index
		var shape := CollisionShape3D.new()
		var cylinder := CylinderShape3D.new()
		cylinder.radius = 1.2
		cylinder.height = 4.0
		shape.shape = cylinder
		obstacle.add_child(shape)
		var angle: float = TAU * float(index) / 3.0
		obstacle.position = Vector3(cos(angle) * 6.0, 1.0, sin(angle) * 6.0)
		add_child(obstacle)

func _setup_avatar() -> void:
	local_avatar = CharacterBody3D.new()
	local_avatar.name = "LocalAvatar"
	var body_mesh := MeshInstance3D.new()
	var capsule := CapsuleMesh.new()
	capsule.radius = 0.3
	capsule.height = 1.5
	body_mesh.mesh = capsule
	var material := StandardMaterial3D.new()
	material.metallic = 0.6
	material.roughness = 0.25
	body_mesh.material_override = material
	local_avatar.add_child(body_mesh)
	local_avatar.position = Vector3(0.0, 0.0, 4.0)
	add_child(local_avatar)

func _setup_network() -> void:
	_presence_http = HTTPRequest.new()
	_presence_http.name = "PresenceRequest"
	add_child(_presence_http)
	_presence_http.request_completed.connect(_on_presence)
	_move_http = HTTPRequest.new()
	_move_http.name = "MoveRequest"
	add_child(_move_http)
	_poll_timer = Timer.new()
	_poll_timer.wait_time = 1.0
	_poll_timer.autostart = true
	_poll_timer.timeout.connect(_poll_presence)
	add_child(_poll_timer)

func _poll_presence() -> void:
	var url: String = hub_url + "/api/hub/" + zone_id + "/presence"
	_presence_http.request(url)

func _on_presence(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code != 200:
		return
	var parsed: Variant = JSON.parse_string(body.get_string_from_utf8())
	if parsed == null or not parsed is Dictionary:
		return
	var members: Array = parsed.get("members", [])
	remote_count = members.size()
	for child in get_children():
		if child.name.begins_with("RemoteAvatar_"):
			child.queue_free()
	for member in members:
		var record: Dictionary = member
		if String(record.get("character_id", "")).begins_with("godot-hero"):
			continue
		var avatar := MeshInstance3D.new()
		avatar.name = "RemoteAvatar_%s" % String(record.get("character_id", "peer"))
		var sphere := SphereMesh.new()
		sphere.radius = 0.35
		avatar.mesh = sphere
		var position_data: Dictionary = record.get("position", {})
		avatar.position = Vector3(float(position_data.get("x", 0.0)),
			float(position_data.get("y", 0.0)), float(position_data.get("z", 0.0)))
		add_child(avatar)

func _physics_process(delta: float) -> void:
	if local_avatar == null:
		return
	var speed: float = 3.0 * delta
	var direction := Vector3.ZERO
	if Input.is_key_pressed(KEY_W): direction.z -= 1.0
	if Input.is_key_pressed(KEY_S): direction.z += 1.0
	if Input.is_key_pressed(KEY_A): direction.x -= 1.0
	if Input.is_key_pressed(KEY_D): direction.x += 1.0
	if direction != Vector3.ZERO:
		local_avatar.position += direction.normalized() * speed
	local_avatar.position.x = clampf(local_avatar.position.x, -18.0, 18.0)
	local_avatar.position.z = clampf(local_avatar.position.z, -18.0, 18.0)
	_move_elapsed += delta
	if _move_elapsed >= 0.12 and _move_http.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
		_move_elapsed = 0.0
		var payload := {
			"character_id": "godot-hero",
			"x": local_avatar.position.x,
			"y": local_avatar.position.y,
			"z": local_avatar.position.z,
		}
		_move_http.request(hub_url + "/api/hub/" + zone_id + "/move",
			["Content-Type: application/json"], HTTPClient.METHOD_POST, JSON.stringify(payload))
