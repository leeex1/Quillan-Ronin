extends Node3D
## Production-scene builder for a zone. Constructs PBR materials, geometry,
## lighting and an environment procedurally so the same code path works in the
## headless test harness and in the rendered game.

const PILLAR_COUNT: int = 6

var zone_id: String = ""
var built: bool = false

func build(id: String) -> bool:
	var registry = load("res://worlds/ZoneRegistry.gd").new()
	if not registry.has_zone(id):
		registry.free()
		return false
	zone_id = id

	var ground_material: StandardMaterial3D = StandardMaterial3D.new()
	ground_material.albedo_color = Color(0.12, 0.14, 0.18, 1.0)
	ground_material.metallic = 0.35
	ground_material.roughness = 0.6

	var ground: MeshInstance3D = MeshInstance3D.new()
	ground.name = "Ground"
	var ground_mesh: BoxMesh = BoxMesh.new()
	ground_mesh.size = Vector3(40.0, 0.5, 40.0)
	ground.mesh = ground_mesh
	ground.material_override = ground_material
	ground.position.y = -0.25
	add_child(ground)

	var pillar_material: StandardMaterial3D = StandardMaterial3D.new()
	pillar_material.albedo_color = Color(0.25, 0.45, 0.85, 1.0)
	pillar_material.metallic = 0.8
	pillar_material.roughness = 0.25
	pillar_material.emission_enabled = true
	pillar_material.emission = Color(0.05, 0.12, 0.3, 1.0)

	for index in range(PILLAR_COUNT):
		var angle: float = TAU * float(index) / float(PILLAR_COUNT)
		var pillar: MeshInstance3D = MeshInstance3D.new()
		pillar.name = "Pillar_%d" % index
		var cylinder: CylinderMesh = CylinderMesh.new()
		cylinder.top_radius = 0.6
		cylinder.bottom_radius = 0.6
		cylinder.height = 6.0
		pillar.mesh = cylinder
		pillar.material_override = pillar_material
		pillar.position = Vector3(cos(angle) * 12.0, 3.0, sin(angle) * 12.0)
		add_child(pillar)

	var sun: DirectionalLight3D = DirectionalLight3D.new()
	sun.name = "Sun"
	sun.light_energy = 1.2
	sun.rotation_degrees = Vector3(-45.0, 30.0, 0.0)
	add_child(sun)

	var fill: OmniLight3D = OmniLight3D.new()
	fill.name = "Fill"
	fill.light_energy = 2.0
	fill.omni_range = 30.0
	fill.position = Vector3(0.0, 8.0, 0.0)
	add_child(fill)

	var world_environment: WorldEnvironment = WorldEnvironment.new()
	world_environment.name = "Environment"
	var environment: Environment = Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.02, 0.03, 0.06, 1.0)
	environment.tonemap_mode = Environment.TONE_MAPPER_ACES
	environment.fog_enabled = true
	environment.fog_density = 0.015
	world_environment.environment = environment
	add_child(world_environment)

	registry.free()
	built = true
	return true

func pillar_material() -> StandardMaterial3D:
	var pillar: MeshInstance3D = get_node_or_null("Pillar_0") as MeshInstance3D
	if pillar == null:
		return null
	return pillar.material_override as StandardMaterial3D
