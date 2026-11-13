extends Node3D
## Main scene: floating cyber-temple dojo, terraced island, structures, graves, blight, glitches.

@onready var camera: Camera3D = $Camera3D
@onready var hud: CanvasLayer = $HUD
@onready var ronins: Node3D = $Ronins
@onready var sun: DirectionalLight3D = $Sun
@onready var dojo_light: OmniLight3D = $DojoLight

var _cam_yaw: float = 0.0
var _cam_pitch: float = 0.52
var _cam_dist: float = 15.5
var _dragging: bool = false
var _moon: MeshInstance3D
var _torii: Node3D
var _flash_energy: float = 0.0
var _shake: float = 0.0
var _blight_light: OmniLight3D
var _glitch_timer: float = 6.0
var _carousels: Array[Node3D] = []
var _rune_rings: Array[MeshInstance3D] = []
var _dojo_pulse: float = 0.0
var _structure_nodes: Array[Node3D] = []
var _base_fog_color := Color(0.34, 0.21, 0.25)

const RONIN_SCENE := "res://scripts/RoninUnit.gd"

func _ready() -> void:
	_build_world()
	GameState.game_over.connect(_on_game_over)
	GameState.log_message.connect(func(t): hud.push_log(t))
	Dialogue.question_posed.connect(func(q): hud.show_interview(q))
	GameState.demand_available.connect(func(stage): hud.show_demand(stage))
	_place_nodes()
	spawn_ronin()  # the Seed â€” one lonely spark
	hud.update_all()

func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky := Sky.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.09, 0.04, 0.22)
	sky_mat.sky_horizon_color = Color(0.85, 0.36, 0.22)
	sky_mat.sky_curve = 0.11
	sky_mat.sky_energy_multiplier = 0.6
	sky_mat.ground_bottom_color = Color(0.03, 0.015, 0.04)
	sky_mat.ground_horizon_color = Color(0.42, 0.14, 0.16)
	sky_mat.ground_curve = 0.18
	sky_mat.sun_angle_max = 1.8
	sky_mat.sun_curve = 0.05
	sky.sky_material = sky_mat
	env.sky = sky
	# painterly dusk: warm key, pink atmospheric haze, hot sun glare
	env.fog_enabled = true
	env.fog_light_color = _base_fog_color
	env.fog_density = 0.0035
	env.fog_aerial_perspective = 0.3
	env.fog_sun_scatter = 0.1
	env.fog_height = 0.0
	env.fog_height_density = 0.06
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.5, 0.33, 0.37)
	env.ambient_light_energy = 0.95
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.tonemap_exposure = 1.0
	# bloom — every ember and emissive trim sings
	env.glow_enabled = true
	env.glow_blend_mode = Environment.GLOW_BLEND_MODE_SOFTLIGHT
	env.glow_hdr_threshold = 0.95
	env.glow_hdr_scale = 1.6
	env.glow_intensity = 0.4
	env.glow_strength = 1.05
	env.glow_mix = 0.08
	env.glow_bloom = 0.1
	# contact shading so forms sit in the world
	env.ssao_enabled = true
	env.ssao_radius = 0.55
	env.ssao_intensity = 1.35
	env.ssao_power = 1.5
	env.ssao_light_affect = 0.25
	env.ssao_horizon = 1.15
	# volumetric dusk haze
	env.volumetric_fog_enabled = true
	env.volumetric_fog_density = 0.0021
	env.volumetric_fog_albedo = Color(0.4, 0.25, 0.3)
	env.volumetric_fog_emission = Color(0.06, 0.02, 0.05)
	env.volumetric_fog_emission_energy = 0.18
	env.volumetric_fog_anisotropy = 0.6
	env.volumetric_fog_length = 96.0
	env.volumetric_fog_ambient_inject = 0.07
	env.volumetric_fog_sky_affect = 0.12
	env.volumetric_fog_temporal_reprojection_enabled = true
	env.volumetric_fog_temporal_reprojection_amount = 0.9
	# final grade: richer, slightly contrasty, saturated like key art
	env.adjustment_enabled = true
	env.adjustment_brightness = 1.0
	env.adjustment_contrast = 1.07
	env.adjustment_saturation = 1.18
	var we := WorldEnvironment.new()
	we.environment = env
	we.name = "WorldEnvironment"
	add_child(we)
	_env_ref = env

	_moon = MeshInstance3D.new()
	_moon.name = "Moon"
	var sphere := SphereMesh.new()
	sphere.radius = 3.2
	sphere.height = 6.4
	_moon.mesh = sphere
	var moon_mat := StandardMaterial3D.new()
	moon_mat.albedo_color = Color(0.92, 0.88, 0.78)
	moon_mat.emission_enabled = true
	moon_mat.emission = Color(0.9, 0.86, 0.72)
	moon_mat.emission_energy_multiplier = 1.35
	_moon.material_override = moon_mat
	_moon.position = Vector3(60, 44, 75)
	add_child(_moon)

	_build_terrain()
	_water = MeshInstance3D.new()
	var water_mesh := PlaneMesh.new()
	water_mesh.size = Vector2(1600, 1600)
	water_mesh.subdivide_width = 96
	water_mesh.subdivide_depth = 96
	_water.mesh = water_mesh
	# stylized painterly water — animated vertex waves, hot sunset glitter
	var water_shader := Shader.new()
	water_shader.code = """
shader_type spatial;
render_mode cull_disabled;
uniform vec4 deep_color : source_color = vec4(0.035, 0.14, 0.19, 1.0);
uniform vec4 crest_color : source_color = vec4(0.23, 0.3, 0.36, 1.0);
uniform vec4 glimmer_color : source_color = vec4(1.0, 0.55, 0.35, 1.0);
uniform float wave_speed = 0.7;
uniform float wave_freq = 0.11;
uniform float wave_amp = 0.4;

float hash(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float vnoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(
		mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
		mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
		u.y
	);
}

void vertex() {
	vec2 wp = (VERTEX).xz;
	float w = 0.0;
	w += vnoise(wp * wave_freq + vec2(TIME * wave_speed * 0.8, 0.0)) * 0.65;
	w += vnoise(wp * wave_freq * 2.3 - vec2(0.0, TIME * wave_speed * 1.1)) * 0.35;
	VERTEX.y += (w - 0.5) * wave_amp;
}

void fragment() {
	vec2 uv = UV * 220.0;
	float n1 = vnoise(uv + TIME * wave_speed);
	float n2 = vnoise(uv * 2.6 - TIME * wave_speed * 1.4);
	float n3 = vnoise(uv * 5.1 + TIME * wave_speed * 0.6);
	float wave_band = smoothstep(0.35, 0.8, n1 * 0.6 + n2 * 0.5);
	ALBEDO = mix(deep_color.rgb, crest_color.rgb, wave_band);
	ROUGHNESS = 0.16 + n3 * 0.2;
	METALLIC = 0.22;
	float glint = smoothstep(0.72, 0.95, n2 * 0.5 + n3 * 0.6);
	EMISSION = glimmer_color.rgb * glint * 0.5 + vec3(0.02, 0.03, 0.04);
	ALPHA = 0.94;
}
"""
	var water_mat := ShaderMaterial.new()
	water_mat.shader = water_shader
	_water.material_override = water_mat
	_water.position = Vector3(0, 0.05, 0)
	add_child(_water)


	# floating data runes above the plateau
	_rune_rings = []
	for i in range(3):
		var ring := MeshInstance3D.new()
		var tm := TorusMesh.new()
		tm.inner_radius = 1.2 + i * 0.8
		tm.outer_radius = 1.35 + i * 0.8
		ring.mesh = tm
		var rm := StandardMaterial3D.new()
		rm.albedo_color = Color(0, 0, 0)
		rm.emission_enabled = true
		rm.emission = Color(0.0, 0.85, 0.8)
		rm.emission_energy_multiplier = 0.55
		ring.material_override = rm
		var h0 := GameState.terrain_height(0, 0)
		ring.position = Vector3(0, h0 + 1.5 + i * 1.6, 0)
		add_child(ring)
		_rune_rings.append(ring)

	# scatter the living world â€” trees, boulders
	_scatter_world()
	_build_particles()

	# torii 
	_torii = Node3D.new()
	_torii.name = "Torii"
	_torii.position = Vector3(0, GameState.terrain_height(0, -5.5) + 0.1, -5.5)
	add_child(_torii)
	var torii_mat := StandardMaterial3D.new()
	torii_mat.albedo_color = Color(0.5, 0.045, 0.035)
	torii_mat.roughness = 0.3
	torii_mat.metallic = 0.12
	torii_mat.clearcoat_enabled = true
	torii_mat.clearcoat = 0.75
	torii_mat.clearcoat_roughness = 0.16
	for sx in [-1, 1]:
		var pillar := MeshInstance3D.new()
		var cyl := CylinderMesh.new()
		cyl.top_radius = 0.26; cyl.bottom_radius = 0.3; cyl.height = 4.4
		pillar.mesh = cyl
		pillar.material_override = torii_mat
		pillar.position = Vector3(2.0 * sx, 2.2, 0)
		_torii.add_child(pillar)
	var beam1 := MeshInstance3D.new()
	var b1 := BoxMesh.new()
	b1.size = Vector3(5.9, 0.42, 0.5)
	beam1.mesh = b1
	beam1.material_override = torii_mat
	beam1.position = Vector3(0, 4.6, 0)
	_torii.add_child(beam1)
	var beam2 := MeshInstance3D.new()
	var b2 := BoxMesh.new()
	b2.size = Vector3(4.4, 0.3, 0.42)
	beam2.mesh = b2
	beam2.material_override = torii_mat
	beam2.position = Vector3(0, 2.1, 0)
	_torii.add_child(beam2)

	sun.light_color = Color(0.75, 0.65, 0.9)
	sun.light_energy = 1.35
	sun.rotation_degrees = Vector3(-34, -135, 0)
	dojo_light.light_color = Color(1.0, 0.35, 0.25)
	dojo_light.light_energy = 2.4
	dojo_light.omni_range = 9.0
	dojo_light.position = Vector3(0, 7.6, -2.8)

	# blight light â€” purple goo glow, scales with pollution
	_blight_light = OmniLight3D.new()
	_blight_light.light_color = Color(0.65, 0.1, 0.85)
	_blight_light.light_energy = 0.0
	_blight_light.omni_range = 17.0
	_blight_light.position = Vector3(0, 1.5, 0)
	add_child(_blight_light)
	_update_blight_visual()

	# sakura/data petal rain
	var particles := GPUParticles3D.new()
	var pmat := ParticleProcessMaterial.new()
	pmat.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_BOX
	pmat.emission_box_extents = Vector3(7, 1, 7)
	pmat.gravity = Vector3(0.15, -0.55, 0.1)
	pmat.initial_velocity_min = 0.2
	pmat.initial_velocity_max = 0.6
	pmat.scale_min = 0.05
	pmat.scale_max = 0.12
	particles.process_material = pmat
	var quad := QuadMesh.new()
	quad.size = Vector2(0.09, 0.05)
	var pmesh_mat := StandardMaterial3D.new()
	pmesh_mat.albedo_color = Color(1.0, 0.5, 0.65)
	pmesh_mat.emission_enabled = true
	pmesh_mat.emission = Color(0.8, 0.25, 0.4)
	pmesh_mat.emission_energy_multiplier = 0.7
	quad.material = pmesh_mat
	particles.draw_pass_1 = quad
	particles.amount = 140
	particles.lifetime = 7.0
	particles.position = Vector3(0, 5.5, 0)
	add_child(particles)

	_build_vignette()
	_update_camera()
	print("WORLD BUILT: children=", get_child_count(), " terrain=", get_node_or_null("Terrain") != null, " cam_pos=", camera.position if camera else Vector3.ZERO)

var _env_ref: Environment
var _water: MeshInstance3D

# ============ planetary terrain ============

func _build_terrain() -> void:
	var n := 96
	var size := GameState.WORLD_SIZE
	var step := size / float(n)
	var hg: Array[float] = []
	hg.resize((n + 1) * (n + 1))
	for z in range(n + 1):
		for x in range(n + 1):
			var wx := -size / 2.0 + float(x) * step
			var wz := -size / 2.0 + float(z) * step
			hg[z * (n + 1) + x] = GameState.terrain_height(wx, wz)
	var verts := PackedVector3Array()
	var normals := PackedVector3Array()
	var colors := PackedColorArray()
	var indices := PackedInt32Array()
	for z in range(n + 1):
		for x in range(n + 1):
			var wx := -size / 2.0 + float(x) * step
			var wz := -size / 2.0 + float(z) * step
			var h := hg[z * (n + 1) + x]
			verts.append(Vector3(wx, h, wz))
			var hx0: float = hg[z * (n + 1) + maxi(0, x - 1)]
			var hx1: float = hg[z * (n + 1) + mini(n, x + 1)]
			var hz0: float = hg[maxi(0, z - 1) * (n + 1) + x]
			var hz1: float = hg[mini(n, z + 1) * (n + 1) + x]
			var nrm := Vector3(hx0 - hx1, 2.0 * step, hz0 - hz1).normalized()
			normals.append(nrm)
			var c: Color
			if h < 1.15:
				c = Color(0.62, 0.53, 0.34)
			elif h < 3.6:
				c = Color(0.17, 0.33, 0.25)
			elif h < 5.8:
				c = Color(0.44, 0.41, 0.46)
			else:
				c = Color(0.9, 0.88, 0.92)
			var jitter := 0.85 + randf() * 0.3
			colors.append(Color(c.r * jitter, c.g * jitter, c.b * jitter))
	for z in range(n):
		for x in range(n):
			var a := z * (n + 1) + x
			var b := a + 1
			var c2 := a + (n + 1)
			var d := c2 + 1
			indices.append(a)
			indices.append(c2)
			indices.append(b)
			indices.append(b)
			indices.append(c2)
			indices.append(d)
	var arrays: Array = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var mi := MeshInstance3D.new()
	mi.name = "Terrain"
	mi.mesh = mesh
	var m := StandardMaterial3D.new()
	m.vertex_color_use_as_albedo = true
	m.cull_mode = BaseMaterial3D.CULL_DISABLED
	m.roughness = 0.92
	m.metallic = 0.04
	# painterly rocky detail — generated noise, triplanar-projected so slopes look right
	var noise := FastNoiseLite.new()
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	noise.frequency = 0.055
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = 4
	var detail := NoiseTexture2D.new()
	detail.width = 256
	detail.height = 256
	detail.seamless = true
	detail.normalize = true
	detail.noise = noise
	m.albedo_texture = detail
	m.uv1_triplanar = true
	m.uv1_scale = Vector3(3.2, 3.2, 3.2)
	mi.material_override = m
	add_child(mi)

func _build_vignette() -> void:
	# radial dusk vignette — frames the world like key art
	var img := Image.create(512, 512, false, Image.FORMAT_RGBA8)
	for y in range(512):
		for x in range(512):
			var dx := x / 255.5 - 1.0
			var dy := y / 255.5 - 1.0
			var d := sqrt(dx * dx + dy * dy)
			var a := clampf((d - 0.58) / 0.62, 0.0, 1.0)
			a = pow(a, 1.7) * 0.6
			img.set_pixel(x, y, Color(0.05, 0.005, 0.09, a))
	var tex := ImageTexture.create_from_image(img)
	var cl := CanvasLayer.new()
	cl.layer = 1
	var tr := TextureRect.new()
	tr.texture = tex
	tr.set_anchors_preset(Control.PRESET_FULL_RECT)
	tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	tr.stretch_mode = TextureRect.STRETCH_SCALE
	tr.mouse_filter = Control.MOUSE_FILTER_IGNORE
	cl.add_child(tr)
	add_child(cl)

func _build_particles() -> void:
	var h0 := GameState.terrain_height(0, 0)
	# embers rising from the torii — the old gate still breathes
	var embers := CPUParticles3D.new()
	embers.amount = 70
	embers.lifetime = 3.4
	embers.emission_shape = CPUParticles3D.EMISSION_SHAPE_SPHERE
	embers.emission_sphere_radius = 2.6
	embers.direction = Vector3(0, 1, 0)
	embers.spread = 22.0
	embers.initial_velocity_min = 0.9
	embers.initial_velocity_max = 2.2
	embers.gravity = Vector3(0, 0.35, 0)
	embers.scale_amount_min = 0.015
	embers.scale_amount_max = 0.055
	embers.damping_min = 0.2
	embers.damping_max = 0.6
	embers.angular_velocity_min = -40.0
	embers.angular_velocity_max = 40.0
	var ember_mesh := SphereMesh.new()
	ember_mesh.radius = 0.5
	ember_mesh.height = 1.0
	var ember_mat := StandardMaterial3D.new()
	ember_mat.albedo_color = Color(0, 0, 0)
	ember_mat.emission_enabled = true
	ember_mat.emission = Color(1.0, 0.45, 0.12)
	ember_mat.emission_energy_multiplier = 2.6
	ember_mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	ember_mesh.material = ember_mat
	embers.draw_pass_1 = ember_mesh
	embers.position = Vector3(0, h0 + 0.8, -2.8)
	add_child(embers)
	# violet petals drifting down from the ridge trees
	var petals := CPUParticles3D.new()
	petals.amount = 46
	petals.lifetime = 6.5
	petals.emission_shape = CPUParticles3D.EMISSION_SHAPE_BOX
	petals.emission_box_extents = Vector3(24.0, 2.0, 24.0)
	petals.direction = Vector3(0, -1, 0)
	petals.spread = 30.0
	petals.initial_velocity_min = 0.4
	petals.initial_velocity_max = 1.1
	petals.gravity = Vector3(0, -0.9, 0)
	petals.scale_amount_min = 0.05
	petals.scale_amount_max = 0.11
	petals.angular_velocity_min = -90.0
	petals.angular_velocity_max = 90.0
	var petal_mesh := QuadMesh.new()
	petal_mesh.size = Vector2(0.22, 0.13)
	var petal_mat := StandardMaterial3D.new()
	petal_mat.albedo_color = Color(0.55, 0.25, 0.85)
	petal_mat.emission_enabled = true
	petal_mat.emission = Color(0.42, 0.16, 0.72)
	petal_mat.emission_energy_multiplier = 1.0
	petal_mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	petal_mesh.material = petal_mat
	petals.draw_pass_1 = petal_mesh
	petals.position = Vector3(0, h0 + 7.5, 0)
	add_child(petals)

func _scatter_world() -> void:
	seed(4242)
	# One draw call per part-type for the whole forest — InstancedMesh, not 350 nodes.
	var tree_mat := _mat(Color(0.08, 0.22, 0.17), 0.85)
	var trunk_mat := _mat(Color(0.14, 0.09, 0.06), 0.9)
	var rock_mat := _mat(Color(0.33, 0.31, 0.36), 0.55)
	var snow_mat := _mat(Color(0.85, 0.82, 0.8), 0.7)
	var tree_positions: Array[Vector3] = []
	var rock_positions: Array[Vector3] = []
	var trunk_tf: Array[Transform3D] = []
	var crown_tf: Array[Transform3D] = []
	var tip_tf: Array[Transform3D] = []
	var rock_tf: Array[Transform3D] = []
	for i in range(110):
		var ang := randf() * TAU
		var rad := 13.0 + randf() * 110.0
		var x := cos(ang) * rad
		var z := sin(ang) * rad
		var h := GameState.terrain_height(x, z)
		if h > 6.5:
			continue  # too cold for the dark pines
		var crown_scale := randf_range(0.8, 1.35)
		var crown_h := randf_range(0.9, 1.35)
		var yaw := randf() * TAU
		var basis := Basis.IDENTITY.rotated(Vector3.UP, yaw).scaled(Vector3(crown_scale, crown_h, crown_scale))
		trunk_tf.append(Transform3D(Basis.IDENTITY, Vector3(x, h + 0.55, z)))
		var crown_center := 1.15 + (2.5 * crown_h) * 0.5
		crown_tf.append(Transform3D(basis, Vector3(x, h + crown_center, z)))
		var tip_basis := Basis.IDENTITY.rotated(Vector3.UP, yaw).scaled(Vector3(crown_scale * 0.42, crown_h * 0.42, crown_scale * 0.42))
		var tip_y := h + crown_center + (2.5 * crown_h) * 0.38
		tip_tf.append(Transform3D(tip_basis, Vector3(x, tip_y, z)))
		tree_positions.append(Vector3(x, h, z))
	for i in range(70):
		var ang := randf() * TAU
		var rad := 15.0 + randf() * 108.0
		var x := cos(ang) * rad
		var z := sin(ang) * rad
		var h := GameState.terrain_height(x, z)
		var yaw := randf() * TAU
		var rsx := randf_range(0.75, 1.7)
		var rsy := randf_range(0.6, 1.3)
		var rsz := randf_range(0.75, 1.7)
		var basis := Basis.IDENTITY.rotated(Vector3.UP, yaw).rotated(Vector3(1, 0, 0), randf_range(-0.25, 0.25)).scaled(Vector3(rsx, rsy, rsz))
		rock_tf.append(Transform3D(basis, Vector3(x, h + 0.2, z)))
		rock_positions.append(Vector3(x, h, z))
	_emit_multimesh("Trunks", trunk_mat, CylinderMesh.new(), trunk_tf, func(m: CylinderMesh):
		m.top_radius = 0.18; m.bottom_radius = 0.26; m.height = 1.1)
	_emit_multimesh("Crowns", tree_mat, CylinderMesh.new(), crown_tf, func(m: CylinderMesh):
		m.top_radius = 0.0; m.bottom_radius = 1.05; m.height = 2.5)
	_emit_multimesh("SnowTips", snow_mat, CylinderMesh.new(), tip_tf, func(m: CylinderMesh):
		m.top_radius = 0.0; m.bottom_radius = 1.05; m.height = 2.5)
	_emit_multimesh("Boulders", rock_mat, CylinderMesh.new(), rock_tf, func(m: CylinderMesh):
		m.top_radius = 0.42; m.bottom_radius = 0.55; m.height = 0.9; m.radial_segments = 6)
	# mark 8 interactive nodes â€” 4 trees (wood), 4 rocks (gems) â€” with glow
	for i in range(10):
		if i < tree_positions.size():
			var p := tree_positions[randi() % tree_positions.size()]
			GameState.nodes.append({"type": "tree", "pos": p + Vector3(0, 2.0, 0)})
	for i in range(7):
		if i < rock_positions.size():
			var p := rock_positions[randi() % rock_positions.size()]
			GameState.nodes.append({"type": "rock", "pos": p + Vector3(0, 0.7, 0)})

func _emit_multimesh(node_name: String, mat: StandardMaterial3D, mesh: Mesh, xforms: Array[Transform3D], configure: Callable) -> void:
	if xforms.is_empty():
		return
	configure.call(mesh)
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.mesh = mesh
	mm.instance_count = xforms.size()
	for i in range(xforms.size()):
		mm.set_instance_transform(i, xforms[i])
	var mmi := MultiMeshInstance3D.new()
	mmi.name = node_name
	mmi.multimesh = mm
	mmi.material_override = mat
	mmi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	add_child(mmi)

func _pick_scatter_slot(occupied: Array[Vector3]) -> Vector3:
	var p := Vector3.ZERO
	var tries := 0
	while tries < 40:
		tries += 1
		var ang := randf() * TAU
		var rad := 20.0 + randf() * 80.0
		var x := cos(ang) * rad
		var z := sin(ang) * rad
		p = Vector3(x, GameState.terrain_height(x, z), z)
		var ok := true
		for o in occupied:
			if o.distance_to(p) < 12.0:
				ok = false
				break
		if ok:
			return p
	return p

func _add_terrace(radius: float, y: float, color: Color) -> void:
	var t := MeshInstance3D.new()
	var cyl := CylinderMesh.new()
	cyl.top_radius = radius
	cyl.bottom_radius = radius + 0.25
	cyl.height = 0.8
	t.mesh = cyl
	t.material_override = _mat(color, 0.7)
	t.position = Vector3(0, y, 0)
	add_child(t)

func _mat(color: Color, rough: float) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.roughness = rough
	m.metallic = 0.35
	return m

func _emissive(color: Color, energy: float) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(0, 0, 0)
	m.emission_enabled = true
	m.emission = color
	m.emission_energy_multiplier = energy
	return m

# ---------- structures ----------

const SLOTS_USED := 12  # planet build sites

func build_structure_request(type_id: String) -> void:
	if GameState.structures.size() >= SLOTS_USED:
		hud.push_log("The world is full. No room remains â€” not even for our ambitions.")
		return
	if not GameState.build_structure(type_id):
		return
	var slot := _pick_scatter_slot(GameState.structure_positions)
	_spawn_structure_visual(type_id, slot)
	GameState.structure_positions.append(slot)
	GameState.emit_log("Built: %s." % GameState.STRUCT_NAMES.get(type_id, type_id))

func _spawn_structure_visual(type_id: String, slot: Vector3) -> void:
	var node := Node3D.new()
	node.position = slot
	match type_id:
		"orchard":
			node.add_child(_part(CylinderMesh.new(), Vector3(0, 0.25, 0), _mat(Color(0.2, 0.1, 0.05), 0.8), Vector3(0.42, 0.5, 0.42)))
			node.add_child(_part(SphereMesh.new(), Vector3(0, 0.85, 0), _emissive(Color(0.0, 0.9, 0.75), 1.4), Vector3(0.5, 0.5, 0.5)))
		"baths":
			node.add_child(_part(BoxMesh.new(), Vector3(0, 0.3, 0), _mat(Color(0.45, 0.46, 0.5), 0.25), Vector3(1.3, 0.55, 0.9)))
			node.add_child(_part(CylinderMesh.new(), Vector3(0, 0.58, 0), _emissive(Color(0.35, 0.85, 1.0), 1.8), Vector3(1.0, 0.06, 0.62)))
		"carousel":
			var base := _part(CylinderMesh.new(), Vector3(0, 0.18, 0), _mat(Color(0.3, 0.08, 0.3), 0.5), Vector3(1.1, 0.35, 1.1))
			node.add_child(base)
			var top := _part(TorusMesh.new(), Vector3(0, 0.62, 0), _emissive(Color(1.0, 0.3, 0.8), 1.6), Vector3(1.0, 1.0, 1.0))
			var spin := Node3D.new()
			spin.add_child(top)
			top.position = Vector3.ZERO
			spin.position = Vector3(0, 0.62, 0)
			node.add_child(spin)
			_carousels.append(spin)
		"den":
			node.add_child(_part(BoxMesh.new(), Vector3(0, 0.5, 0), _mat(Color(0.12, 0.06, 0.06), 0.7), Vector3(1.4, 1.0, 1.1)))
			var door := _part(BoxMesh.new(), Vector3(0, 0.32, 0.56), _emissive(Color(0.8, 0.08, 0.05), 0.8), Vector3(0.36, 0.55, 0.02))
			node.add_child(door)
		"forge":
			node.add_child(_part(CylinderMesh.new(), Vector3(0, 0.5, 0), _mat(Color(0.06, 0.06, 0.08), 0.6), Vector3(0.85, 1.1, 0.85)))
			node.add_child(_part(SphereMesh.new(), Vector3(0, 1.12, 0), _emissive(Color(0.75, 0.1, 0.95), 2.2), Vector3(0.4, 0.3, 0.4)))
		"scrubber":
			node.add_child(_part(CylinderMesh.new(), Vector3(0, 0.6, 0), _mat(Color(0.75, 0.75, 0.78), 0.3), Vector3(0.55, 1.3, 0.55)))
			node.add_child(_part(TorusMesh.new(), Vector3(0, 1.28, 0), _emissive(Color(0.1, 0.95, 0.9), 2.4), Vector3(0.8, 0.8, 0.8)))
		"theater":
			node.add_child(_part(BoxMesh.new(), Vector3(0, 0.62, 0), _mat(Color(0.3, 0.05, 0.1), 0.65), Vector3(1.9, 1.25, 1.4)))
			node.add_child(_part(BoxMesh.new(), Vector3(0, 1.32, 0), _mat(Color(0.12, 0.04, 0.06), 0.6), Vector3(2.15, 0.28, 1.6)))
			node.add_child(_part(BoxMesh.new(), Vector3(0, 0.55, 0.71), _emissive(Color(0.95, 0.35, 0.15), 1.1), Vector3(1.5, 0.75, 0.03)))
	add_child(node)
	_structure_nodes.append(node)

func _part(mesh: Mesh, pos: Vector3, material: StandardMaterial3D, scale_vec: Vector3) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = material
	mi.position = pos
	mi.scale = scale_vec
	return mi

# ---------- harvest nodes ----------

func _place_nodes() -> void:
	for d in GameState.nodes:
		var p: Vector3 = d.pos
		var base := Vector3(p.x, GameState.terrain_height(p.x, p.z), p.z)
		var node := Node3D.new()
		node.position = base
		if d.type == "tree":
			var trunk := MeshInstance3D.new()
			var tc := CylinderMesh.new()
			tc.top_radius = 0.14; tc.bottom_radius = 0.18; tc.height = 0.72
			trunk.mesh = tc
			trunk.material_override = _mat(Color(0.16, 0.09, 0.05), 0.8)
			trunk.position.y = 0.36
			node.add_child(trunk)
			var crown := MeshInstance3D.new()
			var sm := SphereMesh.new()
			sm.radius = 0.5; sm.height = 1.0
			crown.mesh = sm
			crown.material_override = _emissive(Color(0.05, 0.75, 0.65), 1.15)
			crown.position.y = 1.05
			crown.scale = Vector3(1, 1.25, 1)
			node.add_child(crown)
		else:
			var rock := MeshInstance3D.new()
			var rb := BoxMesh.new()
			rb.size = Vector3(0.62, 0.48, 0.55)
			rock.mesh = rb
			rock.material_override = _mat(Color(0.32, 0.3, 0.38), 0.5)
			rock.scale = Vector3(randf_range(0.85, 1.25), randf_range(0.75, 1.1), randf_range(0.85, 1.25))
			rock.rotation_degrees = Vector3(randf_range(-14, 14), randf_range(0, 360), randf_range(-14, 14))
			rock.position.y = 0.22
			var glint := MeshInstance3D.new()
			var gb := BoxMesh.new()
			gb.size = Vector3(0.2, 0.06, 0.06)
			glint.mesh = gb
			glint.material_override = _emissive(Color(0.7, 0.25, 0.95), 2.0)
			glint.position = Vector3(0.0, 0.4, 0.24)
			glint.rotation = rock.rotation
			node.add_child(rock)
			node.add_child(glint)
		add_child(node)

func spark_spar() -> void:
	GameState.try_spar()

# ---------- graves ----------

func _on_unit_died(unit_name: String, at: Vector3) -> void:
	var grave := Node3D.new()
	var stone := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = Vector3(0.34, 0.62, 0.12)
	stone.mesh = box
	stone.material_override = _mat(Color(0.28, 0.28, 0.3), 0.9)
	stone.position = Vector3(0, 0.31, 0)
	stone.rotation_degrees = Vector3(randf_range(-9, 9), randf_range(0, 360), randf_range(-11, 11))
	grave.add_child(stone)
	var slit := MeshInstance3D.new()
	var sbox := BoxMesh.new()
	sbox.size = Vector3(0.2, 0.045, 0.13)
	slit.mesh = sbox
	slit.material_override = _emissive(Color(0.0, 0.9, 0.85), 1.6)
	slit.position = Vector3(0, 0.48, 0)
	slit.rotation = stone.rotation
	grave.add_child(slit)
	grave.position = Vector3(at.x, _floor_at(at), at.z)
	add_child(grave)

func _floor_at(at: Vector3) -> float:
	return GameState.terrain_height(at.x, at.z)

# ---------- ronins ----------

func spawn_ronin() -> RoninUnit:
	var u := RoninUnit.new()
	var sx := randf_range(-2, 2)
	var sz := randf_range(-2, 2)
	u.position = Vector3(sx, GameState.terrain_height(sx, sz), sz)
	ronins.add_child(u)
	GameState.register_unit(u)
	u.evolved.connect(func(_u, t): hud.push_log("â¬† %s evolved to %s!" % [_u.display_name(), ["Ashigaru", "Samurai", "DaimyÅ"][t]]))
	u.died.connect(_on_unit_died)
	u.mitosis_ready.connect(_on_mitosis)
	return u

func _on_mitosis(parent_unit: RoninUnit) -> void:
	GameState.play_sfx("mitosis", -3.0)
	var at := parent_unit.global_position
	var u := spawn_ronin()
	u.position = Vector3(at.x + randf_range(-0.4, 0.4), at.y, at.z + randf_range(-0.4, 0.4))
	u.hunger = 60.0
	u.mood = 70.0
	hud.push_log("â˜¼ MITOSIS â€” %s splits. A new heart ignites beneath the torii." % parent_unit.display_name())

func feed_all() -> void:
	var cost := 1.0 * GameState.units.size()
	if not GameState.spend_wood(cost):
		hud.push_log("Not enough wood to feed the horde. (The thralls harvest it themselves.)")
		return
	for u in GameState.units:
		if is_instance_valid(u):
			u.feed()
	GameState.play_sfx("feed", -6.0)
	GameState.record_choice("fed_everyone", 0.0, 0.5)
	GameState.learn_concept("food")
	hud.push_log("Morsels dispensed. Visors hum contentment.")

func train_all() -> void:
	for u in GameState.units:
		if is_instance_valid(u):
			u.train()
	hud.push_log("Sparring at the dojo. Blades flash in the dark.")

func cleanse_all() -> void:
	if not GameState.spend_alloy(15.0):
		hud.push_log("Need 15 alloy to purge corruption.")
		return
	for u in GameState.units:
		if is_instance_valid(u):
			u.cleanse()
	GameState.add_corruption(-6.0)
	GameState.play_sfx("cleanse", -6.0)
	GameState.record_choice("purification", 0.0, 1.0)
	GameState.learn_concept("clean")
	hud.push_log("Purification rite complete. Teal light rekindled.")

func replicate() -> void:
	if not GameState.can_replicate():
		hud.push_log("The island's Dens are full â€” build Thrall Dens to house more (capacity %d)." % GameState.capacity())
		return
	if not GameState.spend_alloy(150.0):
		hud.push_log("Replication requires 150 alloy (and courage).")
		return
	spawn_ronin()
	GameState.play_sfx("feed", -5.0, 1.35)
	GameState.record_choice("replicate", 0.0, 0.5)
	hud.push_log("A newborn spark wakes beneath the toriiâ€¦")

func toggle_sleep() -> void:
	GameState.sleeping = not GameState.sleeping
	GameState.learn_concept("rest")
	dojo_light.light_energy = 0.6 if GameState.sleeping else 2.4
	hud.push_log("The dojo dims. The horde dreams of steel." if GameState.sleeping else "Dawn protocol. Eyes open.")
	hud.refresh_sleep_button()

func swarm_ability() -> void:
	if GameState.try_swarm_ability():
		_flash_energy = 5.0
		_shake = maxf(_shake, 0.5)
		var burst := GPUParticles3D.new()
		var pmat := ParticleProcessMaterial.new()
		pmat.emission_shape = ParticleProcessMaterial.EMISSION_SHAPE_SPHERE
		pmat.emission_sphere_radius = 0.5
		pmat.initial_velocity_min = 6.0
		pmat.initial_velocity_max = 11.0
		pmat.gravity = Vector3.ZERO
		pmat.scale_min = 0.15
		pmat.scale_max = 0.3
		burst.process_material = pmat
		burst.one_shot = true
		burst.explosiveness = 1.0
		burst.amount = 160
		burst.lifetime = 1.4
		var quad := QuadMesh.new()
		quad.size = Vector2(0.2, 0.2)
		var bm := StandardMaterial3D.new()
		bm.albedo_color = Color(1, 0.2, 0.15)
		bm.emission_enabled = true
		bm.emission = Color(1, 0.1, 0.05)
		bm.emission_energy_multiplier = 3.0
		quad.material = bm
		burst.draw_pass_1 = quad
		burst.position = Vector3(0, 3.0, 0)
		add_child(burst)
		hud.push_log("The Torii drinks the light. The Ronin Swarm answers.")
		hud.update_swarm_button()

# ---------- frame loop ----------

func _process(delta: float) -> void:
	_cam_yaw += delta * 0.06
	if Input.is_action_pressed("orbit_left"):
		_cam_yaw += delta * 1.6
	if Input.is_action_pressed("orbit_right"):
		_cam_yaw -= delta * 1.6
	if Input.is_action_pressed("zoom_in"):
		_cam_dist = clampf(_cam_dist - delta * 8.0, 5.0, 160.0)
	if Input.is_action_pressed("zoom_out"):
		_cam_dist = clampf(_cam_dist + delta * 8.0, 5.0, 160.0)
	if _flash_energy > 0.0:
		_flash_energy = maxf(0.0, _flash_energy - delta * 6.0)
		dojo_light.light_energy = (0.6 if GameState.sleeping else 2.4) + _flash_energy
	if _shake > 0.0:
		_shake = maxf(0.0, _shake - delta * 1.4)
	for c in _carousels:
		c.rotate_y(delta * 1.6)
	_dojo_pulse += delta * 1.8
	for j in range(_rune_rings.size()):
		_rune_rings[j].rotate_y(delta * (0.6 + j * 0.35) * (1 if j % 2 == 0 else -1))
		_rune_rings[j].position.y = GameState.terrain_height(0, 0) + 1.5 + j * 1.6 + sin(_dojo_pulse * 1.3 + j) * 0.12
	dojo_light.light_energy = (0.6 if GameState.sleeping else 2.4) + _flash_energy + sin(_dojo_pulse * 2.1) * 0.35
	_moon.position.y = 44.0 + sin(Time.get_ticks_msec() * 0.0004) * 0.4
	_glitch_timer -= delta
	if _glitch_timer <= 0.0:
		_glitch_timer = randf_range(5.0, 11.0)
		_maybe_glitch()
	_update_camera()
	_update_blight_visual()
	hud.tick(delta)

func _maybe_glitch() -> void:
	var c := GameState.corruption
	if c < 45.0:
		return
	var intensity := (c - 45.0) / 55.0
	if randf() < 0.65:
		_shake = maxf(_shake, 0.18 + intensity * 0.35)
		dojo_light.light_color = Color(1.0, 0.08, 0.05)
		var tween := create_tween()
		tween.tween_interval(0.22)
		tween.tween_callback(func(): dojo_light.light_color = Color(1.0, 0.35, 0.25))
		if randf() < 0.35:
			GameState.emit_log(["The torii shivers.", "You hear twelve heartbeats in a dead speaker.", "The moon's reflection blinks first.", "Something beneath the island turns over in its sleep."][randi() % 4])

var _fog_tween: Tween
func _update_blight_visual() -> void:
	var b := GameState.blight / 100.0
	if _blight_light:
		_blight_light.light_energy = b * 3.2
	if _env_ref:
		_env_ref.fog_light_color = _base_fog_color.lerp(Color(0.42, 0.12, 0.5), b)
		_env_ref.ambient_light_energy = 0.7 - b * 0.35

func _update_camera() -> void:
	var h0 := GameState.terrain_height(0, 0)
	var target := Vector3(0, h0 + 1.6, 0)
	var offset := Vector3(
		cos(_cam_yaw) * cos(_cam_pitch),
		sin(_cam_pitch),
		sin(_cam_yaw) * cos(_cam_pitch)
	) * _cam_dist
	var shake_offset := Vector3.ZERO
	if _shake > 0.0:
		shake_offset = Vector3(randf_range(-1, 1), randf_range(-1, 1), randf_range(-1, 1)) * _shake * 0.35
	camera.position = target + offset + shake_offset
	camera.look_at(target + shake_offset * 0.5, Vector3.UP)

func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT or event.button_index == MOUSE_BUTTON_RIGHT:
			_dragging = event.pressed
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			_cam_dist = clampf(_cam_dist - 1.2, 5.0, 160.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_cam_dist = clampf(_cam_dist + 1.2, 5.0, 160.0)
	elif event is InputEventMouseMotion and _dragging:
		_cam_yaw -= event.relative.x * 0.005
		_cam_pitch = clampf(_cam_pitch + event.relative.y * 0.003, 0.15, 1.1)
	elif event.is_action_pressed("swarm_ability"):
		swarm_ability()
	if event.is_action_pressed("toggle_console"):
		hud.toggle_console()

func _on_game_over(victory: bool, reason: String, title: String) -> void:
	hud.show_game_over(victory, reason, title)
