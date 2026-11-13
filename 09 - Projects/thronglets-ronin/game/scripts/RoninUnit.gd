class_name RoninUnit
extends Node3D
## A ronin-thrall: autonomous worker of the society. Gathers, tends, duels, splits.

signal died(unit_name: String, at: Vector3)
signal evolved(unit: RoninUnit, new_tier: int)
signal mitosis_ready(unit: RoninUnit)

enum Tier { ASHIGARU = 0, SAMURAI = 1, DAIMYO = 2 }

const TIER_NAMES := ["Ashigaru", "Samurai", "Daimyō"]
const TICK_SECONDS := 1.0
const STARVE_DEATH_TICKS := 45

@export var tier: int = Tier.ASHIGARU
var uid: int = 0
var hunger: float = 75.0
var mood: float = 78.0
var cleanliness: float = 92.0
var age_seconds: float = 0.0
var name_seed: int = 0
var role: String = "Forager"
var valor: float = 0.0
var split_progress: float = 0.0
var _wander_target: Vector3 = Vector3.ZERO
var _wander_timer: float = 0.0
var _tick: float = 0.0
var _base_scale: float = 0.85
var _body: Node3D
var _bob_phase: float = 0.0
var _starve_ticks: int = 0
var _harvest_tick: float = 0.0
var dash_override: Vector3 = Vector3.INF
var dead: bool = false
var _anim: AnimationPlayer
var _oneshot: String = ""
var _chirp: Label3D
var _chirp_timer: float = 0.0

const NAMES := ["Kiro", "Sora", "Renji", "Taro", "Musashi", "Jin", "Kaito", "Ryoma", "Hayate", "Genta", "Seiji", "Isamu", "Tetsuo", "Banten", "Jiro", "Kane"]

func _ready() -> void:
	uid = randi()
	name_seed = randi()
	_bob_phase = randf() * TAU
	_build_body()
	_new_wander_target()

func display_name() -> String:
	if GameState.literacy_tier() < 3:
		# they have no names yet — only signal-glyphs they choose for themselves
		var glyphs := ["◓", "◔", "◑", "●", "◆", "▮", "▼", "×", "≈", "∆"]
		return "%s%s-%02d" % [glyphs[name_seed % glyphs.size()], glyphs[(name_seed / 3) % glyphs.size()], name_seed % 100]
	return "%s-%02d" % [NAMES[name_seed % NAMES.size()], name_seed % 100]

func _build_body() -> void:
	_body = Node3D.new()
	add_child(_body)
	var scene := load("res://assets/ronin.glb")
	if scene:
		var model: Node3D = scene.instantiate()
		model.scale = Vector3.ONE * _base_scale
		model.rotation_degrees = Vector3(-90, 0, 0)
		_body.add_child(model)
		_anim = model.find_child("AnimationPlayer", true, false) as AnimationPlayer
		if _anim:
			for clip in ["Idle", "Walk"]:
				var a := _anim.get_animation(clip)
				if a:
					a.loop_mode = Animation.LOOP_LINEAR
			_anim.play("Idle")
			_anim.animation_finished.connect(_on_anim_finished)
	else:
		var mesh := MeshInstance3D.new()
		mesh.mesh = CapsuleMesh.new()
		_body.add_child(mesh)
	_apply_tier_scale()
	# a chirp-glyph that hangs above them — their only voice until speech
	var chirp := Label3D.new()
	chirp.text = "◓"
	chirp.font_size = 88
	chirp.pixel_size = 0.0062
	chirp.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	chirp.no_depth_test = true
	chirp.modulate = Color(0.65, 1.0, 0.95)
	chirp.position = Vector3(0, 2.35, 0)
	add_child(chirp)
	_chirp = chirp
	_chirp_timer = randf_range(0.5, 2.0)

func _apply_tier_scale() -> void:
	var s: float = _base_scale * [1.0, 1.18, 1.38][tier]
	if _body:
		_body.scale = Vector3.ONE * s

func _process(delta: float) -> void:
	if dead:
		return
	_tick += delta
	_bob_phase += delta * 2.2
	_body.position.y = abs(sin(_bob_phase)) * 0.06

	var moving := false
	if dash_override != Vector3.INF:
		var to := dash_override - global_position
		to.y = 0
		var dist := to.length()
		if dist < 0.35:
			dash_override = Vector3.INF
		else:
			moving = true
			var dir := to / dist
			global_position += dir * 9.0 * delta
			look_at(global_position + dir, Vector3.UP)
	elif role == "Forager":
		moving = _forager_behavior(delta)
	elif role == "Sentry":
		moving = _sentry_behavior(delta)
	elif role == "Tender":
		moving = _tender_behavior(delta)
	elif role == "Forge-Warden":
		moving = _forge_warden_behavior(delta)
	else:
		moving = _wander_behavior(delta)
	_update_anim(moving)
	global_position.y = GameState.terrain_height(global_position.x, global_position.z)
	if _chirp:
		_chirp_timer -= delta
		if _chirp_timer <= 0.0:
			_chirp_timer = randf_range(1.1, 2.6)
			_chirp.text = GameState.chirp_word()
			var m_norm := clampf(mood / 100.0, 0.0, 1.0)
			_chirp.modulate = Color(1.0 - m_norm * 0.45, 0.42 + m_norm * 0.6, 0.72 + m_norm * 0.28)

	if _tick >= TICK_SECONDS:
		_tick -= TICK_SECONDS
		_simulate_tick()

# ============ role behaviors ============

func _nearest_node() -> Dictionary:
	var known_tree := GameState.concepts.has("wood") or GameState.concepts.has("food")
	var known_rock := GameState.concepts.has("gems")
	var best := {}
	var best_d := INF
	for n in GameState.nodes:
		# they only chase what they understand — unknown resources are just scenery
		if n.type == "tree" and not known_tree:
			continue
		if n.type == "rock" and not known_rock:
			continue
		var d := Vector2(global_position.x, global_position.z).distance_to(Vector2(n.pos.x, n.pos.z))
		if d < best_d:
			best_d = d
			best = n
	return best

func _forager_behavior(delta: float) -> bool:
	if GameState.nodes.is_empty():
		return _wander_behavior(delta)
	var node := _nearest_node()
	if node.is_empty():
		return _wander_behavior(delta)
	var to: Vector3 = node.pos - global_position
	to.y = 0
	var dist := to.length()
	if dist > 1.1:
		var dir := to.normalized()
		var speed := 2.6
		global_position += dir * speed * delta
		look_at(global_position + dir, Vector3.UP)
		return true
	# harvest
	_harvest_tick += delta
	var rate: float = [1.0, 2.0, 3.0][clampi(GameState.tool_tier, 0, 2)]
	if randf() < delta * 1.2:
		play_attack()
	if _harvest_tick >= 2.3 / rate:
		_harvest_tick = 0.0
		if node.type == "tree":
			GameState.add_wood(1.0 + GameState.tool_tier)
			GameState.learn_concept("wood")
		else:
			GameState.add_gems(1.0 + GameState.tool_tier)
			GameState.learn_concept("gems")
	return false

func _sentry_behavior(delta: float) -> bool:
	# hold near the torii, scanning
	var anchor := Vector3(0, GameState.terrain_height(0, -1.2) + 0.5, -1.2)
	var to := anchor - global_position
	to.y = 0
	if to.length() > 2.8:
		var dir := to.normalized()
		global_position += dir * 2.2 * delta
		look_at(global_position + dir, Vector3.UP)
		return true
	return _wander_behavior(delta, 1.8)

func _tender_behavior(delta: float) -> bool:
	# care for the neediest peer nearby
	var neediest: RoninUnit = null
	var worst := 60.0
	for u in GameState.units:
		if u == self or not is_instance_valid(u):
			continue
		var need := minf(u.hunger, minf(u.cleanliness, u.mood))
		if need < worst:
			worst = need
			neediest = u
	if neediest == null:
		return _wander_behavior(delta)
	var to := neediest.global_position - global_position
	to.y = 0
	if to.length() > 1.3:
		var dir := to.normalized()
		global_position += dir * 2.0 * delta
		look_at(global_position + dir, Vector3.UP)
		return true
	neediest.feed(6.0 * delta, true)
	neediest.cleanse(9.0 * delta)
	return false

func _forge_warden_behavior(delta: float) -> bool:
	# haunt the forge location (or hover near center if no forge built yet)
	var anchor := Vector3.ZERO
	var found := false
	for i in range(GameState.structures.size()):
		if GameState.structures[i] == "forge":
			anchor = _forge_slot()
			found = true
			break
	if not found:
		return _wander_behavior(delta, 2.5)
	var to := anchor - global_position
	to.y = 0
	if to.length() > 1.6:
		var dir := to.normalized()
		global_position += dir * 2.1 * delta
		look_at(global_position + dir, Vector3.UP)
		return true
	return false

func _forge_slot() -> Vector3:
	var idx := GameState.structures.find("forge")
	if idx >= 0 and idx < GameState.structure_positions.size():
		return GameState.structure_positions[idx]
	return Vector3.ZERO

func _wander_behavior(delta: float, radius: float = 5.0) -> bool:
	_wander_timer -= delta
	if _wander_timer <= 0.0:
		_new_wander_target(radius)
	var to := _wander_target - global_position
	to.y = 0
	if to.length() > 0.4:
		var dir := to.normalized()
		var speed := 1.35
		if GameState.blight >= 50.0:
			speed *= 0.6
		global_position += dir * speed * delta
		look_at(global_position + dir, Vector3.UP)
		return true
	return false

func _new_wander_target(radius: float = 5.0) -> void:
	_wander_timer = randf_range(2.5, 6.0)
	var dir := Vector3(randf_range(-1, 1), 0, randf_range(-1, 1)).normalized()
	_wander_target = global_position + dir * randf_range(0.5, radius)
	_wander_target.y = _floor_y()

func _floor_y() -> float:
	return GameState.terrain_height(global_position.x, global_position.z)

# ============ sim tick ============

func _simulate_tick() -> void:
	var pop_pressure: float = maxf(0.0, float(GameState.units.size()) - 6.0) * 0.055
	hunger = maxf(0.0, hunger - ((2.1 if GameState.blight >= 50.0 else 1.6) + pop_pressure))
	cleanliness = maxf(0.0, cleanliness - 0.55)
	age_seconds += TICK_SECONDS

	var target := 58.0
	target += (hunger - 50.0) * 0.6
	target += (cleanliness - 50.0) * 0.35
	if GameState.sleeping:
		target += 12.0
	if GameState.blight >= 50.0:
		target -= 17.0
	mood = clampf(lerpf(mood, clampf(target, 0.0, 100.0), 0.08), 0.0, 100.0)

	if hunger <= 0.0:
		_starve_ticks += 1
		GameState.add_corruption(2.0)
		mood = maxf(0.0, mood - 4.0)
		if _starve_ticks >= STARVE_DEATH_TICKS:
			_die()
			return
	else:
		_starve_ticks = maxi(0, _starve_ticks - 2)

	# evolution requires gems for the rite
	var threshold: float = [90.0, 240.0][clampi(tier, 0, 1)] if tier < 2 else INF
	if age_seconds >= threshold:
		_try_evolve()

	# MITOSIS — they split themselves, but only when truly thriving
	if mood > 68.0 and hunger > 65.0 and cleanliness > 65.0 and age_seconds > 60.0 and GameState.can_mitosis():
		split_progress = minf(100.0, split_progress + 0.5 + valor * 0.15)
		if split_progress >= 100.0:
			split_progress = 0.0
			mitosis_ready.emit(self)
	else:
		split_progress = maxf(0.0, split_progress - 0.9)

func _try_evolve() -> void:
	if not GameState.pay_evolution(tier + 1):
		# rite denied for lack of parts — they wait, resentfully
		mood = maxf(0.0, mood - 3.0)
		return
	tier += 1
	age_seconds = 0.0
	_apply_tier_scale()
	play_celebrate()
	evolved.emit(self, tier)

func feed(amount: float = 34.0, _silent: bool = false) -> void:
	hunger = minf(100.0, hunger + amount)
	mood = minf(100.0, mood + 6.0)

func train() -> void:
	age_seconds += 22.0
	mood = clampf(mood - 4.0, 0.0, 100.0)
	play_attack()

func cleanse(amount: float = 30.0) -> void:
	cleanliness = minf(100.0, cleanliness + amount)
	if amount <= 30.0:
		mood = minf(100.0, mood + 4.0)

func amuse(amount: float = 12.0) -> void:
	mood = minf(100.0, mood + amount * 0.8)

func _die() -> void:
	dead = true
	var uname := display_name()
	var at := global_position
	died.emit(uname, at)
	GameState.register_death(uname)
	queue_free()

func is_content() -> bool:
	return mood >= 55.0 and hunger >= 35.0

func corruption_pressure() -> float:
	var p := 0.0
	if hunger < 25.0:
		p += 1.2
	if cleanliness < 25.0:
		p += 0.8
	if mood < 30.0:
		p += 1.0
	return p

func current_mood() -> String:
	if mood >= 78.0: return "Glorious"
	if mood >= 55.0: return "Content"
	if mood >= 30.0: return "Uneasy"
	return "Miserable"

func dash_to(point: Vector3) -> void:
	dash_override = point

func _update_anim(moving: bool) -> void:
	if _anim == null:
		return
	if _oneshot != "":
		return
	var want := "Walk" if moving else "Idle"
	if _anim.current_animation != want:
		_anim.play(want)

func _on_anim_finished(_name: String) -> void:
	_oneshot = ""
	if _anim:
		_anim.play("Idle")

func play_attack() -> void:
	if _anim and _oneshot == "":
		_oneshot = "Attack"
		_anim.play("Attack")

func play_celebrate() -> void:
	if _anim and _oneshot == "":
		_oneshot = "Celebrate"
		_anim.play("Celebrate")
