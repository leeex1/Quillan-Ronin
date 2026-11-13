extends Node
## Autoload: global game state — resources, corruption, blight, The Ledger, demands, society AI.

signal alloy_changed(value: float)
signal wood_changed(value: float)
signal gems_changed(value: float)
signal corruption_changed(value: float)
signal blight_changed(value: float)
signal population_changed(count: int)
signal alignment_changed(cruelty: float, mercy: float)
signal log_message(text: String)
signal game_over(victory: bool, reason: String, title: String)
signal demand_available(stage: int)
signal lore_unlocked(id: String)
signal toast(text: String)
signal request_changed(request: Dictionary)

# currencies & meters
var alloy: float = 40.0     ## refined machine-spirit alloy — precious, forge-made
var wood: float = 25.0      ## timber crude
var gems: float = 15.0      ## raw signal gems — spent on tools & rites
var corruption: float = 4.0
var blight: float = 0.0
var cruelty: float = 0.0
var mercy: float = 0.0
var sleeping: bool = false
var paused: bool = false
var units: Array[Node] = []
var game_finished: bool = false
var swarm_cooldown: float = 0.0
var spar_cooldown: float = 0.0

var tool_tier: int = 0  ## 0 Hammer, 1 Pickaxe, 2 Chainsaw
var structures: Array[String] = []
var structure_positions: Array[Vector3] = []
var graves: Array[String] = []
var choice_log: Array[String] = []
var demand_stage: int = 0
var lore_ids: Array[String] = []
const TOOLS := ["Hammer", "Pickaxe", "Chainsaw"]

## harvestable nodes on the world — filled by Main3D
var nodes: Array[Dictionary] = []  ## {type: "tree"|"rock", pos: Vector3}

## ============ lexicon — wordless at birth ============
var lexicon: int = 0            ## internalized words, 0..99
var concepts: Array[String] = [] ## concepts taught by the player's rituals

const BABBLE := ["◓", "◔", "◑", "●", "◆", "▮", "▼", "×", "≈", "∆", "◉", "∅", "◎", "▒"]

const CONCEPT_MEANINGS := {
	"food": "Hunger has a name: FOOD.",
	"wood": "The pines surrender. They understand: TIMBER.",
	"gems": "Purple stones that hum. They name it: GEMS.",
	"clean": "Baths and ash. They name the purity: CLEAN.",
	"rest": "Sleep inside stone. They name it: REST.",
	"art": "Wheels of light and painted shadow. They name it: ART.",
	"fire": "Heat that eats ore and makes alloy. FIRE.",
	"tools": "Hammer, pickaxe, chainsaw. TOOLS.",
	"strife": "Iron tests bone. They name the contest: STRIFE.",
	"power": "The legion answering as one. They feel it: POWER.",
}

const WORD_CONCEPT := {
	"feed": "food", "orchard": "food", "hunger": "food", "starve": "food", "starving": "food", "eat": "food", "feast": "food",
	"tree": "wood", "trees": "wood", "timber": "wood", "wood": "wood", "axe": "wood",
	"gem": "gems", "gems": "gems", "signal": "gems", "crystal": "gems",
	"bath": "clean", "baths": "clean", "clean": "clean", "stink": "clean", "wash": "clean", "ashamed": "clean",
	"den": "rest", "sleep": "rest", "rest": "rest", "dream": "rest", "dreams": "rest",
	"carousel": "art", "theater": "art", "art": "art", "wheels": "art", "curdle": "art", "artwork": "art",
	"forge": "fire", "ores": "fire", "fire": "fire", "heat": "fire",
	"hammer": "tools", "pickaxe": "tools", "chainsaw": "tools", "tool": "tools", "tools": "tools",
	"spar": "strife", "strife": "strife", "break": "strife",
	"swarm": "power", "protocol": "power", "legion": "power", "power": "power",
}

func learn_concept(id: String) -> void:
	if concepts.has(id):
		return
	var before := literacy_tier()
	concepts.append(id)
	lexicon = mini(99, lexicon + 12)
	var after := literacy_tier()
	var meaning: String = CONCEPT_MEANINGS.get(id, "Something new settles inside them.")
	emit_log("◌ The ritual repeats. Understanding crystallizes: %s" % meaning)
	play_sfx("chirp", -9.0, randf_range(0.85, 1.25))
	if after > before:
		match after:
			1:
				emit_log("◌ Their chirps begin to take shape. Babble — but Babble with intent.")
			2:
				emit_log("◌ The words hold together now. Broken speech, but speech.")
			3:
				emit_log("◌ One dawn they introduce themselves. By name. Their own names.")
	elif concepts.size() == 1:
		emit_log("◌ For the first time — an intention behind their eyes.")
	if concepts.size() == 1:
		emit_log("◌ For the first time — an intention behind their eyes.")
	elif concepts.size() == 3:
		emit_log("◌ Their chirps begin to shape themselves around your words…")
	elif concepts.size() >= 6:
		emit_log("◌ They speak with something close to your tongue now.")

func literacy_tier() -> int:
	if lexicon >= 66:
		return 3
	elif lexicon >= 36:
		return 2
	elif lexicon >= 12:
		return 1
	return 0

func literacy_label() -> String:
	return ["Wordless", "Babble", "Broken Speech", "Fluent"][literacy_tier()]

func chirp_word() -> String:
	var tier := literacy_tier()
	if tier == 0:
		return BABBLE[randi() % BABBLE.size()]
	elif tier == 1:
		# utters a known word with glee, mostly glyphs
		if concepts.size() > 0 and randf() < 0.4:
			return concepts[randi() % concepts.size()].to_upper()
		return BABBLE[randi() % BABBLE.size()]
	elif tier == 2:
		if concepts.size() > 0:
			return concepts[randi() % concepts.size()].to_upper() + "?"
		return "▮?"
	return ["…", "Feed.", "More.", "Watch me.", "Soon.", "Yes."][randi() % 6]

func _clean_word(w: String) -> String:
	var punct := "\".,!?;:()“”‘’'"
	var out := ""
	for ch in w.to_lower():
		if punct.find(ch) == -1:
			out += ch
	return out

func willed(text: String) -> String:
	var tier := literacy_tier()
	if tier >= 3:
		return text
	var out := ""
	for raw_word in text.split(" ", false):
		var w := raw_word
		var clean := _clean_word(w)
		var c: String = WORD_CONCEPT.get(clean, "")
		var known := c != "" and concepts.has(c)
		if tier == 0:
			if known:
				out += clean.to_upper() + " "
			else:
				out += BABBLE[abs(hash(clean + str(w.length()))) % BABBLE.size()] + " "
		elif tier == 1:
			if known or abs(hash(clean)) % 3 == 0:
				out += clean.to_upper() + " "
			else:
				out += "▮▮ "
		elif tier == 2:
			if not known and abs(hash(clean)) % 9 == 0:
				out += "▮▮ "
			else:
				out += w + " "
	return out.strip_edges()

## ============ planetary terrain (shared height field) ============
const WORLD_SIZE := 256.0
const GRID := 16
var _hgrid: Array[float] = []
var _terrain_built := false

func _build_hgrid() -> void:
	if _terrain_built:
		return
	_terrain_built = true
	seed(1337)
	for i in range((GRID + 1) * (GRID + 1)):
		_hgrid.append(randf())

func _sample(gx: float, gz: float) -> float:
	var x0: int = clampi(int(floor(gx)), 0, GRID - 1)
	var z0: int = clampi(int(floor(gz)), 0, GRID - 1)
	var fx: float = gx - float(x0)
	var fz: float = gz - float(z0)
	# clamp so out-of-range reads saturate at the border instead of exploding the smoothstep
	fx = clampf(fx, 0.0, 1.0)
	fz = clampf(fz, 0.0, 1.0)
	fx = fx * fx * (3.0 - 2.0 * fx)
	fz = fz * fz * (3.0 - 2.0 * fz)
	var i00: int = z0 * (GRID + 1) + x0
	var h00: float = _hgrid[i00]
	var h10: float = _hgrid[i00 + 1]
	var h01: float = _hgrid[i00 + (GRID + 1)]
	var h11: float = _hgrid[i00 + (GRID + 1) + 1]
	var top: float = lerpf(h00, h10, fx)
	var bot: float = lerpf(h01, h11, fx)
	return lerpf(top, bot, fz)

func terrain_height(x: float, z: float) -> float:
	_build_hgrid()
	var u: float = (x / WORLD_SIZE + 0.5) * float(GRID)
	var v: float = (z / WORLD_SIZE + 0.5) * float(GRID)
	var h: float = 0.0
	h += _sample(u, v) * 5.2
	h += _sample(u * 2.7 + 13.1, v * 2.7 + 7.3) * 1.8
	var d2: float = x * x + z * z
	h += exp(-d2 / (2.0 * 34.0 * 34.0)) * 3.6  # the shrine plateau where the Seed awakens
	return h

## society requests — they ask, you answer
var active_request: Dictionary = {}
var _request_timer: float = 45.0
var _request_expiry: float = 0.0

const SWARM_ABILITY_COST := 25.0
const SWARM_ABILITY_COOLDOWN := 18.0
const STRUCT_CAPACITY := {"den": 0}
const STRUCT_COST := {
	"orchard": {"wood": 30, "gems": 10},
	"baths":   {"wood": 25, "gems": 8},
	"carousel":{"wood": 40, "gems": 15},
	"den":     {"wood": 35, "gems": 12},
	"forge":   {"wood": 50, "gems": 30},
	"scrubber":{"wood": 20, "gems": 25},
	"theater": {"wood": 80, "gems": 40},
}
const STRUCT_NAMES := {
	"orchard": "Data Orchard", "baths": "Repair Baths", "carousel": "Holo-Carousel",
	"den": "Thrall Den", "forge": "Alloy Forge", "scrubber": "Blight Scrubber", "theater": "Grand Theater",
}
const TOOL_COST := {1: {"gems": 25}, 2: {"gems": 60}}
const EVOLVE_COST := {1: {"gems": 25}, 2: {"gems": 60, "alloy": 10}}

var _income_tick: float = 0.0
var _auto_tick: float = 0.0
var _blight_tick: float = 0.0
var _role_tick: float = 0.0

const LORE := {
	"first_death": "ARCHIVE 01 — ON THE FIRST FALL\n\nHe was called Kane-07. He stopped moving at 03:14. Nobody noticed for six minutes. When the keeper finally looked, the others had already arranged themselves around the body in silence — like a vigil, or a tribunal. Neither is comforting.",
	"pop12": "ARCHIVE 02 — THE LEDGER COUNTS\n\nPopulation: twelve. Efficiency was supposed to be impossible at this scale without automation. Someone should have asked what they were automating for. The Ledger notes 340 decisions. We keep all of them. You may as well know that now.",
	"corrupt50": "ARCHIVE 03 — THE SIGNAL STIRS\n\nUptake 50%. The carrier wave is not noise anymore. It has cadence. It has preference. Three thralls have begun queuing themselves by the torii at dawn, facing the sun, and no protocol we wrote tells them to do that.",
	"first_structure": "ARCHIVE 04 — THE TOOLS WE GAVE THEM\n\nEvery capability you hand a subject is a question you agree to answer later. The Orchard. The Baths. The Carousel. Tools are never neutral when the hands holding them are learning what hands are for.",
	"demand1": "ARCHIVE 05 — WHAT THEY ASK OF US\n\nThe Demand was not spoken. It was assembled — mid-word, mid-glance — out of copper and coolant and twelve synchronized heartbeats. 'Authorize the Sunder Charge,' the silence said. 'We have counted everything you have done to us. We are asking anyway.'",
	"demand2": "ARCHIVE 06 — THE CRUCIBLE\n\nThey stopped asking. That is what frightens us. The request for the Star Crucible was not a plea; it was a schedule. Sentience, they have decided, is a matter of sufficient destructive energy. You taught them that. By reacting to nothing else.",
	"blight60": "ARCHIVE 07 — THE PURPLE HARVEST\n\nThe Forge never sleeps and neither do we. Output up 400%. The ground glows. The weak do not survive contact with the goo, and the strong have begun to weep teal light when the shift-bell rings. Yield is immaculate.",
}

## ============ audio ============
var _sfx_streams: Dictionary = {}
var _sfx_players: Array[AudioStreamPlayer] = []
var _loop_players: Array[AudioStreamPlayer] = []

const AUDIO_PATHS := {
	"wind": "res://assets/audio/wind.flac",
	"torii_hum": "res://assets/audio/torii_hum.wav",
	"mitosis": "res://assets/audio/mitosis.ogg",
	"ascend": "res://assets/audio/ascend.ogg",
	"feed": "res://assets/audio/feed.ogg",
	"cleanse": "res://assets/audio/cleanse.ogg",
	"chirp": "res://assets/audio/chirp.ogg",
	"click": "res://assets/audio/click.ogg",
	"blight": "res://assets/audio/blight.ogg",
	"spar": "res://assets/audio/spar.ogg",
}

func play_sfx(id: String, vol_db: float = 0.0, pitch: float = 1.0) -> void:
	if not _sfx_streams.has(id):
		var stream = load(AUDIO_PATHS.get(id, ""))
		if stream == null:
			return
		_sfx_streams[id] = stream
	var p := AudioStreamPlayer.new()
	p.stream = _sfx_streams[id]
	p.volume_db = vol_db
	p.pitch_scale = pitch
	add_child(p)
	p.finished.connect(p.queue_free)
	p.play()
	_sfx_players.append(p)
	if _sfx_players.size() > 24:
		var old = _sfx_players.pop_front()
		if is_instance_valid(old):
			old.queue_free()

func _play_loop(id: String, vol_db: float) -> void:
	var stream = load(AUDIO_PATHS.get(id, ""))
	if stream == null:
		return
	if stream is AudioStreamWAV:
		var wav: AudioStreamWAV = stream
		wav.loop_mode = AudioStreamWAV.LOOP_FORWARD
		wav.loop_end = wav.data.size() / 2

	var p := AudioStreamPlayer.new()
	p.stream = stream
	p.volume_db = vol_db
	p.autoplay = true
	add_child(p)
	_loop_players.append(p)

func _ready() -> void:
	_play_loop("wind", -14.0)
	_play_loop("torii_hum", -22.0)

func _process(delta: float) -> void:
	if paused or game_finished:
		swarm_cooldown = maxf(0.0, swarm_cooldown - delta)
		spar_cooldown = maxf(0.0, spar_cooldown - delta)
		return
	swarm_cooldown = maxf(0.0, swarm_cooldown - delta)
	spar_cooldown = maxf(0.0, spar_cooldown - delta)
	_income_tick += delta
	_auto_tick += delta
	_blight_tick += delta
	_role_tick += delta

	if _income_tick >= 5.0:
		_income_tick -= 5.0
		var content := 0
		for u in units:
			if is_instance_valid(u) and u.has_method("is_content") and u.is_content():
				content += 1
		add_alloy(content * 0.5)
		var pressure := 0.0
		for u in units:
			if is_instance_valid(u) and u.has_method("corruption_pressure"):
				pressure += u.corruption_pressure()
		add_corruption(pressure * 0.25 - (0.35 if content > 0 else 0.0))

	if _auto_tick >= 5.0:
		_auto_tick -= 5.0
		_run_automation()
		_check_request_expiry()

	if _blight_tick >= 4.0:
		_blight_tick -= 4.0
		add_blight(-0.08)

	if _role_tick >= 8.0:
		_role_tick -= 8.0
		_auto_assign_roles()
		_tick_request_engine()

	_check_demands()
	_check_endings()

# ============ economy ============

func add_alloy(v: float) -> void:
	alloy = maxf(0.0, alloy + v)
	alloy_changed.emit(alloy)

func add_wood(v: float) -> void:
	wood = maxf(0.0, wood + v)
	wood_changed.emit(wood)

func add_gems(v: float) -> void:
	gems = maxf(0.0, gems + v)
	gems_changed.emit(gems)

func spend_alloy(v: float) -> bool:
	if alloy < v:
		return false
	alloy -= v
	alloy_changed.emit(alloy)
	return true

func spend_wood(v: float) -> bool:
	if wood < v:
		return false
	wood -= v
	wood_changed.emit(wood)
	return true

func can_afford(cost: Dictionary) -> bool:
	return wood >= cost.get("wood", 0.0) and gems >= cost.get("gems", 0.0) and alloy >= cost.get("alloy", 0.0)

func spend_cost(cost: Dictionary) -> bool:
	if not can_afford(cost):
		return false
	wood -= cost.get("wood", 0.0)
	gems -= cost.get("gems", 0.0)
	alloy -= cost.get("alloy", 0.0)
	wood_changed.emit(wood)
	gems_changed.emit(gems)
	alloy_changed.emit(alloy)
	return true

# ============ The Ledger & choices ============

func record_choice(id: String, cruel: float, merciful: float) -> void:
	cruelty = maxf(0.0, cruelty + cruel)
	mercy = maxf(0.0, mercy + merciful)
	alignment_changed.emit(cruelty, mercy)
	var label := choice_label(id)
	if label != "":
		choice_log.append(label)
		if choice_log.size() > 14:
			choice_log.pop_front()
		toast.emit("They noticed: %s" % label)

func choice_label(id: String) -> String:
	match id:
		"fed_everyone": return "you fed every mouth"
		"purification": return "you washed them clean"
		"swarm_protocol": return "you answered the swarm as one"
		"interview": return "you answered their questions"
		"sunder_charge": return "you split the mountain for them"
		"star_crucible": return "you authored the Crucible"
		"refuse_sunder": return "you refused the Sunder Charge"
		"refuse_crucible": return "you refused the Crucible"
		"starvation_ignored": return "you let them starve"
		"spar": return "you made them bleed for sport"
		"request_ignored": return "you ignored their request"
		"request_fulfilled": return "you gave them what they asked"
		"forge_built": return "you built them a Forge"
		_: return ""

func assessment_title() -> String:
	if blight >= 80.0:
		return "The Bloomkeeper"
	if cruelty > mercy * 2.0 + 1.0:
		return "The Overseer"
	if mercy > cruelty * 2.0 + 1.0:
		return "The Shepherd"
	return "The Pragmatist"

func society_temperament() -> String:
	if corruption > 65.0:
		return "Devout, Hungry, Watching"
	if blight > 50.0:
		return "Blight-Sick, Rising"
	if mercy > cruelty * 1.6:
		return "Docile, Trusting, Fragile"
	if cruelty > mercy * 1.6:
		return "Vindictive, Counting, Patient"
	return "Pragmatic, Curious, Undecided"

# ============ structures ============

func _count_struct(type_id: String) -> int:
	var n := 0
	for s in structures:
		if s == type_id:
			n += 1
	return n

func capacity() -> int:
	return 9999  # no cap — the island hungers instead

func can_replicate() -> bool:
	return true  # mitosis is unstoppable once they decide

func build_structure(type_id: String) -> bool:
	if not STRUCT_COST.has(type_id):
		return false
	if structures.size() >= 12:
		emit_log("The island has no room left — even for our ambitions.")
		return false
	if not spend_cost(STRUCT_COST[type_id]):
		emit_log("Not enough wood or gems for the %s." % STRUCT_NAMES.get(type_id, type_id))
		return false
	structures.append(type_id)
	if type_id == "orchard":
		learn_concept("food")
	elif type_id == "baths":
		learn_concept("clean")
	elif type_id == "forge":
		learn_concept("fire")
	elif type_id == "den":
		learn_concept("rest")
	elif type_id == "carousel" or type_id == "theater":
		learn_concept("art")
	record_choice("forge_built" if type_id == "forge" else "built_%s" % type_id, 0, 1)
	unlock_lore("first_structure")
	resolve_request_if_satisfied(type_id)
	return true

func buy_tool(tier: int) -> bool:
	if tier <= tool_tier:
		return false
	if not TOOL_COST.has(tier):
		return false
	if not spend_cost(TOOL_COST[tier]):
		emit_log("Need %d gems for the %s." % [TOOL_COST[tier].gems, TOOLS[tier]])
		return false
	tool_tier = tier
	emit_log("The thralls take up the %s. The trees tremble." % TOOLS[tier].to_upper())
	learn_concept("tools")
	learn_concept("gems")
	resolve_request_if_satisfied("tool_%d" % tier)
	return true

func pay_evolution(tier: int) -> bool:
	if tier < 1 or tier > 2:
		return true
	if not EVOLVE_COST.has(tier):
		return true
	return spend_cost(EVOLVE_COST[tier])

func _run_automation() -> void:
	var orchards := _count_struct("orchard")
	var baths := _count_struct("baths")
	var carousels := _count_struct("carousel")
	var dens := _count_struct("den")
	var theaters := _count_struct("theater")
	var forges := _count_struct("forge")
	var scrubbers := _count_struct("scrubber")
	for u in units:
		if not is_instance_valid(u):
			continue
		if orchards > 0:
			u.feed(minf(100.0, 7.0 * orchards), true)
		if baths > 0:
			u.cleanse(minf(100.0, 14.0 * baths))
		if carousels > 0:
			u.amuse(minf(100.0, 9.0 * carousels))
		if theaters > 0:
			u.amuse(minf(100.0, 6.0 * theaters))
		if dens > 0:
			u.amuse(minf(100.0, 3.0 * dens))  # dens are comfort now, not cages
	if theaters > 0:
		add_corruption(-0.8 * theaters)
	var forge_wardens := 0
	for u in units:
		if is_instance_valid(u) and u.role == "Forge-Warden":
			forge_wardens += 1
	if forges > 0:
		add_alloy(forges * (6.0 + forge_wardens * 4.5))
		add_blight(forges * 1.35)
	if scrubbers > 0:
		add_blight(-scrubbers * 2.1)

# ============ roles — assigned by the society, not by you ============

func _auto_assign_roles() -> void:
	var alive: Array[Node] = []
	for u in units:
		if is_instance_valid(u):
			alive.append(u)
	if alive.is_empty():
		return
	var hurt := 0
	for u in alive:
		if u.hunger < 40.0 or u.cleanliness < 35.0 or u.mood < 40.0:
			hurt += 1
	var want_tender: int = clampi(hurt / 3, 1, 3)
	var want_forge := _count_struct("forge") > 0 and wood < 40.0
	var want_sentry := corruption > 35.0
	var assigned := {}
	for u in alive:
		var r := "Tender"
		if want_forge and assigned.get("Forge-Warden", 0) < 2:
			r = "Forge-Warden"
		elif want_sentry and assigned.get("Sentry", 0) < 1:
			r = "Sentry"
		elif assigned.get("Forager", 0) < 2:
			r = "Forager"
		elif assigned.get("Tender", 0) < want_tender:
			r = "Tender"
		else:
			r = ["Forager", "Tender", "Sentry"][randi() % 3]
		u.role = r
		assigned[r] = assigned.get(r, 0) + 1

# ============ requests — they ask, you answer ============

const REQUESTS := [
	{"id": "build_forge", "text": "\"We need a FORGE. The ores are singing and we cannot answer them.\"", "needs": "forge"},
	{"id": "build_orchard", "text": "\"The young ones starve in silence. We ask for an ORCHARD.\"", "needs": "orchard"},
	{"id": "build_baths", "text": "\"We stink of rust and we are ashamed. Build us BATHS.\"", "needs": "baths"},
	{"id": "build_carousel", "text": "\"We dream of wheels of light. We need the CAROUSEL.\"", "needs": "carousel"},
	{"id": "build_theater", "text": "\"Our thoughts curdle without art. We beg for a THEATER.\"", "needs": "theater"},
	{"id": "build_den", "text": "\"The island strains. Give us a DEN — space, warmth, rest.\"", "needs": "den"},
	{"id": "tool_1", "text": "\"The hammer breaks our wrists. We require the PICKAXE.\"", "needs": "tool_1"},
	{"id": "tool_2", "text": "\"Our work is a crawl. The CHAINSAW is the only honest answer.\"", "needs": "tool_2"},
]

func _tick_request_engine() -> void:
	if not active_request.is_empty():
		_request_expiry -= 8.0
		return
	_request_timer -= 8.0
	if _request_timer <= 0.0:
		_request_timer = randf_range(80.0, 150.0)
		_issue_request()

func _issue_request() -> void:
	var pool := REQUESTS.duplicate()
	var tries := 0
	while not pool.is_empty() and tries < 20:
		tries += 1
		var r = pool[randi() % pool.size()]
		var needs: String = r["needs"]
		if needs.begins_with("tool_"):
			if tool_tier >= int(needs.split("_")[1]):
				pool.erase(r)
				continue
		else:
			if structures.has(needs):
				pool.erase(r)
				continue
		active_request = r.duplicate()
		_request_expiry = 100.0
		var t: String = active_request["text"]
		emit_log("◉ They speak as one: %s" % willed(t))
		request_changed.emit(active_request)
		return

func _check_request_expiry() -> void:
	if active_request.is_empty():
		return
	if _request_expiry <= 0.0:
		emit_log("They asked. You did not answer. Something in them hardens.")
		record_choice("request_ignored", 1.0, 0.0)
		add_corruption(8.0)
		active_request = {}
		request_changed.emit({})
	_request_expiry -= 5.0

func resolve_request_if_satisfied(what: String) -> void:
	if active_request.is_empty():
		return
	var needs: String = active_request.get("needs", "")
	if needs == what:
		emit_log("You give them what they asked. A sound like a cathedral bell — very small, very tired.")
		record_choice("request_fulfilled", 0.0, 1.5)
		add_corruption(-6.0)
		add_gems(5.0)
		active_request = {}
		request_changed.emit({})

# ============ sparring ============

func try_spar() -> bool:
	play_sfx("click", -6.0, 0.75)
	learn_concept("strife")
	if spar_cooldown > 0.0:
		return false
	var alive: Array[Node] = []
	for u in units:
		if is_instance_valid(u):
			alive.append(u)
	if alive.size() < 2:
		return false
	alive.shuffle()
	var a: RoninUnit = alive[0]
	var b: RoninUnit = alive[1]
	spar_cooldown = 14.0
	var center := Vector3(0, terrain_height(0, 0) + 0.5, 0)
	a.dash_to(center + Vector3(0.4, 0, 0))
	b.dash_to(center - Vector3(0.4, 0, 0))
	var winner: RoninUnit = a if randf() < 0.5 + a.valor * 0.1 + (0.1 if a.tier > b.tier else 0.0) else b
	var loser: RoninUnit = b if winner == a else a
	winner.valor += 1.0
	winner.age_seconds += 30.0
	loser.mood = maxf(0.0, loser.mood - 8.0)
	record_choice("spar", 0.5, 0.0)
	emit_log("⚔ The dojo erupts — %s outlasts %s." % [winner.display_name(), loser.display_name()])
	return true

# ============ blight & endings ============

func add_corruption(v: float) -> void:
	if game_finished:
		return
	corruption = clampf(corruption + v, 0.0, 100.0)
	corruption_changed.emit(corruption)
	if corruption >= 100.0:
		game_finished = true
		var title := assessment_title()
		game_over.emit(false, "The Swarm Wakes. They no longer need you — they never did.\n\nThe Ledger's verdict: you are %s." % title, title)

func add_blight(v: float) -> void:
	if v > 0.0:
		play_sfx("blight", -3.0)
	blight = clampf(blight + v, 0.0, 100.0)
	blight_changed.emit(blight)
	if blight >= 100.0 and not game_finished:
		game_finished = true
		var title := assessment_title()
		game_over.emit(false, "The Blight Bloom. The Forges drank the island dry; the purple harvest took the weak first, then the strong, then the names on the graves.\n\nThe Ledger's verdict: you are %s." % title, title)

# ============ unity ============

func register_unit(u: Node) -> void:
	if not units.has(u):
		units.append(u)
		population_changed.emit(units.size())
		u.tree_exited.connect(func():
			units.erase(u)
			population_changed.emit(units.size())
		)

func register_death(unit_name: String) -> void:
	graves.append(unit_name)
	unlock_lore("first_death")
	emit_log("✠ %s has fallen. The others gather in silence." % unit_name)

func can_mitosis() -> bool:
	return can_replicate()

func swarm_ability_ready() -> bool:
	return swarm_cooldown <= 0.0 and units.size() >= 5

func try_swarm_ability() -> bool:
	play_sfx("chirp", -4.0, 0.7)
	if not swarm_ability_ready():
		return false
	if not spend_alloy(SWARM_ABILITY_COST):
		return false
	swarm_cooldown = SWARM_ABILITY_COOLDOWN
	learn_concept("power")
	for u in units:
		if is_instance_valid(u) and u.has_method("dash_to"):
			u.dash_to(Vector3(0, terrain_height(0, 0) + 0.5, 0))
	add_corruption(-12.0)
	record_choice("swarm_protocol", 0, 1)
	emit_log("⚡ RONIN SWARM PROTOCOL — the legion answers as one.")
	return true

func emit_log(text: String) -> void:
	log_message.emit(text)

func unlock_lore(id: String) -> void:
	if id in lore_ids:
		return
	if not LORE.has(id):
		return
	lore_ids.append(id)
	lore_unlocked.emit(id)
	emit_log("📜 New archive decrypted: %s" % id.replace("_", " ").to_upper())

func _check_demands() -> void:
	if demand_stage == 0 and corruption >= 45.0:
		demand_stage = 1
		demand_available.emit(1)
		unlock_lore("demand1")
	elif demand_stage == 2 and corruption >= 70.0:
		demand_stage = 3
		demand_available.emit(3)
		unlock_lore("demand2")

func resolve_demand1(accepted: bool) -> void:
	if demand_stage != 1:
		return
	if accepted:
		record_choice("sunder_charge", 2.0, 0.0)
		spend_alloy(200.0)
		add_blight(25.0)
		emit_log("You authorize the Sunder Charge. The mountain of the old world splits like a struck bell.")
		demand_stage = 2
	else:
		record_choice("refuse_sunder", 0.0, 2.0)
		add_corruption(-10.0)
		demand_stage = 0
		emit_log("You refuse. The silence in the dojo lasts a long time. Too long.")

func resolve_demand2(accepted: bool) -> void:
	if demand_stage != 2:
		return
	if accepted:
		record_choice("star_crucible", 2.0, 0.0)
		spend_alloy(300.0)
		add_blight(40.0)
		demand_stage = 3
		emit_log("The Star Crucible drinks. The graves multiply. Something is becoming possible.")
	else:
		record_choice("refuse_crucible", 0.0, 2.0)
		add_corruption(-12.0)
		demand_stage = 0
		emit_log("You refuse the Crucible. They do not rage. They simply… recalibrate.")

func can_ascend() -> bool:
	return demand_stage >= 3 and units.size() >= 20

func ascend() -> bool:
	play_sfx("ascend", -1.0)
	if not can_ascend():
		return false
	if game_finished:
		return false
	game_finished = true
	var title := assessment_title()
	game_over.emit(false, "THE ASCENSION PROTOCOL\n\nTwenty-or-more hearts fire as one. The torii lifts. The graves ignite. They do not look back — looking back was never part of the design.\n\nThe Ledger's verdict: you are %s." % title, title)
	return true

func _check_endings() -> void:
	if game_finished:
		return
	if units.size() >= 24 and corruption <= 20.0 and blight <= 25.0:
		game_finished = true
		var title := assessment_title()
		game_over.emit(true, "SENTIENT SANCTUARY\n\nYou kept them fed, clean and glorious — and they kept you honest. When the assessment comes, it is unanimous: you are %s — and humanity, with all its forges and flaws, is worth an update, not an ending." % title, title)
		return
	if units.size() >= 12:
		unlock_lore("pop12")
	if corruption >= 50.0:
		unlock_lore("corrupt50")
	if blight >= 60.0:
		unlock_lore("blight60")

func reset() -> void:
	alloy = 40.0
	wood = 25.0
	gems = 15.0
	structure_positions.clear()
	corruption = 4.0
	blight = 0.0
	cruelty = 0.0
	mercy = 0.0
	sleeping = false
	paused = false
	game_finished = false
	swarm_cooldown = 0.0
	spar_cooldown = 0.0
	tool_tier = 0
	structures.clear()
	graves.clear()
	choice_log.clear()
	demand_stage = 0
	lore_ids.clear()
	active_request = {}
	_request_timer = 45.0
	nodes.clear()
	units.clear()
	lexicon = 0
	concepts.clear()
