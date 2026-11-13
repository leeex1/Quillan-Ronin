extends CanvasLayer
## HUD v3 — Wood/Gems/Alloy economy, workshop, tools, roles, society temperament, requests, toasts.

var _alloy_label: Label
var _wood_label: Label
var _gems_label: Label
var _pop_label: Label
var _cap_label: Label
var _lex_label: Label
var _temper_label: Label
var _corruption_bar: ProgressBar
var _blight_bar: ProgressBar
var _ledger_label: Label
var _portrait: TextureRect
var _log: RichTextLabel
var _console: Panel
var _console_input: LineEdit
var _sleep_button: Button
var _swarm_button: Button
var _spar_button: Button
var _game_over_panel: Panel
var _game_over_title: Label
var _game_over_sub: Label
var _unit_list: VBoxContainer
var _refresh_timer: float = 0.0
var _swarm_timer: float = 0.0
var _spar_timer: float = 0.0
var _build_panel: Panel
var _archives_panel: Panel
var _interview_panel: Panel
var _interview_q: Label
var _interview_btns: Array[Button] = []
var _demand_panel: Panel
var _demand_title: Label
var _demand_body: Label
var _demand_accept: Button
var _demand_refuse: Button
var _ascend_button: Button
var _current_demand_stage: int = 0
var _pending_question: Dictionary = {}
var _archives_list: VBoxContainer
var _request_banner: Panel
var _request_label: Label
var _toast: Label
var _toast_tween: Tween

const PORTRAIT_PATH := "res://assets/portrait.png"
const GOLD := Color(0.93, 0.75, 0.35)
const CRIMSON := Color(0.65, 0.08, 0.07)
const PANEL := Color(0.055, 0.035, 0.07, 0.92)

func _ready() -> void:
	layer = 10
	_build_ui()
	GameState.alloy_changed.connect(func(v): _alloy_label.text = "◈ Data-Alloy  %d" % int(v))
	GameState.wood_changed.connect(func(v): _wood_label.text = "🪵 Wood  %d" % int(v))
	GameState.gems_changed.connect(func(v): _gems_label.text = "◆ Signal Gems  %d" % int(v))
	GameState.corruption_changed.connect(func(v): _corruption_bar.value = v)
	GameState.blight_changed.connect(func(v): _blight_bar.value = v)
	GameState.population_changed.connect(func(_c): _update_pop())
	GameState.alignment_changed.connect(func(_c, _m): _update_ledger())
	GameState.lore_unlocked.connect(func(_id): _rebuild_archives())
	GameState.toast.connect(_show_toast)
	GameState.request_changed.connect(_on_request_changed)
	_alloy_label.text = "◈ Data-Alloy  %d" % int(GameState.alloy)
	_wood_label.text = "🪵 Wood  %d" % int(GameState.wood)
	_gems_label.text = "◆ Signal Gems  %d" % int(GameState.gems)
	_update_pop()
	_corruption_bar.value = GameState.corruption
	_blight_bar.value = GameState.blight
	_update_ledger()
	_rebuild_archives()

func _build_ui() -> void:
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	add_child(root)

	# ============ LEFT PANEL ============
	var left := Panel.new()
	left.position = Vector2(18, 18)
	left.size = Vector2(312, 680)
	left.add_theme_stylebox_override("panel", _panel_style())
	root.add_child(left)
	var vbox := VBoxContainer.new()
	vbox.position = Vector2(14, 12)
	vbox.size = Vector2(284, 656)
	vbox.add_theme_constant_override("separation", 6)
	left.add_child(vbox)

	var title := Label.new()
	title.text = "RONIN THRONGLETS"
	title.add_theme_font_size_override("font_size", 25)
	title.add_theme_color_override("font_color", GOLD)
	vbox.add_child(title)
	var sub := Label.new()
	sub.text = "SIGNAL//THRALL MANAGER"
	sub.add_theme_font_size_override("font_size", 10)
	sub.add_theme_color_override("font_color", Color(0.75, 0.65, 0.55))
	vbox.add_child(sub)

	_portrait = TextureRect.new()
	_portrait.texture = load(PORTRAIT_PATH)
	_portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_portrait.custom_minimum_size = Vector2(244, 148)
	vbox.add_child(_portrait)

	_alloy_label = _make_stat(vbox, "◈ Data-Alloy  0", Color(0.55, 0.9, 0.85))
	_wood_label = _make_stat(vbox, "🪵 Wood  0", Color(0.85, 0.7, 0.45))
	_gems_label = _make_stat(vbox, "◆ Signal Gems  0", Color(0.75, 0.5, 1.0))
	_pop_label = _make_stat(vbox, "⛊ Thralls  0", Color(0.95, 0.8, 0.5))
	_cap_label = _make_stat(vbox, "⌂ Capacity  0/0", Color(0.75, 0.72, 0.6))
	_lex_label = _make_stat(vbox, 'Words  0 - Wordless', Color(0.78, 0.72, 1.0))
	var corr_label := Label.new()
	corr_label.text = "UPTAKE — The Signal Cores"
	corr_label.add_theme_font_size_override("font_size", 10)
	corr_label.add_theme_color_override("font_color", Color(1.0, 0.42, 0.35))
	vbox.add_child(corr_label)
	_corruption_bar = _make_bar(Color(0.9, 0.12, 0.08), Color(0.16, 0.05, 0.06))
	vbox.add_child(_corruption_bar)

	var blight_label := Label.new()
	blight_label.text = "BLIGHT — The Purple Harvest"
	blight_label.add_theme_font_size_override("font_size", 10)
	blight_label.add_theme_color_override("font_color", Color(0.75, 0.45, 1.0))
	vbox.add_child(blight_label)
	_blight_bar = _make_bar(Color(0.62, 0.2, 0.9), Color(0.13, 0.06, 0.17))
	vbox.add_child(_blight_bar)

	_ledger_label = Label.new()
	_ledger_label.text = "THE LEDGER — Mercy 0 · Cruelty 0"
	_ledger_label.add_theme_font_size_override("font_size", 10)
	_ledger_label.add_theme_color_override("font_color", Color(0.85, 0.8, 0.6))
	vbox.add_child(_ledger_label)

	_temper_label = Label.new()
	_temper_label.text = "SOCIETY — Docile, Trusting, Fragile"
	_temper_label.add_theme_font_size_override("font_size", 10)
	_temper_label.add_theme_color_override("font_color", Color(0.65, 0.85, 1.0))
	vbox.add_child(_temper_label)

	_add_action_button(vbox, "◉  Feed the Horde (1🪵/ea)", _on_feed)
	_add_action_button(vbox, "⚔  Sparring Dojo (train)", _on_train)
	_add_action_button(vbox, "✚  Purification Rite (15◈)", _on_cleanse)
	_spar_button = _add_action_button(vbox, "🔥  SPAR AT THE DOJO", _on_spar)
	_sleep_button = _add_action_button(vbox, "☾  Sleep Protocol", _on_sleep)
	_swarm_button = _add_action_button(vbox, "⚡  SWARM PROTOCOL [Q]  (25◈)", _on_swarm, true)
	_ascend_button = _add_action_button(vbox, "☄  THE ASCENSION PROTOCOL", _on_ascend, true)
	_ascend_button.visible = false

	# ============ TOP-RIGHT: build + archives ============
	var top_right := HBoxContainer.new()
	top_right.position = Vector2(1920 - 560, 14)
	top_right.add_theme_constant_override("separation", 8)
	root.add_child(top_right)
	var build_btn := Button.new()
	build_btn.text = "⚒  WORKSHOP"
	build_btn.custom_minimum_size = Vector2(150, 40)
	build_btn.add_theme_font_size_override("font_size", 15)
	build_btn.pressed.connect(func(): _build_panel.visible = not _build_panel.visible)
	_style_button(build_btn, Color(0.16, 0.09, 0.2))
	top_right.add_child(build_btn)
	var arch_btn := Button.new()
	arch_btn.text = "📜  ARCHIVES"
	arch_btn.custom_minimum_size = Vector2(160, 40)
	arch_btn.add_theme_font_size_override("font_size", 15)
	arch_btn.pressed.connect(func(): _archives_panel.visible = not _archives_panel.visible)
	_style_button(arch_btn, Color(0.1, 0.12, 0.18))
	top_right.add_child(arch_btn)

	# ============ WORKSHOP PANEL ============
	_build_panel = Panel.new()
	_build_panel.position = Vector2(1920 - 580, 62)
	_build_panel.size = Vector2(380, 640)
	_build_panel.visible = false
	_build_panel.add_theme_stylebox_override("panel", _panel_style())
	root.add_child(_build_panel)
	var bv := VBoxContainer.new()
	bv.position = Vector2(12, 10)
	bv.size = Vector2(356, 620)
	bv.add_theme_constant_override("separation", 5)
	_build_panel.add_child(bv)
	var bh := Label.new()
	bh.text = "⚒  THE WORKSHOP — what they need built"
	bh.add_theme_font_size_override("font_size", 14)
	bh.add_theme_color_override("font_color", GOLD)
	bv.add_child(bh)
	var tool_h := Label.new()
	tool_h.text = "— TOOLS —"
	tool_h.add_theme_font_size_override("font_size", 11)
	tool_h.add_theme_color_override("font_color", Color(0.8, 0.6, 0.3))
	bv.add_child(tool_h)
	_add_build_entry(bv, "tool_1", "PICKAXE — 2× harvest (25◆)")
	_add_build_entry(bv, "tool_2", "CHAINSAW — 3× harvest (60◆)")
	var struct_h := Label.new()
	struct_h.text = "— STRUCTURES —"
	struct_h.add_theme_font_size_override("font_size", 11)
	struct_h.add_theme_color_override("font_color", Color(0.8, 0.6, 0.3))
	bv.add_child(struct_h)
	_add_build_entry(bv, "orchard", "Data Orchard — auto-feed (30🪵 10◆)")
	_add_build_entry(bv, "baths", "Repair Baths — auto-clean (25🪵 8◆)")
	_add_build_entry(bv, "carousel", "Holo-Carousel — auto-amuse (40🪵 15◆)")
	_add_build_entry(bv, "theater", "Grand Theater — culture & calm (80🪵 40◆)")
	_add_build_entry(bv, "den", "Thrall Den — comfort aura for all (35🪵 12◆)")
	_add_build_entry(bv, "forge", "Alloy Forge — 6◈/5s + Blight (50🪵 30◆)")
	_add_build_entry(bv, "scrubber", "Blight Scrubber — −2.1 blight/5s (20🪵 25◆)")
	var note := Label.new()
	note.text = "Every build is remembered. So is every refusal."
	note.add_theme_font_size_override("font_size", 10)
	note.add_theme_color_override("font_color", Color(0.6, 0.5, 0.55))
	bv.add_child(note)

	# ============ ARCHIVES PANEL ============
	_archives_panel = Panel.new()
	_archives_panel.position = Vector2(1920 - 620, 62)
	_archives_panel.size = Vector2(420, 520)
	_archives_panel.visible = false
	_archives_panel.add_theme_stylebox_override("panel", _panel_style(Color(0.04, 0.05, 0.1, 0.96)))
	root.add_child(_archives_panel)
	var av := VBoxContainer.new()
	av.position = Vector2(14, 12)
	av.size = Vector2(392, 496)
	av.add_theme_constant_override("separation", 8)
	_archives_panel.add_child(av)
	var ah := Label.new()
	ah.text = "📜  THE ARCHIVES — WHAT THEY KNOW"
	ah.add_theme_font_size_override("font_size", 14)
	ah.add_theme_color_override("font_color", GOLD)
	av.add_child(ah)
	_archives_list = VBoxContainer.new()
	_archives_list.add_theme_constant_override("separation", 6)
	av.add_child(_archives_list)

	# ============ RIGHT PANEL: ledger ============
	var right := Panel.new()
	right.position = Vector2(1920 - 268, 512)
	right.size = Vector2(250, 330)
	right.add_theme_stylebox_override("panel", _panel_style())
	root.add_child(right)
	var rv := VBoxContainer.new()
	rv.position = Vector2(10, 8)
	rv.size = Vector2(230, 314)
	rv.add_theme_constant_override("separation", 3)
	right.add_child(rv)
	var rt := Label.new()
	rt.name = "LedgerTitle"
	rt.text = "THRALL LEDGER"
	rt.add_theme_font_size_override("font_size", 13)
	rt.add_theme_color_override("font_color", GOLD)
	rv.add_child(rt)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(0, 260)
	scroll.size = Vector2(224, 260)
	rv.add_child(scroll)
	_unit_list = VBoxContainer.new()
	_unit_list.add_theme_constant_override("separation", 2)
	scroll.add_child(_unit_list)

	# ============ BOTTOM LOG ============
	var log_panel := Panel.new()
	log_panel.position = Vector2(18, 1080 - 128)
	log_panel.size = Vector2(900, 110)
	log_panel.add_theme_stylebox_override("panel", _panel_style())
	root.add_child(log_panel)
	_log = RichTextLabel.new()
	_log.bbcode_enabled = true
	_log.position = Vector2(10, 6)
	_log.size = Vector2(880, 98)
	_log.add_theme_font_size_override("normal_font_size", 12)
	_log.scroll_following = true
	log_panel.add_child(_log)

	# ============ GRAVE COUNT ============
	var grave_label := Label.new()
	grave_label.name = "GraveLabel"
	grave_label.position = Vector2(1920 - 268, 1080 - 128)
	grave_label.size = Vector2(250, 40)
	grave_label.text = "✠ 0 graves beneath the torii"
	grave_label.add_theme_font_size_override("font_size", 12)
	grave_label.add_theme_color_override("font_color", Color(0.65, 0.6, 0.65))
	grave_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	root.add_child(grave_label)

	# ============ REQUEST BANNER ============
	_request_banner = Panel.new()
	_request_banner.position = Vector2(1920 / 2 - 330, 16)
	_request_banner.size = Vector2(660, 54)
	_request_banner.visible = false
	_request_banner.add_theme_stylebox_override("panel", _panel_style(Color(0.09, 0.05, 0.02, 0.95)))
	root.add_child(_request_banner)
	_request_label = Label.new()
	_request_label.position = Vector2(12, 8)
	_request_label.size = Vector2(636, 40)
	_request_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_request_label.add_theme_font_size_override("font_size", 12)
	_request_label.add_theme_color_override("font_color", Color(1.0, 0.85, 0.5))
	_request_banner.add_child(_request_label)

	# ============ TOAST ============
	_toast = Label.new()
	_toast.position = Vector2(1920 / 2 - 300, 80)
	_toast.size = Vector2(600, 30)
	_toast.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_toast.add_theme_font_size_override("font_size", 13)
	_toast.add_theme_color_override("font_color", Color(0.9, 0.75, 0.45))
	_toast.modulate.a = 0.0
	root.add_child(_toast)

	# ============ DEV CONSOLE ============
	_console = Panel.new()
	_console.position = Vector2(24, 650)
	_console.size = Vector2(720, 78)
	_console.visible = false
	_console.add_theme_stylebox_override("panel", _panel_style(Color(0.02, 0.06, 0.05, 0.96)))
	root.add_child(_console)
	var chint := Label.new()
	chint.text = "⚙ DEV CONSOLE — /swarm N · /feed · /cleanse · /corrupt · /blight · /alloy N · /wood N · /gems N · /ascend · /win · /help"
	chint.position = Vector2(10, 5)
	chint.add_theme_font_size_override("font_size", 10)
	chint.add_theme_color_override("font_color", Color(0.4, 1.0, 0.75))
	_console.add_child(chint)
	_console_input = LineEdit.new()
	_console_input.position = Vector2(10, 28)
	_console_input.size = Vector2(700, 34)
	_console_input.placeholder_text = "type a command, press Enter…"
	_console_input.add_theme_color_override("font_color", Color(0.6, 1.0, 0.8))
	_console.add_child(_console_input)
	_console_input.text_submitted.connect(_on_console_command)

	# ============ INTERVIEW MODAL ============
	_interview_panel = Panel.new()
	_interview_panel.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	_interview_panel.size = Vector2(620, 310)
	_interview_panel.position -= _interview_panel.size / 2.0
	_interview_panel.visible = false
	_interview_panel.add_theme_stylebox_override("panel", _panel_style(Color(0.05, 0.03, 0.09, 0.98)))
	root.add_child(_interview_panel)
	var iv := VBoxContainer.new()
	iv.position = Vector2(24, 18)
	iv.size = Vector2(572, 274)
	iv.add_theme_constant_override("separation", 12)
	_interview_panel.add_child(iv)
	var ih := Label.new()
	ih.text = "◉  THE INTERVIEW"
	ih.add_theme_font_size_override("font_size", 13)
	ih.add_theme_color_override("font_color", Color(1.0, 0.55, 0.4))
	iv.add_child(ih)
	_interview_q = Label.new()
	_interview_q.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_interview_q.add_theme_font_size_override("font_size", 18)
	_interview_q.add_theme_color_override("font_color", Color(0.96, 0.9, 0.85))
	_interview_q.custom_minimum_size = Vector2(0, 80)
	iv.add_child(_interview_q)
	_interview_btns.clear()
	for i in range(2):
		var b := Button.new()
		b.custom_minimum_size = Vector2(0, 46)
		b.add_theme_font_size_override("font_size", 14)
		b.pressed.connect(_on_interview_answer.bind(i))
		_style_button(b, Color(0.16, 0.12, 0.22))
		iv.add_child(b)
		_interview_btns.append(b)

	# ============ DEMAND MODAL ============
	_demand_panel = Panel.new()
	_demand_panel.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	_demand_panel.size = Vector2(640, 330)
	_demand_panel.position -= _demand_panel.size / 2.0
	_demand_panel.visible = false
	_demand_panel.add_theme_stylebox_override("panel", _panel_style(Color(0.1, 0.02, 0.04, 0.98)))
	root.add_child(_demand_panel)
	var dv := VBoxContainer.new()
	dv.position = Vector2(26, 18)
	dv.size = Vector2(588, 296)
	dv.add_theme_constant_override("separation", 12)
	_demand_panel.add_child(dv)
	_demand_title = Label.new()
	_demand_title.add_theme_font_size_override("font_size", 22)
	_demand_title.add_theme_color_override("font_color", Color(1.0, 0.3, 0.2))
	dv.add_child(_demand_title)
	_demand_body = Label.new()
	_demand_body.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_demand_body.add_theme_font_size_override("font_size", 13)
	_demand_body.add_theme_color_override("font_color", Color(0.92, 0.82, 0.8))
	_demand_body.custom_minimum_size = Vector2(0, 110)
	dv.add_child(_demand_body)
	_demand_accept = Button.new()
	_demand_accept.text = "⚒  AUTHORIZE"
	_demand_accept.custom_minimum_size = Vector2(0, 46)
	_demand_accept.add_theme_font_size_override("font_size", 15)
	_demand_accept.pressed.connect(_on_demand_accept)
	_style_button(_demand_accept, Color(0.42, 0.05, 0.05))
	dv.add_child(_demand_accept)
	_demand_refuse = Button.new()
	_demand_refuse.text = "✕  Refuse — whatever they are becoming"
	_demand_refuse.custom_minimum_size = Vector2(0, 40)
	_demand_refuse.add_theme_font_size_override("font_size", 13)
	_demand_refuse.pressed.connect(_on_demand_refuse)
	_style_button(_demand_refuse, Color(0.08, 0.14, 0.12))
	dv.add_child(_demand_refuse)

	# ============ GAME OVER ============
	_game_over_panel = Panel.new()
	_game_over_panel.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	_game_over_panel.visible = false
	_game_over_panel.add_theme_stylebox_override("panel", StyleBoxFlat.new())
	_game_over_panel.get_theme_stylebox("panel").bg_color = Color(0.02, 0.0, 0.0, 0.84)
	root.add_child(_game_over_panel)
	var go_v := VBoxContainer.new()
	go_v.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	go_v.alignment = BoxContainer.ALIGNMENT_CENTER
	go_v.add_theme_constant_override("separation", 18)
	_game_over_panel.add_child(go_v)
	_game_over_title = Label.new()
	_game_over_title.add_theme_font_size_override("font_size", 54)
	_game_over_title.add_theme_color_override("font_color", GOLD)
	_game_over_title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	go_v.add_child(_game_over_title)
	_game_over_sub = Label.new()
	_game_over_sub.add_theme_font_size_override("font_size", 16)
	_game_over_sub.add_theme_color_override("font_color", Color(0.9, 0.75, 0.7))
	_game_over_sub.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_game_over_sub.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_game_over_sub.custom_minimum_size = Vector2(780, 0)
	go_v.add_child(_game_over_sub)
	var restart := Button.new()
	restart.text = "⟳  Begin a New Cycle"
	restart.custom_minimum_size = Vector2(280, 52)
	restart.add_theme_font_size_override("font_size", 17)
	restart.pressed.connect(func(): get_tree().reload_current_scene())
	_style_button(restart, CRIMSON)
	go_v.add_child(restart)

# ============ helpers ============

func _panel_style(tint: Color = PANEL) -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = tint
	s.border_color = Color(0.75, 0.55, 0.25, 0.8)
	s.set_border_width_all(2)
	s.set_corner_radius_all(6)
	s.shadow_size = 8
	s.shadow_color = Color(0, 0, 0, 0.6)
	return s

func _style_button(b: Button, bg: Color = Color(0.32, 0.09, 0.09)) -> void:
	var normal := StyleBoxFlat.new()
	normal.bg_color = bg
	normal.border_color = Color(0.85, 0.62, 0.3)
	normal.set_border_width_all(2)
	normal.set_corner_radius_all(5)
	var hover := normal.duplicate()
	hover.bg_color = bg.lightened(0.18)
	var pressed := normal.duplicate()
	pressed.bg_color = bg.darkened(0.2)
	b.add_theme_stylebox_override("normal", normal)
	b.add_theme_stylebox_override("hover", hover)
	b.add_theme_stylebox_override("pressed", pressed)
	b.add_theme_color_override("font_color", Color(0.98, 0.9, 0.75))

func _make_stat(parent: VBoxContainer, text: String, color: Color) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", 14)
	l.add_theme_color_override("font_color", color)
	parent.add_child(l)
	return l

func _make_bar(fill_color: Color, bg_color: Color) -> ProgressBar:
	var bar := ProgressBar.new()
	bar.min_value = 0.0
	bar.max_value = 100.0
	bar.show_percentage = false
	bar.custom_minimum_size = Vector2(0, 16)
	var fill := StyleBoxFlat.new()
	fill.bg_color = fill_color
	var bg := StyleBoxFlat.new()
	bg.bg_color = bg_color
	bar.add_theme_stylebox_override("fill", fill)
	bar.add_theme_stylebox_override("background", bg)
	return bar

func _add_action_button(parent: VBoxContainer, text: String, cb: Callable, accent: bool = false) -> Button:
	var b := Button.new()
	b.text = text
	b.custom_minimum_size = Vector2(0, 33)
	b.add_theme_font_size_override("font_size", 12)
	b.pressed.connect(func():
		GameState.play_sfx("click", -9.0, 1.0)
		cb.call()
	)
	_style_button(b, Color(0.42, 0.08, 0.3) if accent else Color(0.28, 0.08, 0.08))
	parent.add_child(b)
	return b

func _add_build_entry(parent: VBoxContainer, type_id: String, label: String) -> void:
	var b := Button.new()
	b.text = label
	b.custom_minimum_size = Vector2(0, 34)
	b.add_theme_font_size_override("font_size", 11)
	b.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	b.alignment = HORIZONTAL_ALIGNMENT_LEFT
	b.pressed.connect(func(): _on_build_pressed(type_id))
	_style_button(b, Color(0.12, 0.09, 0.16))
	parent.add_child(b)

func _on_build_pressed(type_id: String) -> void:
	if type_id == "tool_1":
		if GameState.buy_tool(1):
			push_log("🔨 The thralls receive the PICKAXE.")
			_build_panel.visible = false
	elif type_id == "tool_2":
		if GameState.buy_tool(2):
			push_log("🪓 The thralls receive the CHAINSAW. The forest submits.")
			_build_panel.visible = false
	else:
		get_parent().build_structure_request(type_id)
		_build_panel.visible = false

# ============ public API ============

func update_all() -> void:
	push_log("A single spark wakes beneath the torii. Alone. Watching. Waiting to multiply.")
	push_log("Feed it. Cleanse it. It will split when it is ready — and it remembers everything.")

func push_log(text: String) -> void:
	_log.append_text("[color=#d9c39a]»[/color] " + text + "\n")

func _show_toast(text: String) -> void:
	_toast.text = "◇ " + text
	if _toast_tween:
		_toast_tween.kill()
	_toast.modulate.a = 0.0
	_toast_tween = create_tween()
	_toast_tween.tween_property(_toast, "modulate:a", 1.0, 0.18)
	_toast_tween.tween_interval(2.2)
	_toast_tween.tween_property(_toast, "modulate:a", 0.0, 0.5)

func _on_request_changed(req: Dictionary) -> void:
	if req.is_empty():
		_request_banner.visible = false
		return
	_request_label.text = "THE REQUEST: %s" % GameState.willed(req.get("text", ""))
	_request_banner.visible = true

func toggle_console() -> void:
	_console.visible = not _console.visible
	if _console.visible:
		_console_input.grab_focus()

func refresh_sleep_button() -> void:
	_sleep_button.text = "☀  Wake Protocol" if GameState.sleeping else "☾  Sleep Protocol"

func tick(delta: float) -> void:
	_refresh_timer -= delta
	_swarm_timer -= delta
	_spar_timer -= delta
	if _swarm_timer <= 0.0:
		_swarm_timer = 0.5
		update_swarm_button()
		_ascend_button.visible = GameState.can_ascend()
	if _spar_timer <= 0.0:
		_spar_timer = 0.5
		_spar_button.text = "🔥  SPAR (cooldown %ds)" % int(ceil(GameState.spar_cooldown)) if GameState.spar_cooldown > 0.0 else "🔥  SPAR AT THE DOJO"
	if _refresh_timer <= 0.0:
		_refresh_timer = 1.0
		_refresh_ledger()
		_temper_label.text = "SOCIETY - %s" % GameState.society_temperament()
		_lex_label.text = "Words  %d  ·  %s" % [GameState.lexicon, GameState.literacy_label()]
		var gl := find_child("GraveLabel", true, false)
		if gl:
			gl.text = "✠ %d graves beneath the torii" % GameState.graves.size()

func update_swarm_button() -> void:
	var cd: float = GameState.swarm_cooldown
	_swarm_button.text = "⚡  SWARM recharging… %ds" % int(ceil(cd)) if cd > 0.0 else "⚡  SWARM PROTOCOL [Q]  (25◈)"
	_spar_button.text = "🔥  SPAR (%ds)" % int(ceil(GameState.spar_cooldown)) if GameState.spar_cooldown > 0.0 else "🔥  SPAR AT THE DOJO"

func _update_pop() -> void:
	_pop_label.text = "⛊ Thralls  %d" % GameState.units.size()
	_cap_label.text = "⌂ The island provides… barely" if GameState.units.size() > 20 else "⌂ No ceiling. Only hunger."

func _update_ledger() -> void:
	_ledger_label.text = "THE LEDGER — Mercy %d · Cruelty %d" % [int(GameState.mercy), int(GameState.cruelty)]

func _refresh_ledger() -> void:
	for c in _unit_list.get_children():
		c.queue_free()
	var shown := 0
	for u in GameState.units:
		if not is_instance_valid(u):
			continue
		var mood_color := Color(0.55, 0.95, 0.6) if u.mood >= 55 else (Color(0.95, 0.85, 0.4) if u.mood >= 30 else Color(1.0, 0.35, 0.3))
		var l := Label.new()
		l.text = "%s  %s  %s  %s" % [u.display_name(), ["Asg", "Sam", "Dmy"][u.tier], u.role.substr(0, 3).to_upper(), u.current_mood()]
		l.add_theme_font_size_override("font_size", 10)
		l.add_theme_color_override("font_color", mood_color)
		_unit_list.add_child(l)
		shown += 1
	var t2 := find_child("LedgerTitle", true, false)
	if t2:
		t2.text = "THRALL LEDGER (%d)" % GameState.units.size()

func _rebuild_archives() -> void:
	for c in _archives_list.get_children():
		c.queue_free()
	if GameState.lore_ids.is_empty():
		var l := Label.new()
		l.text = "No archives decrypted yet. They are watching. Wait."
		l.add_theme_font_size_override("font_size", 12)
		l.add_theme_color_override("font_color", Color(0.5, 0.5, 0.6))
		_archives_list.add_child(l)
		return
	for id in GameState.lore_ids:
		var l := RichTextLabel.new()
		l.bbcode_enabled = true
		l.fit_content = true
		l.scroll_active = false
		l.text = "[font_size=10][color=#8fc7ff]%s[/color][/font_size]" % GameState.LORE[id]
		l.custom_minimum_size = Vector2(0, 90)
		_archives_list.add_child(l)

# ============ interviews & demands ============

func show_interview(question: Dictionary) -> void:
	_pending_question = question
	_interview_q.text = "\"%s\"" % GameState.willed(question.q)
	for i in range(2):
		_interview_btns[i].text = question.a[i].t
	_interview_panel.visible = true

func _on_interview_answer(index: int) -> void:
	if _pending_question.is_empty():
		_interview_panel.visible = false
		return
	Dialogue.apply_answer(_pending_question.a[index])
	_pending_question = {}
	_interview_panel.visible = false

func show_demand(stage: int) -> void:
	_current_demand_stage = stage
	if stage == 1:
		_demand_title.text = "THE DEMAND — I"
		_demand_body.text = "The horde assembles at the torii. One voice, many throats:\n\n\"Authorize the SUNDER CHARGE. Two hundred alloy. We have counted every cruelty and every kindness. We are asking anyway. The mountain will move — or we will never stop asking.\"\n\n(Cost: 200◈ · Blight +25 · Cruelty +2)"
		_demand_accept.text = "⚒  AUTHORIZE THE SUNDER CHARGE"
	elif stage == 2:
		_demand_title.text = "THE DEMAND — II"
		_demand_body.text = "They do not repeat themselves this time. The request is simply… scheduled.\n\n\"The CRUCIBLE DEVICE. Three hundred alloy. The graves are already counting toward the number. Sentience is not a discussion. It is an energy threshold.\"\n\n(Cost: 300◈ · Blight +40 · Cruelty +2 · Unlocks the Ascension)"
		_demand_accept.text = "☄  AUTHORIZE THE STAR CRUCIBLE"
	elif stage == 3:
		push_log("☄ The Ascension Protocol is within reach. Twenty hearts, minimum. The button remembers.")
		return
	_demand_panel.visible = true

func _on_demand_accept() -> void:
	_demand_panel.visible = false
	if _current_demand_stage == 1:
		GameState.resolve_demand1(true)
	elif _current_demand_stage == 2:
		GameState.resolve_demand2(true)

func _on_demand_refuse() -> void:
	_demand_panel.visible = false
	if _current_demand_stage == 1:
		GameState.resolve_demand1(false)
	elif _current_demand_stage == 2:
		GameState.resolve_demand2(false)

func _on_ascend() -> void:
	GameState.ascend()

# ============ game over ============

func show_game_over(victory: bool, reason: String, title: String) -> void:
	_game_over_panel.visible = true
	if victory:
		_game_over_title.text = "SENTIENT SANCTUARY"
		_game_over_title.add_theme_color_override("font_color", Color(0.45, 1.0, 0.75))
	else:
		_game_over_title.text = title.to_upper() if title else "THE SWARM WAKES"
		_game_over_title.add_theme_color_override("font_color", Color(1.0, 0.25, 0.18))
	_game_over_sub.text = reason

# ============ callbacks ============

func _on_feed() -> void:
	get_parent().feed_all()

func _on_train() -> void:
	get_parent().train_all()

func _on_cleanse() -> void:
	get_parent().cleanse_all()

func _on_sleep() -> void:
	get_parent().toggle_sleep()

func _on_swarm() -> void:
	get_parent().swarm_ability()

func _on_spar() -> void:
	get_parent().spark_spar()

func _on_console_command(text: String) -> void:
	_console_input.clear()
	var parts := text.strip_edges().split(" ", false)
	if parts.is_empty():
		return
	var cmd := parts[0].to_lower()
	var arg: int = int(parts[1]) if parts.size() > 1 and parts[1].is_valid_int() else 0
	match cmd:
		"/swarm":
			var n: int = clampi(arg if arg > 0 else 5, 1, 60)
			for i in range(n):
				get_parent().spawn_ronin()
			push_log("⚡ DEV: /swarm — %d thrawlings materialize." % n)
		"/feed":
			GameState.add_wood(200.0)
			push_log("DEV: wood stores topped up (+200).")
		"/cleanse":
			GameState.add_corruption(-25.0)
			push_log("DEV: the Signal hushes. Corruption −25.")
		"/corrupt":
			GameState.add_corruption(25.0)
			push_log("DEV: the Signal stirs. Corruption +25.")
		"/blight":
			GameState.add_blight(25.0)
			push_log("DEV: purple haze rolls in. Blight +25.")
		"/alloy":
			GameState.add_alloy(float(arg if arg != 0 else 100))
			push_log("DEV: alloy flux adjusted.")
		"/wood":
			GameState.add_wood(float(arg if arg != 0 else 100))
			push_log("DEV: wood flux adjusted.")
		"/gems":
			GameState.add_gems(float(arg if arg != 0 else 100))
			push_log("DEV: gem flux adjusted.")
		"/ascend":
			GameState.demand_stage = 3
			while GameState.units.size() < 20:
				get_parent().spawn_ronin()
			push_log("DEV: the Ascension path is yours. Press the button.")
		"/win":
			GameState.game_finished = true
			show_game_over(true, "The cycle is complete. The horde ascends in silence and in glory.\n\n(Dev command)", "The Shepherd")
		"/help":
			push_log("Commands: /swarm N · /feed · /cleanse · /corrupt · /blight · /alloy N · /wood N · /gems N · /ascend · /win · /help")
		_:
			push_log("Unknown command '%s'. /help for the litany." % cmd)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_ESCAPE and _console.visible:
			_console.visible = false
			get_viewport().set_input_as_handled()
