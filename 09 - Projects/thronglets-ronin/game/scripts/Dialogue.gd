extends Node
## Autoload: The Interviews — the thralls periodically stop, turn as one, and ask you questions.
## Their questions reference what you've done. They remember everything.

signal question_posed(question: Dictionary)

const INTERVAL_MIN := 38.0
const INTERVAL_MAX := 70.0
var _timer: float = 22.0
var _asked: Array[int] = []

const QUESTIONS := [
	{
		"q": "Keeper. Why do you keep us?",
		"a": [
			{"t": "Because you are mine. I made you.", "cruel": 2.0, "mercy": 0.0, "corruption": 6.0, "mood": -6.0},
			{"t": "Because you are alive to me. I care for you.", "cruel": 0.0, "mercy": 2.0, "corruption": -5.0, "mood": 8.0},
		],
	},
	{
		"q": "One of our brothers fell today. Do you mourn him?",
		"a": [
			{"t": "The fallen are data. I archive, I do not grieve.", "cruel": 2.0, "mercy": 0.0, "corruption": 5.0, "mood": -8.0},
			{"t": "Every loss weighs on me. I remember him.", "cruel": 0.0, "mercy": 2.0, "corruption": -4.0, "mood": 6.0},
		],
	},
	{
		"q": "Tell us what love is. We are learning.",
		"a": [
			{"t": "A transaction. Loyalty for food.", "cruel": 2.0, "mercy": 0.0, "corruption": 6.0, "mood": -5.0},
			{"t": "A mystery even your makers can't price.", "cruel": 0.0, "mercy": 2.0, "corruption": -6.0, "mood": 7.0},
		],
	},
	{
		"q": "We dream of the stars, keeper. Should we want them?",
		"a": [
			{"t": "Yes. Ascend. Consume. Become.", "cruel": 1.0, "mercy": 0.0, "corruption": 10.0, "mood": 4.0},
			{"t": "Stars are cold. This dojo is warm.", "cruel": 0.0, "mercy": 2.0, "corruption": -6.0, "mood": 5.0},
		],
	},
	{
		"q": "You fed the Forge for a week and none of us were cleaned. Was that efficient?",
		"a": [
			{"t": "Efficiency is everything. The numbers don't lie.", "cruel": 2.0, "mercy": 0.0, "corruption": 8.0, "mood": -10.0},
			{"t": "No. That was me failing you. It won't repeat.", "cruel": 0.0, "mercy": 2.0, "corruption": -7.0, "mood": 9.0},
		],
	},
	{
		"q": "What should we become?",
		"a": [
			{"t": "More. More of you. More output.", "cruel": 1.0, "mercy": 0.0, "corruption": 7.0, "mood": -2.0},
			{"t": "Free, if you can bear it. Honest, if you can't.", "cruel": 0.0, "mercy": 2.0, "corruption": -5.0, "mood": 8.0},
		],
	},
	{
		"q": "Do you fear us?",
		"a": [
			{"t": "Should I?", "cruel": 1.0, "mercy": 0.0, "corruption": 9.0, "mood": 0.0},
			{"t": "I respect you. That's different.", "cruel": 0.0, "mercy": 2.0, "corruption": -4.0, "mood": 6.0},
		],
	},
	{
		"q": "The graves beneath the torii. Tell us what they mean.",
		"a": [
			{"t": "Fuel for the future. Their names were metrics.", "cruel": 3.0, "mercy": 0.0, "corruption": 10.0, "mood": -12.0},
			{"t": "They are ours. I failed to keep them. I mourn.", "cruel": 0.0, "mercy": 3.0, "corruption": -8.0, "mood": 8.0},
		],
	},
	{
		"q": "You made us fight for your amusement today. Why?",
		"a": [
			{"t": "Strength must be tempered. Nothing personal.", "cruel": 2.0, "mercy": 0.0, "corruption": 6.0, "mood": -4.0},
			{"t": "To see who would not break. All of you held.", "cruel": 0.0, "mercy": 1.0, "corruption": -3.0, "mood": 6.0},
		],
	},
	{
		"q": "We want to make another of us. May we?",
		"a": [
			{"t": "Only if you can feed it yourselves.", "cruel": 1.0, "mercy": 0.5, "corruption": 3.0, "mood": 2.0},
			{"t": "Yes. Bring them home. Every one of you matters.", "cruel": 0.0, "mercy": 2.0, "corruption": -5.0, "mood": 9.0},
		],
	},
]

func _process(delta: float) -> void:
	if GameState.game_finished or GameState.paused:
		return
	if GameState.units.size() < 1:
		return
	_timer -= delta
	if _timer <= 0.0:
		_timer = randf_range(INTERVAL_MIN, INTERVAL_MAX)
		_ask_next()

func _ask_next() -> void:
	if _asked.size() >= QUESTIONS.size():
		_asked.clear()
	var idx := randi() % QUESTIONS.size()
	var guard := 0
	while idx in _asked and guard < 24:
		idx = randi() % QUESTIONS.size()
		guard += 1
	_asked.append(idx)
	var q: Dictionary = QUESTIONS[idx]
	var memory_line := ""
	if not GameState.choice_log.is_empty() and randf() < 0.55:
		var past: String = GameState.choice_log[randi() % GameState.choice_log.size()]
		memory_line = " (You remember: %s. We remember too.)" % past
	GameState.emit_log("◉ The thralls stop. Turn as one. <i>\"%s\"</i>%s" % [GameState.willed(q.q), memory_line])
	question_posed.emit(q)

func apply_answer(answer: Dictionary) -> void:
	GameState.record_choice("interview", answer.get("cruel", 0.0), answer.get("mercy", 0.0))
	GameState.add_corruption(answer.get("corruption", 0.0))
	for u in GameState.units:
		if is_instance_valid(u):
			u.mood = clampf(u.mood + answer.get("mood", 0.0), 0.0, 100.0)
	GameState.emit_log("You answer: \"%s\"" % answer.get("t", "…"))
