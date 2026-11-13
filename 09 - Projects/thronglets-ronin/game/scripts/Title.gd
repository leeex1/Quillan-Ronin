extends Control
## Title screen — declarative UI in Title.tscn. This script only wires the buttons.

func _ready() -> void:
	var start := get_node_or_null("%StartButton")
	var quit := get_node_or_null("%QuitButton")
	if start:
		start.pressed.connect(func():
			print("TITLE: BEGIN pressed — entering the dojo")
			get_tree().change_scene_to_file("res://scenes/Main.tscn")
		)
	else:
		print("TITLE: ERROR — StartButton not found")
	if quit:
		quit.pressed.connect(func(): get_tree().quit())
	print("TITLE: menu wired, children = ", get_child_count())
