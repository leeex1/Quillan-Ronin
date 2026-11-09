@tool
extends EditorPlugin

var dock: Control

func _enter_tree() -> void:
	dock = preload("res://addons/ronin_ai/ai_dock.gd").new()
	dock.title = "Ronin AI"
	add_dock(dock, DOCK_SLOT_RIGHT_UL)

func _exit_tree() -> void:
	remove_dock(dock)
	dock.queue_free()
