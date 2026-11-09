@tool
extends EditorDock

var input: LineEdit
var log: RichTextLabel
var bridge: Node

func _ready() -> void:
	bridge = preload("res://addons/ronin_ai/mcp_bridge.gd").new()
	add_child(bridge)
	var vbox := VBoxContainer.new()
	vbox.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(vbox)
	input = LineEdit.new()
	input.placeholder_text = "Ask AI: make a test level / add node..."
	input.text_submitted.connect(_on_submit)
	vbox.add_child(input)
	log = RichTextLabel.new()
	log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	log.bbcode_enabled = true
	vbox.add_child(log)
	_append("[color=#ffd166]Ronin AI ready. MCP queue: mcp/queue.jsonl[/color]\n")

func _on_submit(text: String) -> void:
	if text.strip_edges() == "":
		return
	_append("[b]%s[/b]\n" % text)
	bridge.enqueue(text)
	_append("[color=#8ab4ff]Queued -> run: python mcp/server.py[/color]\n")
	input.clear()

func _append(t: String) -> void:
	log.append_text(t)
