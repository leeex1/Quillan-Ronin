extends SceneTree
## Probe the Title scene: what actually gets built, and where?

var _done := false
func _initialize() -> void:
	if _done:
		return
	_done = true
	var scene = load("res://scenes/Title.tscn")
	if scene == null:
		print("PROBE: Title.tscn FAILED TO LOAD")
		quit()
		return
	var inst = scene.instantiate()
	print("PROBE: script = ", inst.get_script())
	print("PROBE: has _ready method: ", inst.has_method("_ready"))
	root.add_child(inst)  # force _ready
	print("PROBE: instantiated, children = ", inst.get_child_count())
	_dump(inst, 0)
	quit()

func _dump(n: Node, depth: int) -> void:
	if depth > 4:
		return
	print("  ".repeat(depth), n.name, " <", n.get_class(), "> visible=", (n.visible if n is CanvasItem else n.get("visible")), " size=", (n.size if n is Control else Vector2.ZERO))
	for c in n.get_children():
		_dump(c, depth + 1)
