extends XROrigin3D
## OpenXR-capable player rig. The headset path feeds the same movement
## contract as desktop input (worlds/parity_input.gd), so VR and non-VR
## users reach the same zones with the same collision and speed.

@export var move_speed: float = 3.0

var left_controller: XRController3D
var right_controller: XRController3D
var camera: Camera3D

func _ready() -> void:
	if camera == null:
		camera = Camera3D.new()
		camera.name = "XRCamera"
		add_child(camera)
	if left_controller == null:
		left_controller = XRController3D.new()
		left_controller.name = "LeftController"
		left_controller.tracker = &"left_hand"
		add_child(left_controller)
	if right_controller == null:
		right_controller = XRController3D.new()
		right_controller.name = "RightController"
		right_controller.tracker = &"right_hand"
		add_child(right_controller)

func movement_from_axes(primary: Vector2, secondary: Vector2) -> Vector3:
	return Vector3(primary.x + secondary.x * 0.5, 0.0, primary.y)

func apply_movement(direction: Vector3, delta: float) -> Vector3:
	if direction == Vector3.ZERO:
		return Vector3.ZERO
	return direction.normalized() * move_speed * delta
