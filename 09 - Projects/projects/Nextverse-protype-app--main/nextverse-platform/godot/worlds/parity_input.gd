extends Node
## Shared locomotion contract for VR and desktop.
## Both paths produce the same Vector3 in the same coordinate space; only the
## input source differs. This is the parity foundation for OpenXR integration.

const MOVE_SPEED: float = 3.0

func from_keyboard(pressed: Dictionary) -> Vector3:
	var direction := Vector3.ZERO
	if bool(pressed.get("w", false)): direction.z -= 1.0
	if bool(pressed.get("s", false)): direction.z += 1.0
	if bool(pressed.get("a", false)): direction.x -= 1.0
	if bool(pressed.get("d", false)): direction.x += 1.0
	return direction

func from_xr_axes(primary: Vector2, secondary: Vector2) -> Vector3:
	# Primary stick: planar movement. Secondary stick: turn/strafe assist.
	var direction := Vector3(primary.x + secondary.x * 0.5, 0.0, primary.y)
	return direction

func apply_movement(direction: Vector3, delta: float) -> Vector3:
	if direction == Vector3.ZERO:
		return Vector3.ZERO
	return direction.normalized() * MOVE_SPEED * delta

func parity_delta(desktop: Vector3, xr: Vector3, delta: float) -> float:
	return apply_movement(desktop, delta).distance_to(apply_movement(xr, delta))
