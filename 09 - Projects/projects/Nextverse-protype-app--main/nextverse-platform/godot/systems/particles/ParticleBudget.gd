extends Node
## Phase 2: GPU particle budget derived from the DVVE performance scalar.
## The legacy alpha engine targeted 100000 particles; that ceiling is preserved.

@export var max_particles: int = 100000

func compute_amount(performance: float, fps_target: float = 60.0) -> int:
	if fps_target <= 0.0 or performance <= 0.0:
		return 0
	var affordable: int = int(performance / fps_target)
	return clampi(affordable, 0, max_particles)

func configure(particles: GPUParticles3D, performance: float, fps_target: float = 60.0) -> int:
	var amount: int = compute_amount(performance, fps_target)
	if particles != null:
		particles.amount = amount
	return amount
