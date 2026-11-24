extends Node
## Port of the legacy DVVE adaptive-quality formula.
## R_p = P_core * F_v * (1 + omega_v) / (1 + nu_v)
## Defaults match game_engine/include/DVVE.hpp so numbers stay comparable.

const DEFAULT_P_CORE: float = 1.0e9
const DEFAULT_F_V: float = 1.0
const DEFAULT_OMEGA_V: float = 0.0
const DEFAULT_NU_V: float = 0.05

const FRAME_BUDGET_MS: float = 16.0

var current_tier: int = 2
var last_performance: float = 0.0

func calculate_performance(
	p_core: float = DEFAULT_P_CORE,
	f_v: float = DEFAULT_F_V,
	omega_v: float = DEFAULT_OMEGA_V,
	nu_v: float = DEFAULT_NU_V
) -> float:
	var denominator: float = 1.0 + nu_v
	if is_equal_approx(denominator, 0.0):
		return 0.0
	return p_core * f_v * (1.0 + omega_v) / denominator

func quality_tier(performance: float) -> int:
	if performance >= 1.2e9:
		return 3
	if performance >= 9.0e8:
		return 2
	return 1

## Applies the negotiated tier to the active 3D viewport.
func apply_tier(tier: int, viewport: Viewport) -> void:
	if viewport == null:
		return
	match tier:
		3: viewport.scaling_3d_scale = 1.0
		2: viewport.scaling_3d_scale = 0.85
		_: viewport.scaling_3d_scale = 0.7
	current_tier = tier

func update_from_performance(viewport: Viewport) -> int:
	last_performance = calculate_performance()
	apply_tier(quality_tier(last_performance), viewport)
	return current_tier

func frame_budget_ok(frame_ms: float) -> bool:
	return frame_ms > 0.0 and frame_ms <= FRAME_BUDGET_MS

func _ready() -> void:
	last_performance = calculate_performance()
	current_tier = quality_tier(last_performance)
	print("[PerformanceDirector] R_p=", last_performance, " tier=", current_tier,
		" budget_ms=", FRAME_BUDGET_MS)
