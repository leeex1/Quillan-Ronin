# Divergent Pair Resolution â€” 2026-11-23

All resolutions were MOVES to '12 - Backups/superseded'. NO deletions. Fully reversible.

| File | Newest kept | Older archived | Evidence | Caveat |
|---|---|---|---|---|
| modeling_quillan.py | 03 - Training & Model\modeling_quillan.py (v6.0.3) | modeling_quillan.v5.4.0-ONI.20261023.py | version string + mtime 2026-11-13 | v5.4.0 had vectorized batched MoE and _eggroll_delta_cache inference caching; v6.0.3 uses per-token nested loops (slower). Consider porting those optimizations if training perf regresses. |
| quillan_fused_optimizer.py | 03 - Training & Model\quillan_fused_optimizer.py | quillan_fused_optimizer.20261018.py | dest adds 'del all_grads' memory release, mtime 2026-11-13 | none |
| trace_signal.py | 03 - Training & Model\scripts\trace_signal.py (root copy, 14:43:29) | trace_signal.phase5_legacy.20261113_144252.py | root newer by 37s; archived copy points at stale C:\Users\Admin\Quillan-Ronin\transplant_weights | These are different diagnostic scripts, not true versions. Both preserved. |
