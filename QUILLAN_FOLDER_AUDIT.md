# Quillan Folder Audit — 2026-11-23

## Size breakdown (before this cleanup, measured)
| Folder | Size |
|---|---:|
| exports | 47.48 GB |
| 05 - Creative Works | 20.12 GB |
| 09 - Projects | 16.82 GB |
| .git | 15.47 GB |
| 03 - Training & Model | 7.01 GB |
| 02 - Knowledge Foundation | 5.83 GB |
| colab_bundle | 1.70 GB |
| docs | 1.27 GB |
| 07 - Memory & LanceDB | 1.26 GB |
| **Total** | **119.77 GB** |

## Export contents
- `donor_cache_v23` 12.66 GB (active donor models)
- `snapshots_pre_phase0` 3.34 GB (stale backups — removed)
- `quillan_1b_sovereign_hf` 2.81 GB, `quillan_mini_hf` 1.18 GB (active)

## Cleanup executed (user-approved)
| Target | Files | Freed | Evidence |
|---|---:|---:|---|
| `knowledge\legacy\00 - Meta\venv_oni_gpu` | 18,402 | 5.26 GB | Last use 2026-09-04; zero executable references |
| `exports\snapshots_pre_phase0` | 5 | 3.34 GB | Phase-0 backup weights; zero script references |
| `colab_bundle` | 184 | 1.70 GB | Zero references; last use 2026-10-18 |
| **Total** | **18,591** | **10.31 GB** | Ledger: `QUILLAN_STALE_CLEANUP_20261123.json` |

Verification: all three paths absent; active `venv_oni_cu126` intact (used by
`qdev.bat` and `launch_12l_samurai_train.bat`); git shows **0 tracked deletions**
(all removed files were gitignored).

## Why the folder is still large
- exports 47 GB: donor caches + model weights (models agent actively working)
- Creative Works 20 GB: media (user-protected)
- Projects 16.8 GB: active venvs and repos
- .git 15.5 GB: repo history + remaining large LFS objects (kept by decision)

Further reduction requires decisions on active model weights or media — neither touched.
