# C: Drive Root-Cause & Cleanup — 2026-11-23

## Root cause (evidence: `disk_rootcause.ps1`)
| Consumer | Size | Status |
|---|---:|---|
| `C:\pagefile.sys` | 21.0 GB | Windows-managed, left alone |
| HF model cache (Phi-3-mini, bart, llama, distilbert) | ~16 GB | **Active models agent — blocked** |
| `02_QUILLAN\exports` + model weights | ~20 GB | **Active models agent — blocked** |
| Soulframe game cache | ~13 GB | User chose to keep (re-download cost) |
| `02_QUILLAN\.git` pack + 14 kept LFS objects | ~10 GB | Repo history, intentional |
| Ollama model blobs | ~6 GB | Active runtime |
| WSL vhdx | 3.7 GB | Active WSL |
| 4× `opencode.db` copies | ~8 GB | 1 active + 3 stale snapshots |
| Duplicate broken repo `CascadeProjects\Quillan-Ronin` | ~3.3 GB | No HEAD, 0 branches |

## Cleanup executed (approved)
| Target | Result |
|---|---:|
| `05_ARCHIVE\Backups\opencode_20260907_125859.db` | deleted, 1.96 GB |
| `04_CONFIG\Development\share\opencode\opencode.db` | deleted, 1.45 GB |
| `CascadeProjects\Quillan-Ronin` broken clone | deleted, 3.27 GB |
| `temp_opencode_copy.db` | **BLOCKED — locked by a running process** (1.93 GB, hashable, not read-only, admin-owned) |

**Freed: 6.52 GB. C: 12.63 GB → 19.28 GB free.**

Ledger: `C_CLEANUP_LEDGER_20261123.json`.

## Why not more
- 80 GB free is unreachable while the models agent writes 14 GB HF cache + 47 GB exports
- Soulframe cache deliberately preserved per user decision
- Active `opencode.db` (2.86 GB) cannot be touched while OpenCode runs
- `temp_opencode_copy.db` needs its holding process closed first

## Repo objective (also completed this turn)
All 7 canonical `leeex1` repos verified MATCH remote HEAD. Main repo published at
`83f88232` with the organized structure; 2 missing mirrors cloned; chrome-extension
mirror fast-forwarded. Full evidence in `REPO_SYNC_REPORT.md`.
