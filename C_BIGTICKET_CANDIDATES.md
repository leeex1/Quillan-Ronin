# C: Large Installed Software — Review Before Any Uninstall

All sizes measured 2026-11-23. Nothing here has been touched.

## GPU / development toolchains (highest space, highest dependency risk)
| Item | Size | Risk |
|---|---:|---|
| `Program Files\NVIDIA GPU Computing Toolkit` | 15.83 GB | Likely needed for CUDA model work — verify before removing |
| `Program Files\NVIDIA Corporation` | 5.99 GB | Driver/control panel components |
| `ProgramData\NVIDIA Corporation` | 5.39 GB | Driver cache/installer data |
| `Program Files\LLVM` | 2.83 GB | Compiler toolchain |
| `Program Files (x86)\Microsoft Visual Studio` | 7.05 GB | IDE/build tools |
| `Program Files (x86)\Windows Kits` | 3.36 GB | Windows SDK |

## Games (user info: Mount & Blade installed, Soulframe present)
| Item | Size | Risk |
|---|---:|---|
| `Program Files (x86)\Steam` | 14.61 GB | Contains Mount & Blade Warband — keep |
| `AppData\Local\Soulframe` | 12.59 GB | Game + ~13 GB caches; user previously chose to keep |
| `Program Files\Epic Games` (EU IV) | 3.33 GB | Installed game |
| `Program Files (x86)\Epic Games` | 2.51 GB | Launcher/services |
| `AppData\Local\CapCut` | 2.90 GB | Video editor |
| `AppData\Roaming\Godot` | 1.95 GB | Godot engine data |

## Dev tools / caches (installed, not plain caches)
| Item | Size | Risk |
|---|---:|---|
| `AppData\Roaming\npm\node_modules` | 6.63 GB | Global npm packages — needed by CLIs |
| `AppData\Local\hermes\hermes-agent` | 5.10 GB | Active agent runtime |
| `AppData\Local\Programs\Python` | 4.71 GB | Multiple Python installs |
| `AppData\Local\BraveSoftware` | 4.47 GB | Browser profile/cache |
| `AppData\Local\Google` | 3.85 GB | Chrome/ Electron data |
| `AppData\Local\wsl` | 3.66 GB | WSL distros |
| `AppData\Roaming\uv` (python/tools) | 2.10 GB | Installed Python/tool versions |
| `AppData\Local\uv\cache` | ~21 MB | Already cleaned (1.5 GB reclaimed) |

## System (do not delete manually)
- `C:\pagefile.sys` 21 GB — 27.9 GB RAM, 26 GB committed, 0 GB available: **required**
- `C:\Windows\WinSxS` 9.02 GB — DISM component cleanup needs admin
- `C:\Windows\Installer` 4.34 GB — required for uninstall/repair

## Cleanup already completed this session (outside 02_QUILLAN)
| Action | Freed |
|---|---:|
| Stale opencode.db snapshots ×2 + broken repo clone | 6.52 GB |
| Temp files older than 7 days (14,817 files) | 2.77 GB |
| uv wheel cache (30,050 files) | 1.5 GB |
| Browser/app caches (18 dirs, 19,296 files) | 2.51 GB |
| **Total** | **~13.3 GB** |

C: free: 12.63 GB → **25.1 GB** (other agents write concurrently, so net varies).

## Per-item approval needed (no action taken)
Large installed software cannot be removed without your explicit decision because
active agents may depend on it:
- NVIDIA GPU Computing Toolkit 15.83 GB + NVIDIA Corporation 5.99 GB + ProgramData NVIDIA 5.39 GB
- Steam 14.61 GB (contains Mount & Blade Warband)
- Soulframe 12.59 GB (user previously chose to keep)
- Visual Studio 7.05 GB + Windows Kits 3.36 GB
- Global npm packages 6.63 GB, hermes-agent 5.10 GB, Python installs 4.71 GB
- VSIX repair payloads 2.23 GB (needed for VS repair — do not delete)
