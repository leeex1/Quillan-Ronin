# PC Duplicate & Space Report — 2026-11-23

## Scope rule
Only `C:\02_QUILLAN` is the Quillan folder. Downloads and E: belong to Quillan's
projects and were treated as protected — no audio or original media deleted.

## Completed duplicate scans

| Scan | Result |
|---|---|
| PC-wide (>1 MB): E:, Downloads, Desktop, Temp, `_archive` | 1,502 files scanned, **190 groups / 1,067 redundant copies / 5.47 GB** |
| E: quarantine verification | 4,306 files; **4,242 with byte-identical originals (6.14 GB)**; **64 with NO original (71.3 MB)** |
| Script dedup scan (6 first-party script dirs) | 1,166 scripts, **133 groups / 243 redundant copies** |
| Downloads audio dataset | 813 files / 2.92 GB, largely duplicating Creative Works audio |

## Protected / excluded (never deleted)
- **64 unique files** copied hash-verified to `_archive\RECOVERED_UNIQUE_20261123`
- Platform subset folders (`Claude`, `Gemini`, `GPT`, `Grok`, `Mistral`, `Perplexity`) — intentional deployment copies of canonical scripts; byte-identical but preserved
- HF model cache 14 GB — models agent actively using
- `exports` 47 GB — models agent actively using
- Steam/Epic installs: MountBlade Warband 5.54 GB, Europa Universalis IV 3.41 GB, launchers 2.57 GB
- All audio in Creative Works and Downloads
- 88 pre-existing dirty entries and all other agents' work

## Space actions completed

| Action | Result |
|---|---:|
| Earlier verified dedup (5 ledgers) | 8.12 GB reclaimed |
| Tier 2 caches (crash dump + npm + electron) | 33,678 files, 3.61 GB freed |
| Site Phase A lossless PNG recompression | 94.2 MB saved, pixel-identical |
| LFS stale index entries | 250 staged removals, 0 disk files deleted |

## Completed after approval

| Action | Result |
|---|---:|
| Tier 1 E: quarantine cleanup | 4,242 duplicates re-hashed + deleted, **6.14 GB freed**, 0 failures |
| 64 unique files protected | copied hash-verified to `_archive\RECOVERED_UNIQUE_20261123` |
| LFS pointer repair | 252 pointers materialized from local objects, 93.8 MB, 0 failures |
| LFS object cleanup | 12,811 objects <50 MB removed, **5.79 GB freed**; 14 large objects kept (7.09 GB) |
| LFS stale index entries | 250 staged removals (git history retained) |

**Current: C: 14.44 GB free / 476 GB (was 6.23), E: 6.76 GB free / 46.87 GB (was 0.35).**
Total reclaimed this turn: **~11.2 GB** on top of the earlier 8.12 GB.

## Integrity verification
- Live site: **513/513 requested assets present, 0 missing**
- Site Phase A: 249 PNGs losslessly recompressed, 94.19 MB saved, pixel-identical
- Broken-path audit: 0 orphan references
- LFS: 0 worktree pointers remaining; 14 large objects retained
- Ledgers: 9 files including `QUARANTINE_CLEANUP_LEDGER_20261123.json`,
  `LFS_MATERIALIZE_LEDGER_20261123.json`, `LFS_OBJECT_REMOVAL_LEDGER_20261123.json`

## Never touched
HF model cache (14 GB, models agent), `exports` (47 GB, models agent), Steam/Epic game
installs (MountBlade Warband 5.54 GB, Europa Universalis IV 3.41 GB), all audio/media
originals, platform deployment subset folders, and every other agent's working files.

## Reality on 80 GB target
Safe cleanup yields ~12 GB more (Tier 1 + LFS cache). Reaching 80 GB requires
decisions on active agent data (14 GB HF cache, 47 GB exports) or game uninstalls —
none of which I will touch without explicit direction.
