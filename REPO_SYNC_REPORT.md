# GitHub Repo Sync Report — 2026-11-23

## Canonical repos (user-provided list, minus Quillan IDE)
All verified against `git ls-remote` / `gh api`.

| Repo | Remote HEAD | Local mirror | Local HEAD | Status |
|---|---|---|---|---|
| Quillan-Ronin | b4704236 | `C:\02_QUILLAN` | b4704236 | synced; **1,121 uncommitted changes** |
| Validation-test-kit- | eb017aed | `09 - Projects\projects\Validation-test-kit` | eb017aed | synced ✓ |
| Quillan-Ronin-chrome-extension | 5c267fcc | `09 - Projects\projects\extension` | 5c267fcc | **pulled** 97dc5b0 → 5c267fcc ✓ |
| quillan.cpp | e6be6e45 | `09 - Projects\projects\Quillan.cpp` | e6be6e45 | synced ✓ |
| metamon-game | 7819bd6a | `09 - Projects\projects\metamon-game` | 7819bd6a | **cloned fresh** ✓ |
| MCP-DevKit | 49ed0927 | `09 - Projects\projects\MCP-DevKit` | 49ed0927 | synced ✓ |
| muse-open-kernal-research- | 85d75d36 | `09 - Projects\projects\muse-open-kernal-research-` | 85d75d36 | **cloned fresh** ✓ |

Additional mirrors: `testing\Quillan-Ronin-chrome-extension` (5c267fcc, clean),
`testing\Validation-test-kit` (eb017aed, 39 dirty), `worker\extension`
(07f72f8, 3 behind + 1 unique local commit — preserved, not overwritten).

Quillan-ide upstream is **empty (0 KB)**; the local clone has detached HEAD and no commit.

## Safety checks
- 7 ledgers checked: **0 live subrepo paths touched** by our moves/deletions
- 988 ledger hits were inside the E: duplicate quarantine only
- 247 LFS-tracked files still on disk are intentionally gitignored (`*.pdf`, `*.pkl`, `*.glb`, build dirs) — they remain local; media lives on Hugging Face by design

## Outstanding decision
The main repo's new file structure (84 organized moves, media dedup, LFS pointer repair,
PNG recompression, reports) is **local and uncommitted**. Pushing requires selecting only
our changes and excluding other agents' active work (modified training .py files, runtime
JSONs, daemon state, exports).
