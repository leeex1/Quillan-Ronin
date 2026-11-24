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

## Main repo published
Commit `f1d4e150` pushed successfully: `b4704236..f1d4e150 main -> main`.
Remote HEAD verified equal to local (`f1d4e15031`).

Scoped commit contents: 961 files — 64 renames (100% similarity), 19 new ledgers/reports,
625 intentional deletions (LFS-tracked binaries that belong on HF + deduplicated media),
253 modifications (PNG recompression, 2 path fixes, .gitignore). Other agents' runtime
state (07 - Memory, daemons, exports, training .py edits) was excluded.

**Final verification — all 7 canonical repos MATCH remote:**

| Repo | Local | Remote | Status |
|---|---|---|---|
| Quillan-Ronin | f1d4e15031 | f1d4e15031 | MATCH |
| Validation-test-kit- | eb017aed76 | eb017aed76 | MATCH |
| chrome-extension | 5c267fcc6b | 5c267fcc6b | MATCH |
| quillan.cpp | e6be6e4552 | e6be6e4552 | MATCH |
| metamon-game | 7819bd6a45 | 7819bd6a45 | MATCH |
| MCP-DevKit | 49ed09276d | 49ed09276d | MATCH |
| muse-open-kernal-research- | 85d75d366c | 85d75d366c | MATCH |

Note: `worker\extension` still holds 1 unique local commit (3 behind, 1 ahead) — preserved,
not overwritten. Quillan-ide upstream is empty (0 KB).
