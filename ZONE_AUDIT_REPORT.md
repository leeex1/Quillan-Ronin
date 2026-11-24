# Per-Zone Duplicate Audit — 2026-11-23 (post-dedup)

After the verified deletion of 250 true duplicates (2.89 GB), the following duplicate set remains.
All figures are from full SHA256 hash groups, with deleted originals excluded.

| Zone | Groups | Redundant copies | Wasted MB | Pages copies in groups | Risk tier |
|---|---:|---:|---:|---:|---|
| `05 - Creative Works\assets` | 156 | 822 | 7,068.2 | 275 | REVIEW — mixed with Pages copies |
| `docs` (GitHub Pages) | 450 | 1,622 | 2,036.7 | 1,540 | BLOCKED/CAUTION — site-served |
| `05 - Creative Works` (other) | 69 | 131 | 118.3 | 0 | REVIEWABLE |
| `05 - Creative Works\docs` | 4 | 8 | 84.5 | 12 | CAUTION — Pages-style |
| `05 - Creative Works\Audio Engineer` | 83 | 87 | 23.0 | 0 | REVIEWABLE |
| `_archive` | 76 | 156 | 18.5 | 0 | REVIEWABLE |

## Interpretation

- **`05CW\assets` (7.0 GB)** — largest remaining. Copies span `assets\06_Media\audio\`,
  `Audio Engineer\Audio files\audio\`, and `docs\audio\`. Some are Pages copies (275 paths) that
  must be retained. Others are cross-directory duplicates that may represent an intentional
  "canonical asset library" vs "working copy" split — **needs your decision on which directory is canonical**.
- **`docs` (2.0 GB)** — 1,540 of 1,622 redundant copies are themselves Pages copies. Removing any
  requires first proving no HTML references that exact filename. **Not actionable without a per-file reference scan.**
- **`05CW\other`, `Audio Engineer`, `_archive`** — small, mostly cross-directory duplicates with no
  Pages exposure. Lowest risk, but the same "subset for deployment" concern you raised applies.

## Already completed (verified)

- 250 byte-identical duplicates proven by full SHA256, each with ≥1 retained survivor.
- Deleted with pre-deletion hash ledger `DELETION_LEDGER_20261123.json`; 2.89 GB freed; 0 failures.
- Canonical `exports\quillan-dragon-math40-headrow.safetensors` retained and verified.
- No platform/deployment subset folders (`grok`, `gpt`, `mistral`, etc.) were in scope.

## Standing rule

No further deletion or move until you approve the specific zone.
