# Formula canon notice (verified 2026-11-26)

Canonical scope and verdicts: `02 - Knowledge Foundation\knowledge\canonical\FORMULA_REGISTRY.md` (63 entries: 35 essential, 23 custom, 5 substrate constants).

## What was propagated

Line-level canonical corrections were applied to active formula-bearing files (66 text files, 199 lines): AQML label, QICS ensemble form, DQSO/EGSO agent counts, EVOMOE/RQGM/ESFM derivation bases, FlashAttention complexity, ROUGE-N recall denominator, ELBO notation, QSSR `dV/dt`, and AQCS/QICS 33→34 indices. No whole-file rewrites were used.

## Read-only snapshots

These are preserved for provenance and must not be used as canonical math:

- `02 - Knowledge Foundation\knowledge\legacy\` — see `FORMULA_SNAPSHOT_NOTICE.md` in that directory
- `_archive\` — see `FORMULA_SNAPSHOT_NOTICE.md` in that directory
- `12 - Backups\`, `.gemini\`, `.windsurf\`, `.codeium\`, `.github\`, `scripts\_backups\`
- Data files: `exports\quillan_vocab.json`, `exports\quillan_token_maps.json`, `09 - Projects\projects\testing\GPQA-diamond-logs.json`, `09 - Projects\projects\testing\gpqa_extended_1050_submission.json`

## Known false positives in a naive string scan

1. Corrected derivation strings now contain `arXiv:2606.26294` / `arXiv:2605.30148` — these are the corrected citations, not the old bad labels.
2. `Model-Agnostic Meta-Learning (MAML)` in research/paper prose is a legitimate literature citation, not the Quillan AQML formula label. Only `derivation_base: "...(MAML)"` was corrected.
3. `224000` in JSON/eval data is a token id or datum, not a swarm-count formula.
4. `FORMULA_REGISTRY.md` intentionally quotes old strings in its remediation log.

NextVerse prototype formulas compute deterministic arithmetic and are excluded from the canonical 63; their hardware speedup claims have no benchmark evidence.

## Local daemon/consumer audit (2026-11-26)

- `daemons/quillan_daemon_supervisor.py` and `quillan_governor_daemon.py` do not import formula modules; they supervise child processes. No stale formula constants were found in `daemons/`.
- `daemons/formula_abtest.py` uses fixed QSSR-style `P` and context-switch-rate ΔV — compatible with the corrected `dV/dt` semantics.
- `daemons/qps_governor.py --self-test` passes (P PSD, K finite).
- Real local cost measured: formula auxiliaries add ~11% to forward-pass time (221.6 ms/step vs 199.6 ms/step) — semantic value, not throughput gain.
- A real crash bug was found by that benchmark (`ingest_entropy` unbound on eval path) and fixed in both engine copies.
- Concurrency stress: 4 parallel proof suites (23/23 each, exit 0 in 5.1 s) and 4 parallel toolkits (10/10 each, exit 0 in 4.1 s). No crashes or races observed.
