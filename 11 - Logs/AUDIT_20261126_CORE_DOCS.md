# Core & knowledge-doc accuracy audit — 2026-11-26

Scope: `F:\02_QUILLAN` core and knowledge files. Seven Council auditors (C1, C7, C18, C19, C21, C24, C27) ran read-only passes; every finding below was re-verified against a file or command output. No model workstream file was claimed complete.

## Owner decisions applied

| Decision | Owner choice | Applied as |
|---|---|---|
| Root authority | F: runtime, C: mirror | Edits made on F:; exact bytes mirrored to C: and hash-verified |
| Swarm total | 9,520,000,000 everywhere | `AGENTS.md`, `memory.md`, README, flowchart, QICS/DQSO note |
| Parameter claims | measured targets | mini ~0.7B, flagship ~1B, Falcon/Dragon ~1B+; "~in-progress sizes (mini ~0.7B / flagship ~1B)" retired as unverified |
| README honesty | fix both | degree claim removed; "Research Team" relabeled brand banner, CrashOverrideX sole architect |
| Covenant | joint surgical acceptance only | Proposal written, covenant untouched |
| Duplicates | unique deployment variants, not full knowledge copies | Classification manifest written, no file moved or deleted |
| Tokenizer | check active models | All three vocabularies recorded as OPEN; no tokenizer locked |

## Confirmed defects and fixes

| Defect | Evidence | Fix |
|---|---|---|
| Council roster drift 33/32/18/C0–C33 | `TheRoninFlowState.md:120,124,191,542,616,758,805,857`; `9-Quillan Brain mapping.md:183`; `10- Quillan Persona Manifest.md:257,267,373`; covenant:258 | 33→34, C0 explicitly Throne not a seat; covenant left for proposal |
| `AGENTS.md` AQCS summed to 33 | formula line in AGENTS.md | now sums to 34 (landed in concurrent formula pass) |
| Swarm total 9B vs 9.52B in same file | `memory.md:31-32`; `AGENTS.md:37,85,577`; README (~20 sites) | exact 9,520,000,000; DQSO note now says 9.52e9 |
| Broken mermaid node references | flowchart `W1_Members`…`W5_Members` undefined; `Quillan_&_ACEFINAL`, `USC1_&…`, `H1_`, `_H12` | all 5 wave chains now aggregate 34 defined nodes; edge syntax fixed |
| Flowchart legacy counts | 96 personas, 34 experts, 50+ WoT, 10 stages, 4B params, 896k agents | 34 / 34 / 30 branches / 5 waves + 10-step process / in-progress sizes / 9.52B |
| Vector precedence unstated | flowchart 12-axis vs spec 9-vector | precedence note added to flowchart and `TheRoninFlowState.md` |
| Reference spec could not run on Windows cp1252 | `python reasoning_engine_oni.spec.py` exited 1 with `UnicodeEncodeError` on `✅` | ASCII `OK` in verification print; re-run passed |
| Spec claimed a gate it never implemented | `"Sandbox verification passed"` on AST parse only; `"council_consensus": True` hardcoded | renamed to AST whitelist check; consensus now `False` with `consensus_gate: not_implemented_in_reference_spec` |
| Spec claimed 12 layers but built 1 | `n_layer=12` unused | explicit single-layer reference-scope caveat added |
| Step 1000 "canonical base" claim void | `HANDOFF_20261002_dsh_fixes.md:30-41`: revisions diverged, best CPT weights destroyed twice | marked SUPERSEDED in `memory.md` |
| 73 subagents | `QUILLAN_OS_INTERCONNECTIVITY.md`: 87 partition | memory.md now says 87 = 1+34+35+17 |
| Stale model status | OS map said flagship fails J4 0/10; board #1269/#1271 say both smoke-tested, C18 limits scope | rewritten as IN PROGRESS, smoke test only, no completion claim |
| Four-donor fusion "in progress" | board #1384-1386: both no-training paths falsified | updated; zero-filled-experts defect marked stale via `donor_experts_aligned.report.json` |
| Tokenizer stated as fixed 50,257 | board #1232 rejection; deployed mini 151,936, flagship 128,256; custom 151,648 build | `LINEAGE.md` and OS map now say OPEN with all three facts |
| README false degree/team claims | `copilot-instructions.md:5,13,14` | corrected |
| Version drift | `LINEAGE.md` rules L21-26; v5.3.1 deprecated per CHANGELOG/MODEL_CARD | version section re-verified; 427M has no primary source, recorded as unverified |

## Verification results

- `python -m py_compile reasoning_engine_oni.spec.py` → exit 0.
- `python reasoning_engine_oni.spec.py` → exit 0 after ASCII fix.
- Pre-edit byte backup: `12 - Backups/audit_20261126_preedit/` + `MANIFEST.sha256` (16 files).
- C:/F: core-doc comparison before edits: 9,343 `.md` files identical, 0 differing.
- Agent structure audit: C: runtime roster 87 files = 1 + 34 + 35 + 17; all 35 swarms have correct parent/scope/272M/rank-32/top-k=19.

## Open, owner-gated items

1. Covenant C34 amendment — proposal at `01 - Core Architecture/COVENANT_AMENDMENT_PROPOSAL_20261126.md`; not applied.
2. Tokenizer lock — benchmark + donor compatibility still required.
3. Deployment-variant consolidation — see `11 - Logs/CONSOLIDATION_20261126.md`; no deletion performed.
4. `427M` figure has no primary source anywhere on disk; delete if owner confirms.

## Follow-up pass — full `02_QUILLAN` tree

- **Wiki links:** 459 broken `[[18-"Novelty Explorer" Agent.md]]` links repaired to the real filename (curly quotes U+201C/U+201D): 447 writable files plus 12 read-only `7-memories.md` copies (attributes cleared, rewritten, restored). The covenant's one link was deliberately reverted so the covenant stays byte-identical (`git diff` = 0 lines).
- **Stale numbers:** README (`$33$ Council Personas`, 33 cognitive domains, "34 councils, 9.52B agents"), `SOVEREIGN_STATE.md:26`, `CLAUDE.md`, and `.github/agents/quillan.agent.md` normalized to 34 council / 9.52B / 272M per swarm. Seven files mirrored to C: with 0 hash mismatches.
- **Consolidation executed (move only):** five byte-identical `Unholy Quillan.md` full copies moved from `06 - Deployment & Platforms/Platforms/<Perplexity,Grok,Mistral,Gemini,Claude>/` to `_archive/deployment_full_copies_20261126/<platform>/`. SHA256 verified before and after each move; legacy copies intact; nothing deleted. Deployment folders now contain only their platform-specific variants.

## Full-tree pass (owner scope: entire `02_QUILLAN` folder)

- **Full sweep:** 3,087 stale-pattern hits across the tree; archived tooling, backups, third-party dossiers and generated file indexes were deliberately excluded from edits.
- **Canonical corpus:** `3-Quillan(reality).md` (both copies), `Masterfile Quillan-Ronin.md` (canonical + knowledge-files + `full docs`) normalized to 34 council / 9.52B / 272M per swarm. Byte-preserving ISO-8859-1 round-trip was used because these files contain mojibake; no re-encoding occurred. UTF-8 en-dash variants of `C0–C33` were included (60 occurrences).
- **Formal papers (ours only):** `Formal Papers/README.md`, bibliography, cognitive-parliament paper, `paper_095`, `Quillan-Samurai.md`, announcement paper, `Quillan_Deliverables_v2.md`, `Quillan_Architecture_v7_0_Updated.md`, thinking template normalized. `quillan-ronin-structured-deliberation.md` has only `TBD` benchmark cells and `arXiv:2505.XXXXX`; no canonical numbers exist, so those were left untouched. Third-party dossiers were not edited.
- **Runtime config:** `.gemini/GEMINI.md` normalized (0 stale hits).
- **Covenant:** byte-identical to HEAD (`git diff` = 0 lines); no amendment applied. Tokenizer remains gated.
- **Verification:** 14 files mirrored to C: with 0 mismatches; targeted stale scan 0 for all edited formal papers, canonical reality docs, and Gemini config; only historical v5.2.2 version-history rows retain the word "34 council" because they describe that older release.

## Final gated decisions (owner, 2026-11-26)

- **Tokenizer LOCKED:** custom Qwen-based 151,648 (151643 base + 5 sovereign specials). `LINEAGE.md`, OS map and the reference spec's `VOCAB_SIZE` updated. Deployed mini 151,936 and flagship 128,256 are legacy until recompiled; 50,262 BPE rejected.
- **Covenant amended by joint acceptance:** C34-PREDATOR added to `council_roster.core_members` (index 33), `core_count: 34`, `total_capacity: 238000`. `IDENTITY.md` evolution log records the joint acceptance. Proposal file marked APPLIED.
- **Platform folders trimmed by published limits (move only, hash-verified):** GPT 20 (OpenAI GPT knowledge attachments), Claude 20 per chat (Anthropic), Gemini 10 per prompt (Google), Grok 20 per message (docs.x.ai; owner's earlier ~10 was approximate), Mistral 8 per chat (third-party, no official count published), Perplexity left at 11 (no published count), Open Source untouched. 257 generic files moved to `_archive/deployment_variant_trim_20261126/<platform>/`; platform-specific prompt/Samurai variants were swapped back into their folders. 0 hash failures. An earlier 321-file hash-based move was fully restored (0 failures) when the owner clarified that folder identity is the file count, not file uniqueness.

## Deployment bundles — final design (owner clarification)

The platform folders are intentionally **copy-ready bundles**: each holds the most useful Quillan knowledge files for that platform, with the platform's own prompt/Samurai variant. Folder identity is the file count, verified against published limits: GPT 20, Claude 20, Gemini 10, Grok 20, Mistral 8, Perplexity 10 (unpublished).

Each bundle now contains `9-Quillan Brain mapping.md`, `10- Quillan Persona Manifest.md`, `6-prime_covenant_codex.md` (except Perplexity, whose source pool never had it), the flow state, architecture flowchart, Novelty Explorer, and the platform variant. Priority scoring was corrected after a regex bug initially ranked `_Index.md` above the brain map; 106 files were then swapped with 0 hash failures. 257 non-bundle files remain archived under `_archive/deployment_variant_trim_20261126/<platform>/`. Nothing deleted.

## Status propagation (final pass, surgical)

Flagged before: README `fusion in progress` + two `Production-Ready` status lines; LINEAGE `Gate A verified`; two Formal Papers README status lines; one Quillan-Samurai model claim. After: status scan hits 0 in README, LINEAGE, OS map, Formal Papers README and Quillan-Samurai. Historical CHANGELOG v5.3.1 entries and OS map `v5.3.1 deprecated` were deliberately preserved; Quillan-Samurai's three code-packaging "production-ready" phrases were preserved because they describe generated artifacts, not model readiness.

## Full-tree numeric and version cleanup (final)

- Strict boundary-aware scans (excluding third-party extensions, archives, backups, generated indexes) were run across all folders including Formal Papers and Creative Works.
- Classified as non-issues: `9 billion` inside dataset tables (`WebText2 19 billion`), an external paper dossier's own text, the copilot rule file that intentionally lists forbidden strings, normalization scripts that intentionally map old strings, and this audit's own before-state evidence.
- Patched as true issues: `9.52B micro-agents` → 9.52B across system prompts; `Council of 34 personas` / `34 personas` → 34; `34 Experts` / `32-council` / `num_experts: int = 33` → 34; `272M per swarm|council node|persona` → 272M per swarm; `272M EGGROLL agents per swarm` → 272M; `Gate A verified` in legacy/knowledge-base LINEAGE rows; `Version 4.2.1 (Latest stable release)` → v5.4.0-oni; stale authorship and degree claims.
- A versioning note was inserted into every copy of `3-Quillan(reality).md`: v5.4.0-oni is the sole canonical counter; older version strings inside embedded examples are historical artifacts.

## All-file-type sweep

A full scan across 19 text extensions (`.md`, `.py`, `.json`, `.yaml`, `.html`, `.js`, etc.) found 1,911 active hits that Markdown-only passes had missed: `9.52 billion`/`9.520 billion` agents, `272M per swarm|council member`, `32 personas`, `34 personas`/`34 experts`, `C1-C34`, `in-progress sizes (mini ~0.7B / flagship ~1B)`, and `Quillan v5.3.1` claims in deployment configs, JSON manifests, HTML prompt text, and JS. 1,335 exact-phrase replacements were applied across 192 files; 295 minified bundles were skipped as generated artifacts and are classified separately.
