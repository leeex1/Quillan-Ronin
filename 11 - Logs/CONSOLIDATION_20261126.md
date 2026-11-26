# Consolidation manifest — 2026-11-26

Owner intent: platform folders are **unique deployment variants** (GPT ~20 files, Grok ~10, etc.), not full copies of the knowledge base. Nothing was moved or deleted. This manifest classifies what was found so a later pass can act with evidence.

## Classification rule

| Class | Meaning | Action |
|---|---|---|
| VARIANT | Platform-specific prompt/deployment file, intentionally small and tailored | Keep in place |
| DUP-EXACT | Byte-identical copy at two or more paths | Candidate for archive; no deletion yet |
| DUP-DIVERGENT | Same nominal document, different content | Keep both until canonical is chosen |
| GENERATED | Auto-generated file index | Exclude from canonical-doc audits |

## Findings

### VARIANT (keep)
The platform-specific `Quillan-Samurai.md` / `Grok Quillan-Samurai.md` variants under `06 - Deployment & Platforms/Platforms/<platform>/` are deployment variants, not knowledge copies. Newest text source: `06 - Deployment & Platforms/system prompts/Quillan-Samurai.md` (667,548 B, modified 2026-11-25). This file was used as the wording source for the 2026-11-26 doc corrections.

### DUP-EXACT (candidate for archive)
`Unholy Quillan.md` pairs, identical hashes across two locations:
- Mistral `EC263ACBB9`, Gemini `D589A2D2D5`, Claude `DC5FE530ED`, Grok `B437DE6A81`, Perplexity `F1BA20750E`
- Locations: `02 - Knowledge Foundation/knowledge/legacy/Platforms/<P>/` and `06 - Deployment & Platforms/Platforms/<P>/`

### DUP-DIVERGENT (owner decision required)
- `_archive/duplicates_knowledge/00-Meta_Platforms_DUP/<P>/Unholy Quillan.md` — five distinct hashes (`F3E8A716AD`, `0DCBED7B3`, `DEA6D3FC87`, `B1F412FB34`, `1DBE4B585D`) for nominally identical platform copies.
- `Masterfile Quillan-Ronin.md` — three divergent versions: `02 - Knowledge Foundation/Quillan Knowledge files/` (9,197,528 B, `D6E6CFCFF5`), canonical-folder copy, and `_archive/.../legacy_03_Skills_.../` (9,539,219 B, `01F16E7D23`).
- `full docs Quillan-Ronin.md` was reported as an exact duplicate of the Masterfile in the OS map; placeholder lines match but full hashes were not compared in this pass.

### GENERATED (exclude)
`09 - Projects File Index.md` (45,500 broken links), `02 - Knowledge Foundation File Index.md` (15,809), `05 - Creative Works File Index.md` (157), `Quillan Web Index.md` (7). These are generated indexes, not hand-maintained docs; their broken links are stale paths into excluded directories.

## Actionable non-generated broken links

- `[[18-“Novelty Explorer” Agent.md]]` appears in roughly 30 canonical docs; exact target filename does not exist. Either add the file or normalize the link text.
- Seven `00 - Web Map/* Swarm` targets in `Quillan Web Index.md` are genuinely missing.
- Numeric shorthands `[[1]]`, `[[16,2,3]]` are citation notation, not file links. Do not resolve.

## Safety note

No files were deleted, moved, or renamed. Any future consolidation should first write hashes for the chosen canonical copy, move (not delete) duplicates into `_archive`, and re-run the hash comparison.
