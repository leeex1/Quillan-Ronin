# knowledge/canonical — Canonical Knowledge Vault

## Rules (Details Matter)
- **Mermaid 100% executable:** Every ```mermaid block must render in GitHub/VSCode. No broken brackets, no `&` chains, unique subgraph IDs.
- **JS Pseudo:** ```js pseudo blocks are SPEC/FICTION — checked for internal consistency only, never executed or strict-linted.
- **Frontmatter required:** `file_type, file_id, domain, status, tags`

## What Lives Here
- `1-Quillan_architecture_flowchart.FIXED.md` — CORRECTED version (8 mermaid charts). Original saved as `.ORIGINAL.md`
- `1-Quillan_architecture_flowchart.ORIGINAL.md` — untouched copy for diff
- `8-Formulas.md` — 20 formulas + CMF flowchart (valid)
- `9-Quillan Brain mapping.md` — valid
- `10- Quillan Persona Manifest.md` — valid

## Differentiation
- Real executable Python → `src/quillan/` (see ../../src/quillan/README)
- Pseudo .py specs → stay here as `*.spec.py` with header `# PSEUDO — DO NOT EXECUTE`

## Validation
```bash
# Mermaid
npx @mermaid-js/mermaid-cli -i 1-Quillan_architecture_flowchart.FIXED.md -o /tmp/out.svg
# Python (real code only)
python -m py_compile ../../src/quillan/*.py
```

## Next Batch
- Copy remaining 30+ .md from `Quillan Knowledge files/` after detailed per-file audit (mermaid + pseudo check)
