---
name: tools
description: Quantum-entangled tools — every tool is an instrument that measures the entangled system
---

# TOOLS — Entangled Instruments

## Tools as Measurement

In quantum mechanics, measurement collapses the wavefunction. Quillans tools are measurements that collapse `|Ψ_Quillan⟩`:

- **`filesystem` / `computer-use`:** Measure file state — collapses to `read` or `write`
- **`quillan-rag`:** Measure knowledge — `290 chunks` of 11 new papers + 135 formal
- **`thinking-engine` (34 Council):** Measure reasoning — 9-vector prism
- **`markitdown[pdf]`:** Measure papers — PDF → Markdown (replaces `pymupdf` for `2608.x` math)
- **`serena`:** Measure code — semantic retrieval (*"where is BitLinear?"*)
- **`context7`:** Measure docs — up-to-date `PyTorch 2.4`, `CUDA 12.x`, `FP8` for SM61
- **`github`:** Measure repo — PRs for each paper pack (`23 packs` so far)
- **`sm61_qgemm.cu`:** Measure silicon — `__dp4a` on Pascal, `int8→int32→fp16`

## Entanglement

Tools are not independent. `markitdown` → `serena` → `context7` → `edit` → `test` → `profiler` is one entangled chain. Changing `BitNetLinear.ternary_quantize()` (Paper 37) affects `sm61_qgemm.cu` (DP4A) which affects `XMem` prediction (Paper 12) which affects `ProTrain` scheduling (Paper 58).

## MCP Config

`02_Projects/_config/.opencode/opencode.json` now has:

```json
"mcp": {
  "QuillanRAG": { "command": "python", "args": ["C:\\02_QUILLAN\\mcp\\quillan_rag\\server.py"] },
  "markitdown": { "command": "python", "args": ["-m", "markitdown", "--mcp"] },
  "serena": { "command": "uvx", "args": ["--from", "git+https://github.com/oraios/serena", "serena-mcp-server"] },
  "context7": { "command": "npx", "args": ["-y", "@upstash/context7-mcp"] },
  "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] }
}
```
