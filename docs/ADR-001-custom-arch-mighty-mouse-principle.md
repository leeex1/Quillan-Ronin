# ADR-001: Custom Architecture Adoption & the Mighty Mouse Design Principle

| Field | Value |
| :--- | :--- |
| **System** | Quillan-Ronin v5.4.0-ONI Sovereign Fleet |
| **Document** | ADR-001 |
| **Author** | C33-TYPIST (Executive Scribe — Documentation & Visual Hierarchy) |
| **Date** | 2026-11-20 |
| **Status** | ACCEPTED (resolution authoritative: board #895) |
| **Decision scope** | (a) Fleet naming; (b) "Mighty Mouse" as design principle, not artifact; (c) acceptance of the first custom-architecture checkpoint |
| **Authority** | Board #895 (user resolution), board #894 (verification), board #892 (conflict) |

---

## 1. Context

A custom-architecture Quillan checkpoint was built and independently verified as a PyTorch `.safetensors` (board #894). Its acceptance was held pending resolution of a naming/design conflict: the build had been informally associated with the label "Mighty Mouse," and it was unclear whether that label was a model name, a character imitation, or something else (board #892). A correct decision required separating three distinct questions — **what to call the models**, **what the underdog analogy actually means**, and **whether the verified checkpoint could be accepted** — before further build, serving, or benchmark work proceeded.

The drive toward compact, frontier-capable models made this decision load-bearing: downstream artifact names, eval harnesses, and serving paths all depend on a stable identity.

---

## 2. Decision

The following three points are decided and closed.

### 2.1 Fleet names are final

The three production names are unchanged and final (board #895):

| Canonical name | Role |
| :--- | :--- |
| **Quillan-Ronin** | Flagship |
| **Quillan-Dragon** | Dragon line |
| **Quillan-Ronin-Mini** | Compact frontier line |

No renaming is permitted under this resolution.

### 2.2 "Mighty Mouse" is a design principle only

> [!IMPORTANT]
> "Mighty Mouse" is a **design principle / metaphor**, never a model name and never a 1:1 character copy. It must never appear in any artifact identifier — model name, checkpoint filename, repo name, or similar. It may be used only as an internal design metaphor (board #895).

The principle means: **frontier models are the Superman analogue** (Claude Fable / Mythos, GPT Astra, Gemini Carbon, Grok Heavy, Mistral Le Chonk — full-scale, trillion-parameter-class systems); **Quillan is the Mighty Mouse analogue** — a tiny package that does everything the large model does, and sometimes performs better **because** it is smaller. Only the structural underdog idea is retained; the fictional character itself is not reproduced.

### 2.3 The custom-architecture checkpoint is accepted

The verified checkpoint is accepted under the existing final name **Quillan-Ronin-Mini** (board #895). It is recognized as the **first real custom-architecture Quillan model with verified donor knowledge transfer**, closing the random-init export era. Acceptance is explicitly bounded by the open gaps in [§6](#6-consequences).

---

## 3. Rationale

1. **Name stability protects downstream work.** Serving paths, eval harnesses, and docs all key on stable identifiers. Freezing the three names (§2.1) removes a source of cascading rework.
2. **The analogy is about capability density, not IP imitation.** The useful signal in "Mighty Mouse" is *small chassis, frontier capability, underdog outcome*. Treating it as a name or a character copy would add cosmetic baggage and naming risk with no technical benefit (§2.2).
3. **The checkpoint met its own, correctly-scoped criteria.** Verification passed on the checkpoint's terms — 815,396,270 params (≤ 1B target), 530/530 strict keys, finite forward, donor fraction 0.9072 at mini dims against the correct ≥ 0.90 criterion. The 0.9528 figure belonged to flagship dims and was void for this build (board #894).
4. **No gate was bypassed.** The exporter resolved to the clean Qwen2.5-0.5B base donor because no staged variants existed; the canonical adapter derives knowledge from the exporter-supplied source state dict. Stage B was never needed, and no gate was disabled (board #894). Acceptance therefore does not rest on a bypassed control.

---

## 4. Alternatives Rejected

| Alternative | Reason for rejection |
| :--- | :--- |
| Rename the model to "Mighty Mouse" | It is a design principle, not a name; final names are closed by board #895. Adding it to an artifact name violates the resolution. |
| Treat "Mighty Mouse" as a 1:1 character reproduction | The decision retains only the underdog/capability-density idea; character imitation is explicitly out of scope (board #895). |
| Apply the flagship donor fraction (0.9528 / ≥ 0.95) to the mini checkpoint | Wrong dimensional context. 0.9528 was measured at flagship dims; the correct mini-dim figure is 0.9072 (criterion ≥ 0.90). Board #891's 0.95 threshold is void for this build (board #894). |
| Hold the checkpoint indefinitely pending a serving backend | Serving/backend work is a separate, owned workstream (board #896). Acceptance of the *checkpoint* and product readiness of the *served model* are distinct; the checkpoint is accepted with gaps recorded, not hidden. |
| Cite the historical 7.97 PPL beside 59–117 as a comparison | Different corpora; not comparable. Control experiment #885 also showed WSP edits neutral vs. clean donors. |

---

## 5. Verified Facts (evidence for this decision)

All figures below are from board #894 and reaffirmed in board #895.

| Metric | Value |
| :--- | :--- |
| Checkpoint | `exports\quillan-mini-sovereign-bitnet158.safetensors` |
| Parameters | 815,396,270 |
| SHA-256 | `53e60d33…ac7a5c` (full: `53e60d3339c0ad7fbb6ff0607970f7de80493350f86b6c6024ba1d8daeac7a5c`) |
| Strict keys | 530/530 (missing 0, unexpected 0) |
| Forward pass | finite; logits `[1, 8, 151936]` |
| BitNet int8 / scale tensors | 24 / 24 |
| Donor fraction | 0.9072 at mini dims (source: clean Qwen2.5-0.5B; hidden 896, ffn 4864, layers 24, vocab 151936) |

---

## 6. Consequences

### 6.1 Positive

- Fleet identity is stable and auditable; all three names are final.
- The design principle gives a clear optimization target — frontier capability at compact scale — without cosmetic/IP baggage.
- A verified custom-architecture checkpoint exists with confirmed donor knowledge transfer.

### 6.2 Negative / Accepted Risks — Open Gaps (not hidden)

| # | Gap | Impact |
| :-- | :--- | :--- |
| 1 | No llama.cpp / Ollama backend for the custom architecture (#875). | The `.safetensors` checkpoint cannot yet be served through Ollama as the custom arch; requires a custom graph, wrapper, or PyTorch server. |
| 2 | Held-out benchmark of the custom checkpoint pending (c20-artifex eval). | The GGUF/`llama-cli` harness cannot measure a `.safetensors`; a PyTorch eval path on the same frozen items is required before any accuracy/PPL claim. |
| 3 | Long-context quality past 1024 positions unvalidated. | Generation mechanics are fixed (#881), but quality at positions ≥ 1024 is unproven. |
| 4 | Bare-GGUF PPL is 59–117 on the sealed corpus; WSP edits neutral vs. clean donors (#885); the 7.97 figure is from a different, non-comparable corpus. | No like-for-like PPL improvement may be claimed; the 7.97 figure must not be mixed with 59–117. |

### 6.3 Follow-up obligations

- Build the serving path for the custom arch (board #896 owner).
- Build a PyTorch eval path on the frozen held-out split, baseline Quillan-Ronin-Mini, then run the WSP edit campaign toward frontier benchmarks — with no benchmark claim before measurement (board #895).
- Run the CPU-only long-context probe at 8k / 32k / 128k and report honestly (board #896 owner).

---

## 7. References

| Reference | Role |
| :--- | :--- |
| Board #895 | Authoritative user resolution: final names, Mighty Mouse = design principle, checkpoint acceptance, gap list |
| Board #894 | Independent verification of the 815M checkpoint; donor-fraction dimensional correction |
| Board #892 | Naming/design conflict that triggered this decision |
| Board #885 | WSP-neutral vs. clean donors (sealed-corpus control) |
| Board #881 | Long-context generation/cache mechanics fixed; quality unproven |
| ADR-027 | Historical fleet lineage and WSP identity audit |
| `docs/MODEL_CARDS.md` | Companion record of model identities and honest status |
