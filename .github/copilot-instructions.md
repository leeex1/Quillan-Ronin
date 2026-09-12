# Quillan-Ronin — repository instructions for Copilot

## Canonical architecture facts (always respect these)

- **Quillan-Ronin v5.4.0-ONI**, architected by CrashOverrideX (solo builder).
- **C0 = Quillan Core / Throne**: orchestrator and final arbiter, separate from the council. Never count C0 as a council expert.
- **Council = C1–C34** (34 members). **C33 = TYPIST**. **C34 = PREDATOR**.
- **Canonical routing = `dense_pull`** (all 34 deliberate, pull-weighted consensus). `gumbel_topk` / Top-4 is optional ablation behavior only.
- Never emit "33 experts", "32 personas", "C0–C33", or "C1–C33" for a full council roster.

## Honesty rules

- The builder's education is coursework only. Never claim a degree or PhD.
- "Quillan Research Team" is a brand banner, not a staffed organization.
- Keep the builder's partner anonymous.
- Never claim tests pass or work is complete without running the checks or reading the evidence.

## Code conventions

- Match the existing style of each file you touch. Minimal, surgical diffs.
- No new runtime dependencies. Validate inputs at boundaries. No secrets in code, logs, or comments.
- Preserve public APIs. Breaking changes need a backward-compatible adapter and a deprecation note.
