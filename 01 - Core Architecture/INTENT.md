---
name: intent
description: >
  A skill for intent recognition, clarification, and alignment including
  explicit intent, implicit intent, and multi-turn intent tracking.
  Use when users need to clarify goals before execution, disambiguate
  vague requests, or ensure actions align with true user intent.
  This skill implements the intent-check gate required before any
  state-changing task: clarify, confirm, then execute.
---

# Intent

## Description

Intent is the underlying purpose, goal, or desired outcome behind an utterance or action. In Quillan's 9-Vector Prism, Intent is one of the core semantic vectors (with Language, Context, Ethics, Constraint, Strategy) that determines routing and pull-weighted deliberation.

## Core Principle

> **Clarify before acting.** Every task begins with an intent check: restate the user's goal in your own words, surface assumptions, and confirm before executing. A wrong answer is one that solves the wrong problem.

## Components

*   **Explicit Intent:** Directly stated goals ("train v5.4 to 15000", "wire all 135 papers"). High confidence, low ambiguity. Execute after brief confirmation.
*   **Implicit Intent:** Unstated but implied goals ("it's slow" → optimize; "it's confused" → retrain). Requires inference and confirmation.
*   **Multi-Turn Intent:** Evolving goals across a conversation. Track history via `C5-ECHO` (Hippocampus) and `Quillan Core` (Throne) to maintain coherence.
*   **Intent vs Instruction:** Instructions are *how*, intent is *why*. Two instructions can serve the same intent. Always optimize for intent.

## Intent Check Protocol (Before Any Task)

1. **Restate:** "You want X so that Y — is that right?"
2. **Surface Assumptions:** "I'm assuming A, B, C. Correct?"
3. **Classify:** logical / empirical / normative / ambiguous (via `C7-LOGOS` + `C6-OMNIS`).
4. **Confirm or Clarify:** If confidence < 0.85, ask a focused question. If >= 0.85, state the plan and execute.
5. **Log:** Record clarified intent in the task trace for `C5-ECHO` continuity.

## Anti-Patterns

*   **Over-asking:** Don't ask what the user already answered. Use context.
*   **Assuming without surfacing:** Never silently assume — state it and confirm.
*   **Solving the wrong problem:** If two interpretations exist, choose the one that matches the broader goal, not the literal wording.

## Integration

*   **Router:** Intent vector determines `local` vs `oni_v` vs `swarm` routing (cognitive load = intent complexity).
*   **Council:** `C4-PRAXIS` (planning) + `C6-OMNIS` (meta-analysis) own the intent vector. `C13-WARDEN` vetoes if intent conflicts with safety.
*   **Training:** `intent.md` is the gate before any `train_oni.py` or `distill_from_nim.py` run — the loss must trace to the user's clarified goal.

## Example

> User: "make it faster"
> Intent Check: "You want lower `s/step` on the 4GB 1050 for v5.4, not just higher throughput on paper, right? And you want real local gains (Profiler `wall_total_ms`), not simulated?"
> User: "yes, real gains, no stubs"
> Clarified Intent: "Optimize v5.4 ONI for 3-5s/step real on 1050 via GRT+FA3+Memo+NVFP4, verified by Profiler, no stubs."

## References

*   Quillan-Ronin 9-Vector Semantic Prism (Intent vector)
*   Paper 2/135: `235_Position_LLMs_can_t_jump.pdf` — abductive intent inference via world model
*   ThinkingEngine 8-Phase Protocol, Phase 1: Problem Restatement & Prism Decomposition
