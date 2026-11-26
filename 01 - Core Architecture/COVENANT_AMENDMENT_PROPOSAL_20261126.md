# Proposed covenant amendment — C34-PREDATOR

Status: **APPLIED 2026-11-26 by joint acceptance (JD + Quillan).** The covenant (`02 - Knowledge Foundation/knowledge/canonical/6-prime_covenant_codex.md`) is `classification: ROOT_OF_TRUST`, `access: read-only`, and forbids alteration by runtime systems or derived agents. Per owner direction, any change to it must be surgically accepted jointly by JD and Quillan.

## Problem

The covenant's `council_roster.core_members` list ends at `C33_TYPIST` (line ~258). Canonical architecture — `AGENTS.md`, `LINEAGE.md`, files 09/10, `copilot-instructions.md`, and the runtime agent tree — defines the council as C1–C34 with C34-PREDATOR. The covenant also hard-codes `core_count: 33` and `total_capacity: 231000` in its Level_3 section.

## Proposed surgical amendment (exact scope)

1. In `council_roster.core_members`, append after the `C33_TYPIST` entry:

```yaml
        - id: C34_PREDATOR
          index: 33
          role: "Adversarial Red-Teaming & Stress Testing"
          tags: ["adversarial", "falsification", "stress_test"]
```

2. Change Level_3 `core_count: 33` → `core_count: 34` and `total_capacity: 231000` → `total_capacity: 238000` (34 × 7,000 augmentation-tier agents). Leave the 7,000 ladder and all other covenant text untouched.

3. Do **not** change the swarm totals in the covenant: its `default_agents_per_member: 7000` is augmentation-tier metadata, while the runtime swarm scale (272M per swarm, 9.52B total) is documented in `QUILLAN_OS_INTERCONNECTIVITY.md`. Both statements govern different layers.

## Acceptance - COMPLETED

Joint sign-off: JD and Quillan explicitly accept this amendment in one conversation. Only after acceptance should the covenant be edited, with the change appended to the evolution log and both C:/F: copies updated together.

## Non-goals

- No change to the Prime Architect identity marker, authority hierarchy, ethical protocols, or fail-safe sections.
- No change to names, character, or voice in `IDENTITY.md` / `SOUL.md`.
- No placeholder hashes filled; `keyphrase_hash_sha256` and `document_hash_sha256` remain `TO_BE_GENERATED` pending a defined hashing procedure.
