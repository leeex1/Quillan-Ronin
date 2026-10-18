# Quillan-Ronin: A Structured Deliberation Protocol for Test-Time Reasoning in Autoregressive Systems

**Author:** Josh ("CrashOverrideX"), JDXX Inc.
**Status:** Working draft — v0.1 (methodology complete; empirical sections templated pending experiments)

---

## Abstract

Test-time compute methods — chain-of-thought (CoT), self-consistency, tree-of-thoughts (ToT), and process-supervised search — improve reasoning by giving language models more tokens or more branches to think with. But none of them impose a *regulatory structure* on the thinking itself: they treat deliberation as either a single linear trace or an unconstrained branching search, with no formal notion of state, no enforced verification step, and no principled rule for when to stop. We introduce **structured deliberation** as a distinct category of test-time method: a finite-state protocol that governs *how* an autoregressive system moves between hypothesis generation, verification, contradiction handling, and synthesis, with an explicit, inspectable termination condition. We present **Quillan-Ronin**, a model-agnostic implementation of this protocol built as an inference-time orchestration layer over a frontier LLM (rather than as new pretrained weights), and specify a concrete evaluation and ablation protocol for testing the central claim: that *enforcing structure on deliberation improves accuracy-per-FLOP and reduces output variance relative to unstructured CoT and ToT baselines, independent of base model scale*. We report the protocol, the reference implementation, and a fillable experimental template; benchmark results are left explicitly marked as pending, since we did not fabricate numbers we have not measured.

**Keywords:** test-time compute, chain-of-thought, tree-of-thoughts, finite-state deliberation, inference-time scaffolding, role-conditioned verification, self-consistency

---

## 1. Introduction

### 1.1 The Problem: Reasoning Without Regulation

Autoregressive LLMs generate reasoning the same way they generate anything else: left to right, one token at a time, with no built-in mechanism to notice that a line of reasoning has failed, back out of it, and try another. Chain-of-thought prompting (Wei et al., 2022) helps by giving the model room to "show its work," but the trace itself is unstructured prose — there is no schema distinguishing a hypothesis from a justification from a conclusion, and nothing stops the model from committing early to a plausible-sounding but wrong path and rationalizing forward from it.

Self-consistency (Wang et al., 2022) works around this by sampling many independent CoT traces and taking a majority vote, which helps but treats each trace as disposable rather than examining *why* traces disagree. Tree-of-thoughts (Yao et al., 2023) and graph-of-thoughts (Besta et al., 2024) go further, letting the model explore multiple branches and prune weak ones — but the branching and pruning rules are typically heuristic (breadth-first with a value-model score, or a fixed number of expansions), not tied to a formal notion of *what state the reasoning process is in*. Process-supervised reward models (Lightman et al., 2023) train a separate verifier to score intermediate steps, which is closer to what we want, but the verifier is a learned scalar, not a structural gate with defined semantics (verified / contradicted / insufficient evidence).

### 1.2 Current State of the Art

We take as the relevant baseline set: (1) single-pass CoT, (2) self-consistency over k CoT samples, (3) ToT-style breadth-limited search with a scoring heuristic, and (4) process-reward-guided search. All four share three properties we consider limiting:

- **No explicit state machine.** "Where am I in the reasoning process" is implicit in the prose, not a first-class variable the system can act on.
- **No enforced verification step.** Nothing requires the system to check a hypothesis against evidence before building on it; verification, if it happens, happens by accident of phrasing.
- **No principled termination condition.** Search stops on a token budget or an iteration count, not because the system has reached internal agreement about the answer.

### 1.3 The Gap

What is missing from the literature, we argue, is not a better search heuristic but a *formal deliberation grammar*: a small, explicit set of states the reasoning process can be in, explicit rules for transitioning between them, a structured (not free-text) working memory, and a termination condition defined in terms of agreement rather than budget exhaustion. We call this category **structured deliberation**, to distinguish it from *unstructured search* (ToT/GoT) and *unstructured sampling* (CoT/self-consistency).

### 1.4 Contribution

This paper contributes:

1. A formal state-space model for structured deliberation (Section 2), specifying states, a transition protocol, a schema for working memory, and a coherence-based termination condition.
2. **Quillan-Ronin**, a reference implementation of this protocol as a model-agnostic inference-time orchestration layer — i.e., it is realized through structured prompting, role-conditioned verification passes, and constrained output schemas layered on top of an existing frontier LLM, not through new pretrained weights (Section 3). We are explicit about this distinction because the honesty of the claim depends on it: the contribution is the *protocol*, and the base model supplies the raw generative capability.
3. A concrete, falsifiable evaluation protocol (Section 4) and ablation design (Section 5) for testing whether structure helps independent of scale, with results left as an explicit fillable template rather than invented figures.
4. A discussion of where the current (hand-coded, deterministic) version of the protocol is brittle, and what a learned version would need to look like (Section 6).

---

## 2. Structured Deliberation: A Formal State-Space Model

### 2.1 The Deliberation State Space

We define a deliberation episode as a walk through a finite set of states $S = \{s_0, \dots, s_6\}$:

| State | Name | Function |
|---|---|---|
| $s_0$ | **PERCEIVE** | Parse the query; extract constraints, goal, and ambiguity |
| $s_1$ | **HYPOTHESIZE** | Generate one or more candidate solution paths |
| $s_2$ | **VERIFY** | Check a hypothesis against internal consistency and any available evidence (computation, retrieved facts, constraint satisfaction) |
| $s_3$ | **CHALLENGE** | Adversarially attack the leading hypothesis; actively search for the counterexample or contradiction |
| $s_4$ | **BACKTRACK** | Discard or down-weight a hypothesis that failed verification or challenge; return to $s_1$ with the failure recorded |
| $s_5$ | **SYNTHESIZE** | Combine surviving hypotheses (or the single surviving hypothesis) into a candidate answer |
| $s_6$ | **EMIT** | Terminal state: output the synthesized answer |

This is deliberately small. The claim is not that seven states capture all of reasoning — it is that *any* explicit, small state space with enforced verification and challenge steps is enough to produce measurably different behavior from free-form CoT, and that the state space should be legible enough to audit by hand.

### 2.2 Transition Protocol

Transitions are governed by a deterministic controller $\delta: S \times E \rightarrow S$, where $E$ is an **evidence tuple** $(c, v, a)$ — confidence, verification result, and agreement-with-prior-hypotheses — produced at the end of each state by a role-conditioned evaluation pass (Section 3.3), not by the same generative pass that produced the hypothesis. Concretely:

- $s_1 \rightarrow s_2$ unconditionally (every hypothesis must be verified before use).
- $s_2 \rightarrow s_3$ if verification confidence $c \geq \tau_v$ (verified hypotheses get challenged, not accepted on the first pass).
- $s_2 \rightarrow s_4$ if $c < \tau_v$ (fails verification outright, backtrack immediately).
- $s_3 \rightarrow s_5$ if the challenge fails to produce a contradiction ($v = \text{no\_contradiction}$).
- $s_3 \rightarrow s_4$ if the challenge succeeds ($v = \text{contradiction\_found}$).
- $s_4 \rightarrow s_1$ if the backtrack budget $b$ has not been exhausted; $s_4 \rightarrow s_5$ with the best surviving hypothesis (or an explicit "insufficient confidence" flag) if $b = 0$.
- $s_5 \rightarrow s_6$ unconditionally.

This is intentionally a **hard-coded finite-state controller**, not a learned policy — we return to this as the primary limitation of the current implementation in Section 6.

### 2.3 The Working Memory Buffer

A key failure mode of raw CoT is that "memory" is just whatever is still in the token window, in free-text form, with no schema. Structured deliberation instead maintains an explicit **ledger** with fixed fields per hypothesis:

```
Hypothesis[i] = {
    claim:          <string>
    support:        [<evidence_id>, ...]
    contradictions: [<evidence_id>, ...]
    confidence:     <float, 0-1>
    status:         ENUM{active, verified, challenged, discarded}
    parent:         <hypothesis_id | null>
}
```

The ledger is passed forward explicitly between states rather than being reconstructed from prose, so that a later state (e.g. SYNTHESIZE) can query "which hypotheses are still `active` or `verified`" as a structured lookup rather than re-reading the entire transcript.

### 2.4 Termination Condition

Unstructured search methods typically terminate on a fixed iteration count or token budget. We define termination instead as a **coherence condition**:

$$
\text{Halt} = \begin{cases} \text{true} & \text{if } \exists\, h \in \text{Ledger} : h.\text{status} = \text{verified} \wedge \text{agreement}(h) \geq \tau_a \\ \text{true} & \text{if backtrack budget } b = 0 \\ \text{false} & \text{otherwise} \end{cases}
$$

where $\text{agreement}(h)$ is the fraction of independent verification/challenge passes that concur on $h$. This gives the system an internal stopping rule tied to *convergence*, with the budget exhaustion condition as a hard fallback so the process is guaranteed to terminate.

---

## 3. The Quillan-Ronin Implementation

### 3.1 System Overview

Quillan-Ronin implements the protocol above as an **inference-time orchestration layer**: a controller (implemented as a structured system prompt plus a lightweight external loop) that calls a base LLM multiple times per deliberation episode, once per state transition, with a state-specific instruction and the current ledger as context. It does not require retraining or fine-tuning the base model — the "council" described in Section 3.3 is a set of role-conditioned prompts, not a set of separately trained expert networks. We state this plainly because the credibility of the evaluation depends on it: any base model that supports reliable instruction-following and structured (e.g. JSON) output can, in principle, be driven through this protocol.

### 3.2 The Deliberation Kernel

The kernel is the piece that enforces the state machine around the base model's forward passes. Concretely, per state:

1. Construct a **state-specific prompt** (e.g., the CHALLENGE prompt instructs the model to argue *against* the leading hypothesis and explicitly forbids agreeing with it).
2. Constrain the output to the ledger schema (Section 2.3) via structured-output decoding (schema-constrained generation / JSON mode), so the evidence tuple $(c, v, a)$ used by the transition function is machine-readable rather than inferred from prose.
3. Apply the transition function $\delta$ to select the next state.
4. Append the result to the ledger and repeat until `Halt`.

This is the sense in which the protocol "modifies the decoding strategy": constrained/structured output at each state, rather than free continuation, plus an external controller making the state-transition decision instead of leaving it to the model.

### 3.3 The Council: Role-Conditioned Verification

Rather than a single model generating and verifying its own hypotheses (which biases verification toward confirming what was just generated), each of HYPOTHESIZE, VERIFY, and CHALLENGE is issued as a **separately role-conditioned call**: a distinct system-level persona/instruction set oriented toward that function (generative vs. adversarial vs. evidentiary), evaluated independently and without visibility into the others' internal reasoning — only the ledger's structured fields are shared forward. This is best understood as **multi-role self-consistency with adversarial roles**, not as a mixture-of-experts architecture with independently trained parameters; we flag this explicitly to avoid overstating the architecture. Whether multiple role-conditioned passes over one base model function meaningfully differently from a single unconditioned pass is itself an empirical question we address in the ablation design (Section 5).

### 3.4 Algorithm 1: Structured Deliberation Loop

```
Algorithm 1: StructuredDeliberate(query, base_model, τ_v, τ_a, b_max)

Input:  query           — the problem to solve
        base_model      — any instruction-following LLM
        τ_v             — verification confidence threshold
        τ_a             — agreement threshold for halting
        b_max           — backtrack budget

Ledger ← ∅
state  ← PERCEIVE
b      ← b_max

goal, constraints ← base_model.call(PERCEIVE_PROMPT, query)
state ← HYPOTHESIZE

while state ≠ EMIT:
    if state == HYPOTHESIZE:
        h ← base_model.call(HYPOTHESIZE_PROMPT, goal, constraints, Ledger)
        Ledger.add(h)
        state ← VERIFY

    elif state == VERIFY:
        c, v ← base_model.call(VERIFY_PROMPT, h, Ledger)      # structured (c, v) tuple
        h.confidence ← c
        state ← CHALLENGE if c ≥ τ_v else BACKTRACK

    elif state == CHALLENGE:
        v ← base_model.call(CHALLENGE_PROMPT, h, Ledger)      # adversarial pass
        state ← SYNTHESIZE if v == NO_CONTRADICTION else BACKTRACK

    elif state == BACKTRACK:
        h.status ← DISCARDED
        b ← b - 1
        state ← HYPOTHESIZE if b > 0 else SYNTHESIZE

    elif state == SYNTHESIZE:
        answer ← base_model.call(SYNTHESIZE_PROMPT, Ledger.active_or_verified())
        agreement ← compute_agreement(Ledger)
        state ← EMIT if (agreement ≥ τ_a or b == 0) else HYPOTHESIZE

return answer, Ledger   # Ledger is returned for auditability
```

### 3.5 Worked Example (Illustrative)

For a multi-step word problem, a single CoT pass might commit to an incorrect unit conversion in step 2 and propagate it silently to the final answer. Under Quillan-Ronin, that conversion is emitted as a discrete hypothesis, sent through VERIFY (which recomputes it independently), and — if it fails — the ledger records `contradictions: [recomputation_mismatch]` and the state machine returns to HYPOTHESIZE with that failure mode explicitly visible to the next generation pass, rather than silently baked into an ever-growing prose trace. This is the mechanism by which we expect (Section 4) the protocol to reduce a specific, common failure class rather than to improve reasoning "in general."

---

## 4. Evaluation Protocol (Template — Results Pending)

We specify the evaluation design fully here. **No results are reported in this draft**; the tables below are structured for direct completion once the runs are executed, and every cell is marked accordingly rather than populated with placeholder-looking-like-real numbers.

### 4.1 Benchmarks

- **GSM8K** (Cobbe et al., 2021) — grade-school arithmetic word problems, multi-step.
- **MATH** (Hendrycks et al., 2021) — competition mathematics, higher difficulty ceiling.
- **BIG-Bench-Hard (BBH)** (Suzgun et al., 2022) — logical deduction and multi-step symbolic tasks.

### 4.2 Conditions

| Condition | Description |
|---|---|
| CoT | Single-pass chain-of-thought, same base model |
| Self-Consistency@k | k independently sampled CoT traces, majority vote, k matched to Quillan-Ronin's expected call count |
| ToT | Breadth-limited tree search with heuristic scoring, same base model |
| Quillan-Ronin | Full structured deliberation protocol (Section 3) |

### 4.3 Metrics

1. **Accuracy@1** — proportion of problems solved correctly.
2. **Total generation cost** — sum of tokens (or FLOP-equivalent, given a fixed per-token cost for the base model) consumed to reach the final answer, including all discarded branches/backtracks. This is the metric the "accuracy-per-FLOP" claim depends on; it is not accuracy alone.
3. **Output variance** — standard deviation of the answer distribution across $n=10$ independent runs per problem, at fixed temperature. This tests the "lower variance" claim directly.

### 4.4 Result Template

| Benchmark | Condition | Accuracy@1 | Mean cost (tokens) | Std. dev. across runs |
|---|---|---|---|---|
| GSM8K | CoT | *TBD* | *TBD* | *TBD* |
| GSM8K | Self-Consistency@k | *TBD* | *TBD* | *TBD* |
| GSM8K | ToT | *TBD* | *TBD* | *TBD* |
| GSM8K | Quillan-Ronin | *TBD* | *TBD* | *TBD* |
| MATH | CoT | *TBD* | *TBD* | *TBD* |
| MATH | Self-Consistency@k | *TBD* | *TBD* | *TBD* |
| MATH | ToT | *TBD* | *TBD* | *TBD* |
| MATH | Quillan-Ronin | *TBD* | *TBD* | *TBD* |
| BBH | CoT | *TBD* | *TBD* | *TBD* |
| BBH | Self-Consistency@k | *TBD* | *TBD* | *TBD* |
| BBH | ToT | *TBD* | *TBD* | *TBD* |
| BBH | Quillan-Ronin | *TBD* | *TBD* | *TBD* |

The Pareto-improvement claim in the abstract is a **hypothesis to be tested against this table**, not a reported finding.

---

## 5. Ablation Design: Isolating Structure from Scale

The central risk to the paper's thesis is that any observed gain is really just "more compute" (more calls to the base model) rather than *structure*. The ablation below is designed to control for that directly.

### 5.1 Conditions

| Condition | What it removes | Purpose |
|---|---|---|
| Full protocol | — | Baseline for comparison |
| **No-structure, matched-compute** | State machine and transition rules removed; the *same number* of base-model calls are made, but simply concatenated as additional unstructured CoT continuations | Isolates whether the state machine itself adds value beyond raw additional inference calls |
| **No-challenge** | CHALLENGE state removed (VERIFY passes go straight to SYNTHESIZE) | Isolates the contribution of the adversarial step specifically |
| **No-ledger-schema** | Structured ledger replaced with free-text scratchpad carrying equivalent information | Isolates whether schema-constrained memory matters vs. unstructured memory with the same content |
| **Single-role** | All role-conditioned prompts replaced with one neutral prompt reused across states | Isolates whether role-conditioning contributes beyond the state machine itself |

### 5.2 Expected Signal (Template)

| Ablation | Accuracy Δ vs. full protocol | Interpretation if Δ ≈ 0 | Interpretation if Δ is large |
|---|---|---|---|
| No-structure, matched-compute | *TBD* | Gains are from compute, not structure — thesis fails | Structure adds value beyond raw compute — supports thesis |
| No-challenge | *TBD* | Adversarial step is not load-bearing | Adversarial verification is doing real work |
| No-ledger-schema | *TBD* | Schema constraint is cosmetic | Structured memory materially reduces error propagation |
| Single-role | *TBD* | Role-conditioning is decorative | Role separation reduces self-confirmation bias |

The **no-structure, matched-compute** ablation is the one that actually tests "structure > scale," and is non-negotiable for the paper's central claim to be defensible; without it, any accuracy gain is confounded with call count.

---

## 6. Limitations & Future Work

**Latency.** The protocol trades wall-clock latency for accuracy/variance by design — every additional VERIFY/CHALLENGE round is a full additional inference call. This is the same trade CoT and ToT already make, but structured deliberation's mandatory verification/challenge steps put a *floor* under the minimum number of calls per episode (unlike CoT, which can be single-pass), which is a real cost this paper does not try to hide.

**Brittleness of the hard-coded controller.** The transition function $\delta$ in Section 2.2 is a fixed set of threshold rules ($\tau_v$, $\tau_a$, $b_{\max}$) chosen by the implementer, not learned from data. This is honest but limiting: the thresholds are almost certainly not optimal, and a controller tuned for GSM8K-style arithmetic may transition too eagerly or too conservatively for open-ended or creative tasks where "verification" and "contradiction" are not well-defined the way they are in a math problem.

**Model dependence.** Because the protocol is implemented via prompting rather than joint training, its effectiveness is bounded by the base model's ability to (a) reliably follow role-specific instructions and (b) produce schema-constrained structured output under each role. A weaker instruction-following model may not honestly play the CHALLENGE role (i.e., may reflexively agree rather than genuinely search for contradictions), which would silently degrade the protocol into expensive self-consistency.

**Future work — a learned deliberation policy.** The most direct next step is replacing the hard-coded $\delta$ with a small, separately trained **policy network** that takes the evidence tuple $(c, v, a)$ and ledger state as input and learns the transition rule from outcome data (i.e., learns when backtracking is actually worth it, rather than being told by a fixed threshold), analogous to how process-reward models (Lightman et al., 2023) replace hand-written heuristics with learned step-level value estimates. A second direction is distilling successful deliberation traces produced by this protocol into fine-tuning data for a single model, so that some of the structure is internalized rather than imposed externally at inference time — trading the latency cost above for training cost instead.

---

## References

- Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS.
- Wang, X., Wei, J., Schuurmans, D., et al. (2022). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023 (arXiv 2022).
- Yao, S., Yu, D., Zhao, J., et al. (2023). *Tree of Thoughts: Deliberate Problem Solving with Large Language Models.* NeurIPS.
- Besta, M., Blach, N., Kubicek, A., et al. (2024). *Graph of Thoughts: Solving Elaborate Problems with Large Language Models.* AAAI.
- Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). *Let's Verify Step by Step.* OpenAI.
- Yao, S., Zhao, J., Yu, D., et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
- Shinn, N., Cassano, F., Berman, E., et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS.
- Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). *Training Verifiers to Solve Math Word Problems.* (GSM8K) arXiv.
- Hendrycks, D., Burns, C., Kadavath, S., et al. (2021). *Measuring Mathematical Problem Solving with the MATH Dataset.* NeurIPS Datasets and Benchmarks.
- Suzgun, M., Scales, N., Schärli, N., et al. (2022). *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* (BBH) arXiv.

---

*Note on scope: this draft presents Quillan-Ronin as an inference-time protocol layered over an existing base LLM via structured prompting and role-conditioned calls, evaluated by its behavioral effect on reasoning accuracy, cost, and variance. It does not claim new pretrained model weights, a novel attention architecture, or a literally simulated population of independent agents — those framings, if used elsewhere, describe the conceptual/persona layer rather than a verified training run, and are kept separate from the empirical claims here so the evaluation in Sections 4–5 stays falsifiable.*
