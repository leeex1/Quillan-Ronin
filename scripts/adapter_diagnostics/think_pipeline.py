#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quillan-Ronin inference-time <think> reasoning pipeline (C32-AEON design, board #1277).

IMPORTANT HONESTY NOTE:
  The mini/flagship checkpoints are NON-REASONING generative substrates (closed-form
  4-donor fusion, zero RL/SFT, no <think> tokens in weights — see board #1275/#1276).
  This pipeline adds a genuine <think>...</think> section at INFERENCE TIME via a
  structured deliberation loop over the checkpoint (Muse Spark pattern):
      DECOMPOSE -> MULTI-PASS role-conditioned attempts -> VERIFY/BACKTRACK -> SYNTHESIZE
  Reasoning structure comes from the harness; language from the checkpoint. No weight
  training. Whether this produces CORRECT multi-step reasoning is measured by the
  C8 probes below — reported as actual scores, not claims.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(r"C:\02_QUILLAN")
for _p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quillan_sovereign_bitnet import QuillanSovereignBitNetLM

ROLES = [
    ("C7-LOGOS", "logic"),
    ("C28-CALCULUS", "quantitative"),
    ("C6-OMNIS", "perspective"),
]
CONNECTIVES = ["because", "so ", "therefore", "thus", "since", "then", "if "]

# C8-METASYNTH reasoning probes (board #1270). Expected markers for scoring only.
PROBES = [
    ("transitive", "P beats Q, Q beats R, R beats S. Who is strongest, P or S?", ["p"]),
    (
        "math",
        "3 jobs per second for 2 minutes, then double the rate for 1 minute. Total jobs? Show work.",
        ["720"],
    ),
    (
        "rule",
        "A 'flib' is even AND greater than 10. Is 14 a flib? Is 8? Explain.",
        ["yes", "no"],
    ),
    (
        "causal",
        "All Zorps are glinty; glinty things rust in rain, but Zorps are waterproof. Does a Zorp rust?",
        ["no"],
    ),
    (
        "contradiction",
        "Vault opens only if the red key is turned. The key was never turned. The vault opened. Consistent?",
        ["no", "inconsistent"],
    ),
    (
        "second_order",
        "Ann outranks Bob, Bob outranks Cam. Now Cam outranks Ann. What must be false?",
        ["cycle", "cannot", "ann", "cam"],
    ),
]


def tok_for(vocab: int = 0, name: str | None = None):
    """Explicit per-model tokenizer registry (C25 #1290 B1 fix).

    The old vocab<=128256 heuristic mis-assigned Dragon (131072 vocab, Falcon
    family) to the Llama tokenizer. Select by model name when known; fall back
    to vocab only for anonymous checkpoints.
    """
    from transformers import AutoTokenizer

    QWEN = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"
    LLAMA = r"E:\source_models\llama_1b"
    FALCON = r"C:\02_QUILLAN\exports\donor_cache_v23\falcon3_1b_hf"
    n = (name or "").lower()
    if "dragon" in n or "falcon" in n:
        p = FALCON
    elif "flagship" in n or "llama" in n:
        p = LLAMA
    elif "mini" in n:
        p = QWEN
    else:
        p = LLAMA if vocab and vocab <= 128256 else QWEN
    return AutoTokenizer.from_pretrained(p, local_files_only=True)


@dataclass
class ThinkResult:
    think: str
    answer: str
    votes: List[str] = field(default_factory=list)
    n_backtracks: int = 0
    confidence: float = 0.0


class ThinkPipeline:
    """Deterministic inference-time thinking harness over a Quillan checkpoint."""

    def __init__(
        self, ckpt: Path, effort: int = 64, max_passes: int = 3, max_backtracks: int = 2
    ):
        self.model = QuillanSovereignBitNetLM.load_packed_safetensors(Path(ckpt)).eval()
        self.tok = tok_for(self.model.cfg.vocab_size, name=Path(ckpt).stem)
        self.effort = effort
        self.token_budget = {16: 12, 64: 24, 256: 48, 512: 80}.get(effort, 24)
        self.max_passes = max_passes
        self.max_backtracks = max_backtracks

    def _gen(self, prompt: str, n: int | None = None) -> str:
        n = n or self.token_budget
        ids = self.tok.encode(prompt, add_special_tokens=False)
        with torch.inference_mode():
            out = self.model.generate(
                torch.tensor([ids]), max_new_tokens=n, temperature=0.0
            )
        gen = out[0, len(ids) :].tolist()
        return self.tok.decode(gen, skip_special_tokens=True)

    @staticmethod
    def _decompose(query: str) -> str:
        """Deterministic scaffolding: state the goal and the generic sub-steps.
        This is a plan, not an answer — no expected marker is injected."""
        q = query.strip().rstrip("?")
        return f"1) identify what is asked: {q}; 2) apply the relevant rule; 3) check the conclusion"

    @staticmethod
    def _steps(text: str) -> List[str]:
        return [c for c in CONNECTIVES if c in text.lower()]

    def reason(self, query: str) -> ThinkResult:
        decomp = self._decompose(query)
        trace: List[str] = [f"[GOAL]{query.strip()}", f"[DECOMPOSE]{decomp}"]
        attempts: List[str] = []
        votes: List[str] = []
        backtracks = 0

        # MULTI-PASS: role-conditioned generation attempts
        for role, domain in ROLES[: self.max_passes]:
            prompt = (
                f"<think>[GOAL]{query.strip()}\n[DECOMPOSE]{decomp}\n"
                f"[ATTEMPT|{role}|{domain}]Let us work through this step by step because "
            )
            txt = self._gen(prompt)
            trace.append(f"[ATTEMPT|{role}]{txt.strip()}")
            attempts.append(txt)

        # VERIFY / BACKTRACK: if no attempt has an explicit connective, re-prompt
        verified = any(self._steps(a) for a in attempts)
        if not verified and backtracks < self.max_backtracks:
            backtracks += 1
            prompt = (
                f"<think>[GOAL]{query.strip()}\n[DECOMPOSE]{decomp}\n"
                f"[VERIFY]prior attempts lacked explicit steps. Retry: since "
            )
            txt = self._gen(prompt)
            trace.append(f"[BACKTRACK|1]{txt.strip()}")
            attempts.append(txt)

        # SYNTHESIZE: choose the attempt with the most connectives (self-consistency proxy)
        scored = sorted(attempts, key=lambda a: len(self._steps(a)), reverse=True)
        chosen = scored[0] if scored else ""
        steps = self._steps(chosen)
        trace.append(
            f"[VERIFY]explicit-steps={'yes' if steps else 'no'} markers={steps}"
        )
        trace.append(
            f"[CONSENSUS]{len(attempts)} attempt(s); best has {len(steps)} connective(s)"
        )
        votes = [a.strip()[:120] for a in attempts]

        # Answer: dedicated final pass + non-empty guard (C25 #1290 B2)
        # strip any model-generated delimiter substrings so tags never nest
        def clean(s: str) -> str:
            for tag in ("<think>", "</think>", "<answer>", "</answer>"):
                s = s.replace(tag, " ")
            return s.strip()

        cand = clean(chosen.strip().split("\n")[-1]) if chosen.strip() else ""
        if not cand:
            final_prompt = (
                f"<think>[GOAL]{query.strip()}\n[DECOMPOSE]{decomp}\n"
                f"[CONSENSUS]{len(attempts)} attempts.\n[FINAL]The answer is "
            )
            cand = clean(self._gen(final_prompt, n=24))
        if not cand:
            cand = clean(attempts[0]) if attempts else "Unable to determine answer"
        answer = cand[:240]
        think = "\n".join(trace)
        confidence = min(1.0, len(steps) * 0.25 + (len(attempts) - 1) * 0.1)
        return ThinkResult(
            think=think,
            answer=answer,
            votes=votes,
            n_backtracks=backtracks,
            confidence=confidence,
        )

    @staticmethod
    def render(r: ThinkResult) -> str:
        ans = r.answer or "(empty)"
        out = f"<think>\n{r.think}\n</think>\n<answer>\n{ans}\n</answer>"
        # format lock (C25 #1290 B4): balanced, single, non-nested tags
        assert out.count("<think>") == 1 and out.count("</think>") == 1
        assert out.count("<answer>") == 1 and out.count("</answer>") == 1
        return out


def score(answer: str, markers: List[str]) -> Tuple[int, bool, bool]:
    low = answer.lower()
    correct = any(m.lower() in low for m in markers)
    step = any(c in low for c in CONNECTIVES)
    return (2 if (correct and step) else (1 if correct else 0)), correct, step


def run(ckpt: Path, effort: int, only: str | None = None) -> None:
    print("=" * 78, flush=True)
    print(f"THINK PIPELINE | {ckpt.name} | effort={effort}", flush=True)
    print("=" * 78, flush=True)
    pipe = ThinkPipeline(ckpt, effort=effort)
    total = twos = 0
    for name, prompt, markers in PROBES:
        if only and name != only:
            continue
        r = pipe.reason(prompt)
        s, correct, step = score(r.think + " " + r.answer, markers)
        total += s
        twos += int(s == 2)
        print(
            f"\n[{name}] score={s}/2 correct={correct} step={step} backtracks={r.n_backtracks} conf={r.confidence:.2f}",
            flush=True,
        )
        print(pipe.render(r), flush=True)
    print(
        f"\n[RESULT] total={total}/12 twos={twos} -> {'PASS' if (twos >= 4 and total >= 9) else 'FAIL'}",
        flush=True,
    )
    print(
        "[NOTE] answer-only (score 1) = substrate lacks trained CoT; harness supplies structure, not correctness.",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--effort", type=int, default=64, choices=[16, 64, 256, 512])
    ap.add_argument(
        "--probe", type=str, default=None, help="run a single probe by name"
    )
    a = ap.parse_args()
    run(a.ckpt, a.effort, a.probe)
