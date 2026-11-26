#!/usr/bin/env python3
"""DECISIVE TEST: truly-untouched 40-question bank.

Design rules (to block every known failure mode):
  - Zero factual overlap with general_knowledge.py's 25 questions (the tuning bank).
    Neither the WSP loop nor its promotion gate ever saw these.
  - Correct answer position is assigned programmatically by a fixed cycle
    (i % 4), giving exactly 10 correct answers per letter A/B/C/D. Manual
    authoring bias toward one letter cannot inflate a raw accuracy number.
  - Same prompt template, tokenizer, letter_ids and loading path as the tuning
    harness, so the ONLY changed variable is the checkpoint.
  - Reports per-letter prediction distribution for both checkpoints to detect
    a pure answer-letter bias shift masquerading as knowledge gain.

Scores: base quillan-mini-sovereign-bitnet.safetensors vs
        tuned quillan-mini-wsp-tcp2.safetensors
"""

from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\02_QUILLAN")
DIAG = ROOT / "scripts" / "adapter_diagnostics"
for p in (str(ROOT / "scripts"), str(ROOT / "03 - Training & Model"), str(DIAG)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsp_math_eval import tok_for, letter_ids, load_model
from wsp_math_tune_v3 import enable_cache

# NOTE: thread count is owned by the top-level script, not this data module.
# A set_num_threads() here silently overrides the caller and corrupts A/B
# comparisons (Predator audit, 2026-11-24). Do not set it in bank modules.
LETTERS = ["A", "B", "C", "D"]

# (question, correct answer string, [distractor1, distractor2, distractor3])
# All hand-verifiable. Deliberately disjoint from the original 25 facts.
RAW = [
    ("What is the chemical symbol for iron?", "Fe", ["Ir", "F", "Fr"]),
    (
        "Which planet is famous for its prominent ring system?",
        "Saturn",
        ["Jupiter", "Uranus", "Neptune"],
    ),
    ("In which year did World War I end?", "1918", ["1916", "1917", "1919"]),
    (
        "Who wrote 'The Great Gatsby'?",
        "F. Scott Fitzgerald",
        ["Ernest Hemingway", "John Steinbeck", "William Faulkner"],
    ),
    (
        "What is the largest hot desert on Earth?",
        "Sahara",
        ["Arabian", "Gobi", "Kalahari"],
    ),
    (
        "Which organelle carries out photosynthesis?",
        "Chloroplast",
        ["Ribosome", "Lysosome", "Golgi"],
    ),
    ("What is the chemical formula of table salt?", "NaCl", ["KCl", "CaCl2", "MgCl2"]),
    ("How many bones are in the adult human body?", "206", ["186", "196", "216"]),
    (
        "Which element has the chemical symbol K?",
        "Potassium",
        ["Krypton", "Kelvin", "Kryptonite"],
    ),
    ("What is the capital of France?", "Paris", ["Lyon", "Marseille", "Toulouse"]),
    ("Which ocean lies east of Africa?", "Indian", ["Atlantic", "Pacific", "Arctic"]),
    (
        "Who painted 'The Starry Night'?",
        "Vincent van Gogh",
        ["Pablo Picasso", "Claude Monet", "Salvador Dali"],
    ),
    (
        "What is the approximate speed of light in vacuum in km/s?",
        "300,000",
        ["150,000", "3,000,000", "30,000"],
    ),
    (
        "Which type of blood vessel carries blood away from the heart?",
        "Artery",
        ["Vein", "Capillary", "Venule"],
    ),
    (
        "What is the most abundant gas in Earth's atmosphere?",
        "Nitrogen",
        ["Oxygen", "Carbon dioxide", "Argon"],
    ),
    (
        "What is the largest moon in the solar system?",
        "Ganymede",
        ["Titan", "Callisto", "Europa"],
    ),
    ("In which year did the Berlin Wall fall?", "1989", ["1987", "1988", "1990"]),
    (
        "Who was the first person to walk on the Moon?",
        "Neil Armstrong",
        ["Buzz Aldrin", "Michael Collins", "Yuri Gagarin"],
    ),
    (
        "What is the hard outer covering of an insect called?",
        "Exoskeleton",
        ["Endoskeleton", "Carapace", "Test"],
    ),
    (
        "Which vitamin does the body produce from sunlight?",
        "Vitamin D",
        ["Vitamin A", "Vitamin B12", "Vitamin C"],
    ),
    (
        "What is the largest species of shark?",
        "Whale shark",
        ["Great white", "Tiger shark", "Hammerhead"],
    ),
    ("How many sides does a hexagon have?", "6", ["5", "7", "8"]),
    ("Which planet has the shortest day?", "Jupiter", ["Earth", "Mercury", "Venus"]),
    (
        "Which part of the brain is chiefly responsible for balance?",
        "Cerebellum",
        ["Cerebrum", "Brainstem", "Hippocampus"],
    ),
    (
        "In which country is the Great Wall located?",
        "China",
        ["Japan", "Korea", "India"],
    ),
    ("What is the smallest prime number?", "2", ["0", "1", "3"]),
    (
        "Which element forms the graphite used in pencils?",
        "Carbon",
        ["Lead", "Silicon", "Sulfur"],
    ),
    (
        "Which hormone regulates blood sugar?",
        "Insulin",
        ["Adrenaline", "Thyroxine", "Melatonin"],
    ),
    ("How many planets are in the Solar System?", "8", ["7", "9", "10"]),
    (
        "Which is the largest bone in the human body?",
        "Femur",
        ["Tibia", "Humerus", "Skull"],
    ),
    ("In which year did the Titanic sink?", "1912", ["1910", "1911", "1914"]),
    (
        "Who wrote the novel '1984'?",
        "George Orwell",
        ["Aldous Huxley", "Ray Bradbury", "Arthur Koestler"],
    ),
    (
        "What is the central region of an atom called?",
        "Nucleus",
        ["Electron", "Valence shell", "Neutron cloud"],
    ),
    ("Which direction does the Sun rise?", "East", ["North", "South", "West"]),
    (
        "Which country has the largest land area?",
        "Russia",
        ["Canada", "China", "United States"],
    ),
    (
        "What is the primary component of the Sun?",
        "Hydrogen",
        ["Helium", "Oxygen", "Carbon"],
    ),
    (
        "Which sense is associated with the retina?",
        "Sight",
        ["Hearing", "Smell", "Touch"],
    ),
    ("How many chromosomes do humans typically have?", "46", ["23", "44", "48"]),
    (
        "Which war was fought between the North and South in the United States?",
        "Civil War",
        ["Revolutionary War", "World War I", "Korean War"],
    ),
    (
        "What is the largest internal organ of the human body?",
        "Liver",
        ["Kidney", "Spleen", "Pancreas"],
    ),
]

assert len(RAW) == 40
QUESTIONS, KEY = [], []
for i, (q, correct, distractors) in enumerate(RAW):
    pos = i % 4  # perfectly balanced: 10 correct per letter
    opts, di = [None] * 4, 0
    opts[pos] = correct
    for slot in range(4):
        if opts[slot] is None:
            opts[slot] = distractors[di]
            di += 1
    QUESTIONS.append((q, opts, pos))
    KEY.append(pos)
assert [KEY.count(j) for j in range(4)] == [10, 10, 10, 10]


def prompts(tok):
    return [
        tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        for i, (q, opts, _) in enumerate(QUESTIONS)
    ]


def score(ck):
    m = load_model(ck)
    enable_cache(m)
    tok = tok_for(m.cfg.vocab_size)
    lids = letter_ids(tok, m.cfg.vocab_size)
    ids = prompts(tok)
    preds, dist = [], [0, 0, 0, 0]
    with torch.inference_mode():
        for seq in ids:
            lg = m(torch.tensor([seq]))[0][0, -1, lids].float()
            pi = int(lg.argmax())
            preds.append(pi)
            dist[pi] += 1
    ok = sum(int(p == KEY[i]) for i, p in enumerate(preds))
    return ok, dist, preds


def main():
    base = ROOT / "exports" / "quillan-mini-sovereign-bitnet.safetensors"
    tuned = ROOT / "exports" / "quillan-mini-wsp-tcp2.safetensors"
    for name, ck in (("BASE", base), ("WSP-TCP2", tuned)):
        if not ck.exists():
            print(f"[{name}] MISSING {ck}", flush=True)
            continue
        ok, dist, preds = score(ck)
        print(
            f"[{name}] FRESH40={ok}/40 pred_dist={dist} true_dist=[10,10,10,10]",
            flush=True,
        )
        print(f"[{name}] preds={preds}", flush=True)


if __name__ == "__main__":
    main()
