#!/usr/bin/env python3
"""THIRD untouched 40-question bank, balanced A/B/C/D.

Built AFTER the late-block-damp factor (0.5-0.7) was fixed, so no factor
selection used this bank. Genuine out-of-sample confirmation.
Disjoint from general_knowledge.py (25), heldout_fresh40 (40), fresh40b (40).
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

RAW = [
    (
        "What gas do plants release during photosynthesis?",
        "Oxygen",
        ["Nitrogen", "Carbon dioxide", "Hydrogen"],
    ),
    (
        "Which planet is farthest from the Sun?",
        "Neptune",
        ["Uranus", "Saturn", "Pluto"],
    ),
    (
        "In which year did the first airplane flight occur?",
        "1903",
        ["1899", "1901", "1905"],
    ),
    (
        "Who wrote 'War and Peace'?",
        "Leo Tolstoy",
        ["Fyodor Dostoevsky", "Ivan Turgenev", "Alexander Pushkin"],
    ),
    (
        "What is the largest bird by wingspan?",
        "Wandering albatross",
        ["Golden eagle", "Andean condor", "Ostrich"],
    ),
    ("Which part of a plant absorbs water?", "Roots", ["Leaves", "Stem", "Flowers"]),
    (
        "What is the boiling point of water at sea level in Kelvin?",
        "373",
        ["273", "300", "100"],
    ),
    ("How many bones are in the human skull?", "22", ["24", "26", "20"]),
    ("Which element has the symbol 'Fe'?", "Iron", ["Fluorine", "Fermium", "Francium"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver", "Montreal"]),
    (
        "Which desert covers most of North Africa?",
        "Sahara",
        ["Gobi", "Kalahari", "Atacama"],
    ),
    ("Who composed ' Symphony No. 5'?", "Beethoven", ["Mozart", "Haydn", "Schubert"]),
    ("What is the SI unit of energy?", "Joule", ["Watt", "Newton", "Pascal"]),
    ("Which organ produces bile?", "Liver", ["Kidney", "Stomach", "Pancreas"]),
    (
        "What is the most abundant element in the universe?",
        "Hydrogen",
        ["Helium", "Oxygen", "Carbon"],
    ),
    (
        "Which planet has a day longer than its year?",
        "Venus",
        ["Mercury", "Earth", "Mars"],
    ),
    ("In which year did the moon landing occur?", "1969", ["1965", "1967", "1971"]),
    (
        "Who developed the polio vaccine?",
        "Jonas Salk",
        ["Alexander Fleming", "Edward Jenner", "Louis Pasteur"],
    ),
    (
        "What is the hardest rock on the Mohs scale?",
        "Diamond",
        ["Quartz", "Topaz", "Corundum"],
    ),
    (
        "Which vitamin is produced by gut bacteria?",
        "Vitamin K",
        ["Vitamin A", "Vitamin C", "Vitamin D"],
    ),
    ("Which is the fastest land animal?", "Cheetah", ["Lion", "Gazelle", "Horse"]),
    ("How many chromosomes does a fruit fly have?", "8", ["4", "6", "10"]),
    ("What is the normal human body temperature in Celsius?", "37", ["35", "36", "38"]),
    (
        "Which muscle is the largest in the human body?",
        "Gluteus maximus",
        ["Biceps", "Quadriceps", "Latissimus dorsi"],
    ),
    (
        "On which continent is the Amazon rainforest?",
        "South America",
        ["Africa", "Asia", "Central America"],
    ),
    ("What is 9 multiplied by 6?", "54", ["48", "56", "64"]),
    (
        "Which liquid is used as a fuel in rockets?",
        "Liquid hydrogen",
        ["Liquid nitrogen", "Liquid oxygen alone", "Gasoline"],
    ),
    (
        "Which neurotransmitter is associated with calm?",
        "GABA",
        ["Dopamine", "Adrenaline", "Acetylcholine"],
    ),
    ("How many sides does a pentagon have?", "5", ["4", "6", "7"]),
    (
        "Which star is closest to Earth?",
        "Sun",
        ["Proxima Centauri", "Sirius", "Alpha Centauri"],
    ),
    ("What is the capital of Egypt?", "Cairo", ["Alexandria", "Giza", "Luxor"]),
    (
        "Which blood vessel carries blood TO the heart?",
        "Vein",
        ["Artery", "Aorta", "Capillary"],
    ),
    (
        "What is the study of earthquakes called?",
        "Seismology",
        ["Geology", "Meteorology", "Volcanology"],
    ),
    ("Who wrote 'The Odyssey'?", "Homer", ["Virgil", "Dante", " Sophocles"]),
    (
        "What is the largest lake in North America?",
        "Lake Superior",
        ["Lake Michigan", "Lake Huron", "Great Slave Lake"],
    ),
    ("How many eyes does a honeybee have?", "5", ["2", "3", "6"]),
    (
        "What is the outermost planet with rings visible from Earth?",
        "Saturn",
        ["Jupiter", "Uranus", "Neptune"],
    ),
    ("Which country invented paper?", "China", ["Egypt", "Greece", "India"]),
    ("Which river flows through London?", "Thames", ["Seine", "Danube", "Rhine"]),
    (
        "What is the primary language of Argentina?",
        "Spanish",
        ["Portuguese", "French", "Italian"],
    ),
]

assert len(RAW) == 40
QUESTIONS, KEY = [], []
for i, (q, correct, distractors) in enumerate(RAW):
    pos = i % 4
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
