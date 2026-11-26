#!/usr/bin/env python3
"""SECOND untouched 40-question bank, balanced A/B/C/D, disjoint from both
general_knowledge.py (25) and heldout_fresh40.py (40).

Built AFTER the letter-rebalance edit was locked, so it is a genuine
out-of-sample confirmation set. Answer positions assigned by i % 4.
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

RAW = [
    ("What is the chemical symbol for copper?", "Cu", ["Ag", "Au", "Co"]),
    ("Which planet has the Great Red Spot?", "Jupiter", ["Saturn", "Mars", "Neptune"]),
    ("In which year did World War II begin?", "1939", ["1937", "1938", "1940"]),
    (
        "Who wrote 'The Catcher in the Rye'?",
        "J.D. Salinger",
        ["John Steinbeck", "F. Scott Fitzgerald", "Ernest Hemingway"],
    ),
    (
        "What is the largest island in the world?",
        "Greenland",
        ["New Guinea", "Borneo", "Madagascar"],
    ),
    ("Which molecule forms the double helix?", "DNA", ["RNA", "Glucose", "Lipid"]),
    ("What is the pH of pure water?", "7", ["5", "6", "9"]),
    ("How many colors are in a rainbow?", "7", ["5", "6", "8"]),
    ("Which element has the symbol Na?", "Sodium", ["Nitrogen", "Neon", "Nickel"]),
    ("What is the capital of Italy?", "Rome", ["Milan", "Naples", "Florence"]),
    ("Which ocean is the smallest?", "Arctic", ["Indian", "Atlantic", "Pacific"]),
    ("Who composed 'The Four Seasons'?", "Vivaldi", ["Bach", "Mozart", "Beethoven"]),
    ("What is the SI unit of force?", "Newton", ["Joule", "Watt", "Pascal"]),
    (
        "Which part of the eye controls how much light enters?",
        "Iris",
        ["Retina", "Cornea", "Lens"],
    ),
    (
        "What is the most abundant metal in Earth's crust?",
        "Aluminium",
        ["Iron", "Copper", "Silver"],
    ),
    (
        "What is the name of our galaxy?",
        "Milky Way",
        ["Andromeda", "Triangulum", "Sombrero"],
    ),
    ("In which year did the Soviet Union dissolve?", "1991", ["1989", "1990", "1992"]),
    (
        "Who was the first woman to win a Nobel Prize?",
        "Marie Curie",
        ["Rosalind Franklin", "Ada Lovelace", "Grace Hopper"],
    ),
    (
        "What is the hardest natural substance?",
        "Diamond",
        ["Quartz", "Granite", "Steel"],
    ),
    (
        "Which vitamin is ascorbic acid?",
        "Vitamin C",
        ["Vitamin A", "Vitamin D", "Vitamin B6"],
    ),
    ("Which is the largest cat species?", "Tiger", ["Lion", "Leopard", "Jaguar"]),
    ("How many hearts does an octopus have?", "3", ["1", "2", "4"]),
    ("What is the freezing point of water in Fahrenheit?", "32", ["0", "20", "100"]),
    (
        "Which bone connects the shoulder to the elbow?",
        "Humerus",
        ["Femur", "Radius", "Tibia"],
    ),
    (
        "On which continent is the Sahara?",
        "Africa",
        ["Asia", "Australia", "South America"],
    ),
    ("What is 7 times 8?", "56", ["48", "54", "64"]),
    (
        "Which metal was traditionally used in thermometers?",
        "Mercury",
        ["Gold", "Silver", "Lead"],
    ),
    (
        "Which neurotransmitter is associated with reward?",
        "Dopamine",
        ["Serotonin", "GABA", "Acetylcholine"],
    ),
    ("How many strings does a standard guitar have?", "6", ["4", "5", "7"]),
    (
        "Which planet is known as the Morning Star?",
        "Venus",
        ["Mercury", "Mars", "Jupiter"],
    ),
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ("Which blood group is the universal donor?", "O", ["A", "B", "AB"]),
    (
        "What is the study of fossils called?",
        "Paleontology",
        ["Archaeology", "Geology", "Botany"],
    ),
    (
        "Who painted 'Guernica'?",
        "Picasso",
        ["Salvador Dali", "Joan Miro", "Francisco Goya"],
    ),
    (
        "What is the largest lake in Africa?",
        "Lake Victoria",
        ["Lake Tanganyika", "Lake Malawi", "Lake Chad"],
    ),
    ("How many legs does a crab have?", "10", ["6", "8", "12"]),
    (
        "What is the Sun's outermost layer called?",
        "Corona",
        ["Core", "Mantle", "Photosphere"],
    ),
    (
        "What is the smallest country in the world by area?",
        "Vatican City",
        ["Monaco", "San Marino", "Liechtenstein"],
    ),
    ("Which river flows through Paris?", "Seine", ["Loire", "Rhone", "Thames"]),
    (
        "What is the main language of Brazil?",
        "Portuguese",
        ["Spanish", "French", "English"],
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
