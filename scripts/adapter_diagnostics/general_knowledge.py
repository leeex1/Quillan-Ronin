#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out GENERAL KNOWLEDGE benchmark (A/B/C/D), disjoint from math + WSP calib.

Verifiable factual/science/history questions. Used to measure whether pattern
surgery IMPROVES general accuracy (and reduces failure points) vs a baseline —
the real test the user asked for, distinct from answer-memorization.
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

# (question, [A,B,C,D], correct index) — all hand-verifiable
QUESTIONS = [
    ("Which planet is closest to the Sun?", ["Venus", "Mercury", "Earth", "Mars"], 1),
    ("What is the chemical symbol for gold?", ["Go", "Gd", "Au", "Ag"], 2),
    (
        "Who wrote 'Pride and Prejudice'?",
        ["Jane Austen", "Charles Dickens", "George Eliot", "Emily Bronte"],
        0,
    ),
    ("In which year did World War II end?", ["1943", "1944", "1945", "1946"], 2),
    (
        "What is the largest ocean on Earth?",
        ["Atlantic", "Indian", "Arctic", "Pacific"],
        3,
    ),
    (
        "Which gas do plants absorb from the atmosphere?",
        ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
        2,
    ),
    (
        "What is the powerhouse of the cell?",
        ["Nucleus", "Mitochondrion", "Ribosome", "Golgi"],
        1,
    ),
    (
        "Who was the first president of the United States?",
        ["Thomas Jefferson", "John Adams", "George Washington", "James Madison"],
        2,
    ),
    ("What is the square root of 144?", ["11", "12", "13", "14"], 1),
    (
        "Which element has atomic number 1?",
        ["Helium", "Hydrogen", "Carbon", "Oxygen"],
        1,
    ),
    ("What is the capital of Japan?", ["Kyoto", "Osaka", "Tokyo", "Nagoya"], 2),
    (
        "Which is the largest mammal?",
        ["Blue whale", "Elephant", "Giraffe", "Whale shark"],
        0,
    ),
    ("How many continents are there?", ["5", "6", "7", "8"], 2),
    (
        "What organ pumps blood through the body?",
        ["Liver", "Heart", "Lungs", "Kidney"],
        1,
    ),
    (
        "Which language has the most native speakers?",
        ["English", "Hindi", "Mandarin Chinese", "Spanish"],
        2,
    ),
    (
        "What is H2O commonly known as?",
        ["Oxygen", "Water", "Hydrogen peroxide", "Salt"],
        1,
    ),
    (
        "Who developed the theory of relativity?",
        ["Newton", "Einstein", "Galileo", "Kepler"],
        1,
    ),
    (
        "Which is the longest river in the world?",
        ["Amazon", "Nile", "Yangtze", "Mississippi"],
        1,
    ),
    ("What color is the sky on a clear day?", ["Red", "Green", "Blue", "Yellow"], 2),
    ("How many legs does a spider have?", ["6", "8", "10", "12"], 1),
    (
        "Which planet is known as the Red Planet?",
        ["Venus", "Mars", "Jupiter", "Saturn"],
        1,
    ),
    ("What is the freezing point of water in Celsius?", ["-10", "0", "10", "100"], 1),
    (
        "Which blood cells fight infection?",
        ["Red cells", "White cells", "Platelets", "Plasma"],
        1,
    ),
    (
        "Who painted the Mona Lisa?",
        ["Michelangelo", "Leonardo da Vinci", "Raphael", "Donatello"],
        1,
    ),
    (
        "What is the largest organ of the human body?",
        ["Heart", "Skin", "Liver", "Brain"],
        1,
    ),
]

KEY = [q[2] for q in QUESTIONS]


def format_q(i):
    q, opts, _ = QUESTIONS[i]
    return (
        f"Question {i + 1}: {q}\n"
        + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
        + "Answer:"
    )
