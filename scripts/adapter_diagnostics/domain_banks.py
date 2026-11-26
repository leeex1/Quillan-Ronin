#!/usr/bin/env python3
"""Domain-scoped evaluation banks for the surgical freeze protocol.

Each bank tests a KNOWLEDGE DOMAIN, so a domain edit can be measured on its
own domain (positive), a sibling domain (transfer), and a protected baseline
(regression guard). All hand-verifiable, balanced A/B/C/D.
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

# SCIENCE domain
SCIENCE = [
    ("What is the chemical symbol for gold?", "Au", ["Go", "Gd", "Ag"]),
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Earth", "Mars"]),
    (
        "What is the powerhouse of the cell?",
        "Mitochondrion",
        ["Nucleus", "Ribosome", "Golgi"],
    ),
    (
        "Which gas do plants absorb?",
        "Carbon dioxide",
        ["Oxygen", "Nitrogen", "Hydrogen"],
    ),
    (
        "What is H2O commonly known as?",
        "Water",
        ["Oxygen", "Hydrogen peroxide", "Salt"],
    ),
    ("Which element has atomic number 1?", "Hydrogen", ["Helium", "Carbon", "Oxygen"]),
    (
        "Who developed the theory of relativity?",
        "Einstein",
        ["Newton", "Galileo", "Kepler"],
    ),
    (
        "What is the largest organ of the human body?",
        "Skin",
        ["Heart", "Liver", "Brain"],
    ),
    ("How many legs does a spider have?", "8", ["6", "10", "12"]),
    ("What is the freezing point of water in Celsius?", "0", ["-10", "10", "100"]),
]
# HISTORY domain
HISTORY = [
    (
        "Who wrote 'Pride and Prejudice'?",
        "Jane Austen",
        ["Charles Dickens", "George Eliot", "Emily Bronte"],
    ),
    ("In which year did World War II end?", "1945", ["1943", "1944", "1946"]),
    (
        "Who was the first president of the United States?",
        "George Washington",
        ["Thomas Jefferson", "John Adams", "James Madison"],
    ),
    (
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci",
        ["Michelangelo", "Raphael", "Donatello"],
    ),
    ("In which year did the Berlin Wall fall?", "1989", ["1987", "1988", "1990"]),
    (
        "Who was the first person to walk on the Moon?",
        "Neil Armstrong",
        ["Buzz Aldrin", "Michael Collins", "Yuri Gagarin"],
    ),
    ("Which empire built the Colosseum?", "Roman", ["Greek", "Egyptian", "Persian"]),
    (
        "Who wrote 'The Great Gatsby'?",
        "F. Scott Fitzgerald",
        ["Hemingway", "Steinbeck", "Faulkner"],
    ),
    ("In which year did WWI begin?", "1914", ["1912", "1913", "1915"]),
    (
        "Who developed the polio vaccine?",
        "Jonas Salk",
        ["Fleming", "Jenner", "Pasteur"],
    ),
]
# GEOGRAPHY domain
GEOGRAPHY = [
    (
        "What is the largest ocean on Earth?",
        "Pacific",
        ["Atlantic", "Indian", "Arctic"],
    ),
    ("What is the capital of Japan?", "Tokyo", ["Kyoto", "Osaka", "Nagoya"]),
    (
        "Which is the largest mammal?",
        "Blue whale",
        ["Elephant", "Giraffe", "Whale shark"],
    ),
    ("How many continents are there?", "7", ["5", "6", "8"]),
    (
        "Which language has the most native speakers?",
        "Mandarin Chinese",
        ["English", "Hindi", "Spanish"],
    ),
    (
        "Which is the longest river in the world?",
        "Nile",
        ["Amazon", "Yangtze", "Mississippi"],
    ),
    ("What is the capital of France?", "Paris", ["Lyon", "Marseille", "Toulouse"]),
    (
        "Which planet is known as the Red Planet?",
        "Mars",
        ["Venus", "Jupiter", "Saturn"],
    ),
    (
        "What is the largest island in the world?",
        "Greenland",
        ["New Guinea", "Borneo", "Madagascar"],
    ),
    (
        "Which desert covers most of North Africa?",
        "Sahara",
        ["Gobi", "Kalahari", "Atacama"],
    ),
]
# MATH domain
MATH = [
    ("What is the square root of 144?", "12", ["11", "13", "14"]),
    ("What is 7 times 8?", "56", ["48", "54", "64"]),
    ("What is 9 multiplied by 6?", "54", ["48", "56", "64"]),
    ("What is the smallest prime number?", "2", ["0", "1", "3"]),
    ("How many sides does a hexagon have?", "6", ["5", "7", "8"]),
    ("What is 12 divided by 4?", "3", ["2", "4", "6"]),
    ("What is 15 percent of 200?", "30", ["25", "35", "40"]),
    ("What is the area of a square with side 5?", "25", ["10", "20", "30"]),
    ("How many degrees in a right angle?", "90", ["45", "180", "360"]),
    ("What is 2 to the power of 8?", "256", ["128", "512", "64"]),
]


def build(raw):
    Q, K = [], []
    for i, (q, correct, distractors) in enumerate(raw):
        pos = i % 4
        opts, di = [None] * 4, 0
        opts[pos] = correct
        for slot in range(4):
            if opts[slot] is None:
                opts[slot] = distractors[di]
                di += 1
        Q.append((q, opts, pos))
        K.append(pos)
    return Q, K


BANKS = {
    name: build(raw)
    for name, raw in [
        ("science", SCIENCE),
        ("history", HISTORY),
        ("geography", GEOGRAPHY),
        ("math", MATH),
    ]
}


def prompts(tok, bank):
    Q, _ = BANKS[bank]
    return [
        tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        for i, (q, opts, _) in enumerate(Q)
    ]
