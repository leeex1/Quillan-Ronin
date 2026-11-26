#!/usr/bin/env python3
"""FINAL held-out bank: 40 questions authored after all model selection.
Never used for gate selection, rank choice, or any edit decision. Balanced A/B/C/D.
Disjoint from fresh40/40b/40c, domain_banks, heldout_domain_large.
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]
RAW = [
    (
        "What is the largest planet in the solar system?",
        "Jupiter",
        ["Earth", "Saturn", "Neptune"],
    ),
    ("Which element has the symbol 'Au'?", "Gold", ["Silver", "Argon", "Aluminium"]),
    (
        "Who wrote 'The Hobbit'?",
        "J.R.R. Tolkien",
        ["C.S. Lewis", "George Orwell", "J.K. Rowling"],
    ),
    ("In which year did the Berlin Wall fall?", "1989", ["1987", "1991", "1985"]),
    (
        "What is the largest ocean on Earth?",
        "Pacific",
        ["Atlantic", "Indian", "Southern"],
    ),
    (
        "Which gas do plants release during photosynthesis?",
        "Oxygen",
        ["Carbon dioxide", "Nitrogen", "Methane"],
    ),
    (
        "What is the powerhouse of the cell?",
        "Mitochondrion",
        ["Nucleus", "Ribosome", "Vacuole"],
    ),
    (
        "Who was the first president of the United States?",
        "George Washington",
        ["Thomas Jefferson", "John Adams", "James Madison"],
    ),
    ("What is the square root of 169?", "13", ["12", "14", "11"]),
    ("Which element has atomic number 1?", "Hydrogen", ["Helium", "Lithium", "Oxygen"]),
    ("What is the capital of Italy?", "Rome", ["Milan", "Venice", "Florence"]),
    (
        "Which is the largest mammal?",
        "Blue whale",
        ["African elephant", "Giraffe", "Whale shark"],
    ),
    ("How many continents are there?", "7", ["5", "6", "8"]),
    ("What organ pumps blood through the body?", "Heart", ["Liver", "Lung", "Spleen"]),
    (
        "Which language has the most native speakers?",
        "Mandarin Chinese",
        ["English", "Hindi", "Spanish"],
    ),
    (
        "What is H2O commonly known as?",
        "Water",
        ["Hydrogen peroxide", "Oxygen", "Salt"],
    ),
    (
        "Who developed the theory of relativity?",
        "Albert Einstein",
        ["Isaac Newton", "Galileo Galilei", "Niels Bohr"],
    ),
    (
        "Which is the longest river in the world?",
        "Nile",
        ["Amazon", "Yangtze", "Mississippi"],
    ),
    ("What color is the sky on a clear day?", "Blue", ["Green", "Red", "Violet"]),
    ("How many legs does a spider have?", "8", ["6", "10", "12"]),
    (
        "Which planet is known as the Red Planet?",
        "Mars",
        ["Venus", "Jupiter", "Saturn"],
    ),
    ("What is the freezing point of water in Celsius?", "0", ["-10", "10", "100"]),
    (
        "Which blood cells fight infection?",
        "White blood cells",
        ["Red blood cells", "Platelets", "Plasma"],
    ),
    (
        "Who painted the Mona Lisa?",
        "Leonardo da Vinci",
        ["Michelangelo", "Raphael", "Donatello"],
    ),
    (
        "What is the largest organ of the human body?",
        "Skin",
        ["Liver", "Brain", "Heart"],
    ),
    ("Which is the smallest prime number?", "2", ["0", "1", "3"]),
    ("What is the capital of Japan?", "Tokyo", ["Kyoto", "Osaka", "Nagoya"]),
    ("Which planet has the most moons?", "Saturn", ["Jupiter", "Uranus", "Neptune"]),
    ("What is the chemical symbol for iron?", "Fe", ["Ir", "F", "Fr"]),
    ("How many bones in the adult human body?", "206", ["186", "226", "246"]),
    (
        "Which vitamin does the body make from sunlight?",
        "Vitamin D",
        ["Vitamin A", "Vitamin B12", "Vitamin C"],
    ),
    (
        "What is the most abundant gas in Earth's atmosphere?",
        "Nitrogen",
        ["Oxygen", "Argon", "Carbon dioxide"],
    ),
    ("Which animal is the fastest on land?", "Cheetah", ["Lion", "Gazelle", "Horse"]),
    (
        "What is the largest desert on Earth?",
        "Antarctica",
        ["Sahara", "Arabian", "Gobi"],
    ),
    (
        "Who composed 'The Four Seasons'?",
        "Antonio Vivaldi",
        ["Johann Bach", "Wolfgang Mozart", "Ludwig Beethoven"],
    ),
    ("In which year did World War II end?", "1945", ["1943", "1944", "1946"]),
    (
        "What is the hardest natural substance?",
        "Diamond",
        ["Quartz", "Topaz", "Corundum"],
    ),
    ("How many chambers does a human heart have?", "4", ["2", "3", "5"]),
    (
        "Which country has the largest population?",
        "India",
        ["China", "United States", "Indonesia"],
    ),
    ("What is the boiling point of water in Celsius?", "100", ["90", "95", "110"]),
]
assert len(RAW) == 40
Q, K = [], []
for i, (q, correct, dist) in enumerate(RAW):
    pos = i % 4
    opts = [None] * 4
    di = 0
    opts[pos] = correct
    for s in range(4):
        if opts[s] is None:
            opts[s] = dist[di]
            di += 1
    Q.append((q, opts, pos))
    K.append(pos)
assert [K.count(j) for j in range(4)] == [10, 10, 10, 10]
QUESTIONS = Q
KEY = K


def prompts(tok):
    return [
        tok.encode(
            f"Question {i + 1}: {q}\n"
            + "".join(f"{L}) {o}\n" for L, o in zip(LETTERS, opts))
            + "Answer:",
            add_special_tokens=False,
        )
        for i, (q, opts, _) in enumerate(Q)
    ]
