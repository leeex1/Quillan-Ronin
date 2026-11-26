#!/usr/bin/env python3
"""Larger pre-locked held-out bank (20/domain x 4 = 80), disjoint from the
10/domain domain_banks.py and the fresh40/40b/40c banks. Used to confirm that
glassbox section edits generalize beyond the small gates used to accept them.
Balanced A/B/C/D (5 per letter per domain).
"""

from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

SCIENCE = [
    ("What is the chemical symbol for silver?", "Ag", ["Au", "Al", "S"]),
    (
        "Which planet has the most moons?",
        "Saturn",
        ["Venus", "Jupiter", "Saturn"],
    ),
    ("What organ pumps blood?", "Heart", ["Liver", "Lung", "Kidney"]),
    (
        "What gas is essential for human respiration?",
        "Oxygen",
        ["Nitrogen", "Helium", "Argon"],
    ),
    ("What is the largest organ of the body?", "Skin", ["Liver", "Brain", "Heart"]),
    ("Which element has symbol 'K'?", "Potassium", ["Krypton", "Kalium", "Kelvin"]),
    ("How many bones in the adult human body?", "206", ["106", "306", "406"]),
    ("What is the pH of pure water?", "7", ["5", "6", "9"]),
    (
        "Which particle has a negative charge?",
        "Electron",
        ["Proton", "Neutron", "Photon"],
    ),
    ("What is the speed of light in km/s?", "300000", ["30000", "3000000", "150000"]),
    (
        "Which vitamin does sunlight produce?",
        "Vitamin D",
        ["Vitamin A", "Vitamin C", "Vitamin K"],
    ),
    ("What is the chemical formula for water?", "H2O", ["CO2", "O2", "NaCl"]),
    (
        "Which is the largest cell in the human body?",
        "Oocyte",
        ["Neuron", "Red blood cell", "Sperm"],
    ),
    (
        "What force keeps planets in orbit?",
        "Gravity",
        ["Magnetism", "Friction", "Tension"],
    ),
    ("How many chambers does a human heart have?", "4", ["2", "3", "5"]),
    (
        "What is the study of liquids called?",
        "Hydrology",
        ["Biology", "Geology", "Optics"],
    ),
    (
        "Which metal is liquid at room temperature?",
        "Mercury",
        ["Iron", "Gold", "Silver"],
    ),
    (
        "What is the hardest natural substance?",
        "Diamond",
        ["Quartz", "Granite", "Topaz"],
    ),
    (
        "Which gland regulates metabolism?",
        "Thyroid",
        ["Adrenal", "Pituitary", "Pancreas"],
    ),
    (
        "What is the most abundant element in the universe?",
        "Hydrogen",
        ["Helium", "Oxygen", "Carbon"],
    ),
]
HISTORY = [
    (
        "Who wrote '1984'?",
        "George Orwell",
        ["Aldous Huxley", "Ray Bradbury", "Arthur Koestler"],
    ),
    ("In which year did WWII begin?", "1939", ["1937", "1938", "1940"]),
    (
        "Who was the first emperor of Rome?",
        "Augustus",
        ["Julius Caesar", "Nero", "Tiberius"],
    ),
    ("Which civilization built Machu Picchu?", "Inca", ["Aztec", "Maya", "Olmec"]),
    (
        "Who led the Civil Rights Movement in the US?",
        "Martin Luther King Jr.",
        ["Malcolm X", "Rosa Parks", "Thurgood Marshall"],
    ),
    ("In which year did the Soviet Union dissolve?", "1991", ["1989", "1990", "1992"]),
    (
        "Who painted the Sistine Chapel ceiling?",
        "Michelangelo",
        ["Raphael", "Leonardo", "Botticelli"],
    ),
    (
        "Which war was fought 1775-1783?",
        "American Revolutionary",
        ["Civil", "World War I", "Korean"],
    ),
    (
        "Who was the first woman Nobel laureate?",
        "Marie Curie",
        ["Rosalind Franklin", "Ada Lovelace", "Grace Hopper"],
    ),
    (
        "Which ancient city was buried by Vesuvius?",
        "Pompeii",
        ["Troy", "Atlantis", "Carthage"],
    ),
    ("Who wrote 'The Republic'?", "Plato", ["Aristotle", "Socrates", "Homer"]),
    (
        "In which century did the French Revolution occur?",
        "18th",
        ["17th", "19th", "20th"],
    ),
    (
        "Who was the first man to walk on the Moon?",
        "Neil Armstrong",
        ["Buzz Aldrin", "Michael Collins", "Yuri Gagarin"],
    ),
    (
        "Which empire had its capital at Constantinople?",
        "Byzantine",
        ["Roman", "Ottoman", "Persian"],
    ),
    ("Who composed 'The Four Seasons'?", "Vivaldi", ["Bach", "Mozart", "Beethoven"]),
    ("Which treaty ended WWI?", "Versailles", ["Paris", "Berlin", "Geneva"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Pasteur", "Jenner", "Koch"]),
    (
        "Which president signed the Emancipation Proclamation?",
        "Abraham Lincoln",
        ["George Washington", "Thomas Jefferson", "Andrew Jackson"],
    ),
    ("In which year did the Titanic sink?", "1912", ["1910", "1911", "1914"]),
    (
        "Who wrote 'Wuthering Heights'?",
        "Emily Bronte",
        ["Charles Dickens", "George Eliot", "Emily Bronte"],
    ),
]
GEOGRAPHY = [
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    (
        "Which is the largest desert on Earth?",
        "Antarctic",
        ["Sahara", "Arabian", "Gobi"],
    ),
    ("What is the longest river in Africa?", "Nile", ["Congo", "Zambezi", "Niger"]),
    ("Which country has the most people?", "India", ["China", "USA", "Indonesia"]),
    (
        "What is the capital of Brazil?",
        "Brasilia",
        ["Rio de Janeiro", "Sao Paulo", "Salvador"],
    ),
    ("Which ocean is the deepest?", "Pacific", ["Atlantic", "Indian", "Arctic"]),
    (
        "What is the highest mountain on Earth?",
        "Mount Everest",
        ["K2", "Kangchenjunga", "Makalu"],
    ),
    (
        "Which continent is Greenland part of?",
        "North America",
        ["Europe", "Asia", "Antarctica"],
    ),
    ("What is the capital of Germany?", "Berlin", ["Munich", "Frankfurt", "Hamburg"]),
    (
        "Which country is home to the Great Barrier Reef?",
        "Australia",
        ["New Zealand", "Indonesia", "Fiji"],
    ),
    (
        "What is the smallest country by area?",
        "Vatican City",
        ["Monaco", "San Marino", "Liechtenstein"],
    ),
    (
        "Which sea is between Europe and Africa?",
        "Mediterranean",
        ["Baltic", "Black", "Red"],
    ),
    ("What is the capital of South Korea?", "Seoul", ["Busan", "Pyongyang", "Incheon"]),
    ("Which river flows through Cairo?", "Nile", ["Seine", "Thames", "Danube"]),
    (
        "What is the largest island in Asia?",
        "Borneo",
        ["New Guinea", "Borneo", "Madagascar"],
    ),
    ("Which country has the most time zones?", "France", ["Russia", "USA", "China"]),
    (
        "What is the capital of Argentina?",
        "Buenos Aires",
        ["Cordoba", "Rosario", "Mendoza"],
    ),
    (
        "Which mountain range separates Europe and Asia?",
        "Ural",
        ["Alps", "Himalaya", "Andes"],
    ),
    (
        "What is the largest lake in Africa?",
        "Lake Victoria",
        ["Lake Tanganyika", "Lake Malawi", "Lake Chad"],
    ),
    ("Which planet has rings?", "Saturn", ["Mars", "Venus", "Mercury"]),
]
MATH = [
    ("What is 6 times 7?", "42", ["36", "48", "40"]),
    ("What is 144 divided by 12?", "12", ["11", "14", "10"]),
    ("What is 5 squared?", "25", ["15", "20", "35"]),
    ("What is 20 percent of 150?", "30", ["25", "35", "40"]),
    ("What is the next prime after 7?", "11", ["9", "10", "13"]),
    ("What is the sum of angles in a triangle?", "180", ["90", "360", "270"]),
    ("What is 3 to the power of 4?", "81", ["64", "27", "100"]),
    ("What is half of 90?", "45", ["40", "50", "35"]),
    ("What is 8 plus 15?", "23", ["22", "24", "21"]),
    ("What is the square root of 81?", "9", ["8", "10", "7"]),
    ("What is 100 minus 37?", "63", ["73", "67", "53"]),
    ("What is 9 times 9?", "81", ["72", "89", "90"]),
    ("What is 1/4 as a decimal?", "0.25", ["0.5", "0.75", "0.4"]),
    ("What is the perimeter of a square with side 4?", "16", ["8", "12", "20"]),
    ("What is 12 times 3?", "36", ["32", "39", "42"]),
    ("What is 50 divided by 5?", "10", ["5", "15", "25"]),
    ("What is 7 plus 8?", "15", ["14", "16", "13"]),
    ("What is 2 cubed?", "8", ["6", "4", "9"]),
    ("What is 30 percent of 200?", "60", ["50", "70", "80"]),
    ("What is the median of 1,3,5,7,9?", "5", ["4", "6", "7"]),
]

RAW = {"science": SCIENCE, "history": HISTORY, "geography": GEOGRAPHY, "math": MATH}
BANKS = {}
for name, rows in RAW.items():
    assert len(rows) == 20, f"{name} has {len(rows)}"
    Q, K = [], []
    for i, (q, correct, dist) in enumerate(rows):
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
    BANKS[name] = (Q, K)


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
