#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministically materialize the FROZEN v2 evaluation item set.

Writes ``items_v2.jsonl`` next to this file. Run once; the emitted JSONL is the
frozen artifact consumed by ``holdout_baseline_v2.py``. Regeneration is
byte-identical for a given Python version because ordering is explicit.

Design invariant: every CALIBRATION item uses a vocabulary pool that is
disjoint from every FROZEN_HELD_OUT item (content words, two-plus-digit numbers,
tool names, and expected values). The harness re-verifies this independently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT: Path = Path(__file__).with_name("items_v2.jsonl")


def emit(items: list[dict[str, Any]]) -> None:
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(items)} items -> {OUT}")


def mk(
    iid: str, domain: str, split: str, prompt: str, expected: str, fmt: str
) -> dict[str, Any]:
    return {
        "id": iid,
        "domain": domain,
        "split": split,
        "prompt": prompt,
        "expected": expected,
        "answer_format": fmt,
    }


def build_arc(items: list[dict[str, Any]]) -> None:
    cal_colors = ["amber", "indigo", "violet", "crimson"]
    cal_shapes = ["circle", "diamond", "triangle", "star"]
    cal_rules = ["copy-right", "swap-rows", "fill-center"]
    hold_colors = ["teal", "ochre", "magenta", "coral"]
    hold_shapes = ["hexagon", "trapezoid", "crescent", "oblong"]
    hold_rules = ["quarter-turn", "invert-fill", "shift-down"]
    letters = ["A", "B", "C", "D"]
    for split, colors, shapes, rules, prefix in (
        ("CALIBRATION", cal_colors, cal_shapes, cal_rules, "ARC_CAL"),
        ("FROZEN_HELD_OUT", hold_colors, hold_shapes, hold_rules, "ARC_HOLD"),
    ):
        for i in range(20):
            color = colors[i % 4]
            shape = shapes[(i * 3) % 4]
            rule = rules[i % 3]
            prompt = (
                f"Abstract grid task. Example: a small array holds a {color} {shape} "
                f"in the upper-left cell; applying rule '{rule}' places it in the "
                f"lower-right cell. Query: the same {color} {shape} is now in the "
                "upper-right cell. Which option shows the result? "
                "A) lower-left  B) lower-right  C) center  D) unchanged. "
                "Reply with one letter."
            )
            items.append(
                mk(
                    f"{prefix}_{i + 1:02d}",
                    "arc_agi",
                    split,
                    prompt,
                    letters[i % 4],
                    "letter",
                )
            )


GLOBAL_CAL = [
    ("Which river flows through Cairo?", "Nile"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Which gas do plants absorb during photosynthesis?", "carbon"),
    ("Which ocean lies west of Africa?", "Atlantic"),
    ("What is the largest mammal species?", "whale"),
    ("Which planet is nearest to the Sun?", "Mercury"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many bones are in the adult human body?", "206"),
    ("Which language has the most native speakers?", "Mandarin"),
    ("Which continent contains the Sahara desert?", "Africa"),
    ("Who wrote the novel Pride and Prejudice?", "Austen"),
    ("What is the hardest naturally occurring substance?", "diamond"),
    ("Which instrument measures earthquake magnitude?", "seismograph"),
    ("What is the SI unit of force?", "newton"),
    ("Which star is closest to Earth?", "Sun"),
    ("What organ pumps blood through the body?", "heart"),
    ("Which element has the symbol Na?", "sodium"),
    ("In which country does the Eiffel Tower stand?", "France"),
    ("What is two raised to the eighth power?", "256"),
    ("Which bird is known for migrating to Egypt in spring?", "swallow"),
]

GLOBAL_HOLD = [
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for potassium?", "K"),
    ("Which small moon orbits Jupiter closely?", "Io"),
    ("Which organ covers the entire body surface?", "skin"),
    ("Which gas makes up most of the atmosphere?", "nitrogen"),
    ("What is the approximate speed of light in vacuum, in km per second?", "300000"),
    ("Which element has atomic number ten?", "neon"),
    ("What is the capital of Australia?", "Canberra"),
    ("Which river flows through Paris?", "Seine"),
    ("Which hormone regulates blood sugar levels?", "insulin"),
    ("What is the tallest mountain above sea level?", "Everest"),
    ("Which planet displays prominent rings of ice?", "Saturn"),
    ("What is the chemical formula of table salt?", "NaCl"),
    ("Which bone is the longest in the human body?", "femur"),
    ("What is the positive square root of 144?", "12"),
    ("Which ocean is the deepest on Earth?", "Pacific"),
    ("Which sweet substance is made by honeybees?", "honey"),
    ("Which vitamin is produced in skin by sunlight?", "D"),
    ("What is the boiling point of water in Fahrenheit?", "212"),
    ("Which sea creature has eight arms and blue blood?", "octopus"),
]


def build_global(items: list[dict[str, Any]]) -> None:
    for split, rows, prefix in (
        ("CALIBRATION", GLOBAL_CAL, "GK_CAL"),
        ("FROZEN_HELD_OUT", GLOBAL_HOLD, "GK_HOLD"),
    ):
        for i, (question, answer) in enumerate(rows, 1):
            prompt = f"General knowledge. {question} Reply with the answer only."
            items.append(
                mk(
                    f"{prefix}_{i:02d}",
                    "general_knowledge",
                    split,
                    prompt,
                    answer,
                    "exact",
                )
            )


MATH_CAL = [
    (17, "+", 23, 40),
    (31, "-", 9, 22),
    (23, "*", 2, 46),
    (64, "/", 4, 16),
    (17, "+", 31, 48),
    (88, "-", 23, 65),
    (42, "+", 17, 59),
    (75, "-", 31, 44),
    (56, "+", 23, 79),
    (93, "-", 42, 51),
    (23, "*", 3, 69),
    (64, "+", 17, 81),
    (88, "/", 2, 44),
    (17, "*", 2, 34),
    (75, "+", 23, 98),
    (93, "-", 31, 62),
    (42, "+", 23, 65),
    (56, "-", 17, 39),
    (31, "+", 31, 62),
    (88, "+", 17, 105),
]
MATH_HOLD = [
    ("What is the remainder when 67 is divided by 9?", 4),
    ("Compute 41 plus 67.", 108),
    ("What is 73 minus 41?", 32),
    ("Compute the modulus: 89 mod 8.", 1),
    ("What is seven squared?", 49),
    ("What is five cubed?", 125),
    ("Compute 41 times 2.", 82),
    ("What is 67 plus 73?", 140),
    ("What is 89 minus 52?", 37),
    ("What is 73 minus 47?", 26),
    ("What is 89 minus 32?", 57),
    ("What is 73 plus 18?", 91),
    ("What is 89 plus 29?", 118),
    ("What is 73 plus 60?", 133),
    ("What is 41 minus 14?", 27),
    ("What is 73 minus 18?", 55),
    ("What is 41 plus 33?", 74),
    ("What is 67 plus 30?", 97),
    ("What is 73 plus 38?", 111),
    ("What is 67 plus 83?", 150),
]


def build_math(items: list[dict[str, Any]]) -> None:
    for i, (a, op, b, res) in enumerate(MATH_CAL, 1):
        prompt = f"Arithmetic. Compute {a} {op} {b}. Reply with the integer only."
        items.append(
            mk(
                f"MATH_CAL_{i:02d}",
                "math_reasoning",
                "CALIBRATION",
                prompt,
                str(res),
                "numeric",
            )
        )
    for i, (question, res) in enumerate(MATH_HOLD, 1):
        prompt = f"Arithmetic. {question} Reply with the integer only."
        items.append(
            mk(
                f"MATH_HOLD_{i:02d}",
                "math_reasoning",
                "FROZEN_HELD_OUT",
                prompt,
                str(res),
                "numeric",
            )
        )


TOOL_CAL = [
    ("get_weather", "city", "Oslo"),
    ("lookup_time", "date", "Friday"),
    ("convert_units", "amount", "Lima"),
]
TOOL_HOLD = [
    ("send_message", "recipient", "Marseille"),
    ("book_flight", "destination", "Kyoto"),
    ("set_timer", "minutes", "Zurich"),
]


def build_tool(items: list[dict[str, Any]]) -> None:
    for split, rows, prefix, opening in (
        ("CALIBRATION", TOOL_CAL, "TOOL_CAL", "weather, clock, and conversion"),
        ("FROZEN_HELD_OUT", TOOL_HOLD, "TOOL_HOLD", "messaging, travel, and timing"),
    ):
        for i in range(20):
            fn, key, val = rows[i % 3]
            available = ", ".join(f"{f}({k})" for f, k, _ in rows)
            prompt = (
                f"Tool-call task. Available {opening} functions: {available}. "
                f'User request: "{val}". Reply with a single JSON object naming '
                f"the chosen function in field tool and putting the argument in "
                f"field {key}. No prose."
            )
            expected = json.dumps({"tool": fn, key: val}, separators=(",", ":"))
            items.append(
                mk(
                    f"{prefix}_{i:02d}",
                    "instruction_toolcall",
                    split,
                    prompt,
                    expected,
                    "json",
                )
            )


def main() -> None:
    items: list[dict[str, Any]] = []
    build_arc(items)
    build_global(items)
    build_math(items)
    build_tool(items)
    assert len(items) == 160, len(items)
    emit(items)


if __name__ == "__main__":
    main()
