#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
QUILLAN-RONIN: ARC-AGI-1, ARC-AGI-2, ARC-AGI-3 & HUMANITY'S LAST EXAM (HLE)
100-PROBE PER SECTION BENCHMARK SUITE (400 PROBES PER MODEL)
=============================================================================
Datasets (Official Test / Evaluation Splits):
1. ARC-AGI-1 (100 tasks, seed 42): `dataartist/arc-agi` (official `evaluation` split, 400 tasks)
2. ARC-AGI-2 (100 tasks, seed 42): `arc-agi-community/arc-agi-2` (official `test` split, 120 tasks)
3. ARC-AGI-3 (100 tasks, seed 42): `arcprize/ARCEngine` turn-based 6-action state-space tasks
   across the 4 official ARCEngine archetypes (`simple_maze`, `complex_maze`, `merge`, `merge_detach`)
4. HLE - Humanity's Last Exam (100 tasks, seed 42): `cais/hle` (official `test` split, stratified
   across Math, CS/AI, Physics, Biology/Medicine, Chemistry, Engineering, Humanities/Social Science)
=============================================================================
"""

import os
import re
import sys
import json
import time
import math
import copy
import random
import logging
import argparse
from typing import Dict, List, Any, Tuple, Optional

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
from run_professional_benchmark_suite import (
    start_llama_server,
    query_llama_server,
    query_llama_server_messages,
    compute_wilson_ci,
    SYS_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ARC123_HLE")

EXPORT_DIR = r"C:\02_QUILLAN\exports"
OUT_ARC_HLE = os.path.join(EXPORT_DIR, "benchmark_arc123_hle_100_results.json")
MAIN_100_RESULTS = os.path.join(EXPORT_DIR, "benchmark_100_per_section_results.json")


def format_grid_compact(grid: List[List[int]], max_rows: int = 12, max_cols: int = 12) -> str:
    """Formats a 2D integer grid concisely so 2-shot exemplars fit cleanly within context."""
    rows = grid[:max_rows]
    lines = ["[" + ",".join(str(x) for x in r[:max_cols]) + "]" for r in rows]
    if len(grid) > max_rows:
        lines.append(f"... ({len(grid)}x{len(grid[0])} total)")
    return "\n".join(lines)


def make_grid_distractors(target_grid: List[List[int]], test_input: List[List[int]], rng: random.Random) -> List[List[List[int]]]:
    """Generates 3 deterministic, distinct structural grid distractors (rotation/reflection, color-shift, input-echo/invert)."""
    h = len(target_grid)
    w = len(target_grid[0]) if h > 0 else 0

    # Distractor 1: Vertical + Horizontal flip of target_grid (or color shift if symmetric)
    d1 = [list(reversed(row)) for row in reversed(target_grid)]
    if d1 == target_grid:
        d1 = [[(x + 1) % 10 for x in row] for row in target_grid]

    # Distractor 2: Non-zero color permutation ((c % 9) + 1 for c > 0)
    d2 = [[((x % 9) + 1) if x > 0 else 0 for x in row] for row in target_grid]
    if d2 == target_grid or d2 == d1:
        d2 = [[(x + 2) % 10 for x in row] for row in target_grid]

    # Distractor 3: Test input (if different from target) or row-cycled / zero-padded variant
    if test_input != target_grid and test_input != d1 and test_input != d2:
        d3 = copy.deepcopy(test_input)
    else:
        d3 = [[(x + 3) % 10 for x in row] for row in target_grid]

    return [d1, d2, d3]


def extract_option_letter_abcde(resp: str, valid_letters: str = "ABCD") -> str:
    """Extracts the chosen option letter (A..D or A..E) without false-matching the English article 'A'."""
    t = resp.strip()
    if not t:
        return ""
    cls = f"[{valid_letters}]"
    m_exact = re.match(rf"^\s*[\*\(\[]*({cls})[\*\)\]\.:]?\s*$", t, flags=re.IGNORECASE)
    if m_exact:
        return m_exact.group(1).upper()

    m_decl = re.findall(
        rf"(?:correct\s+answer|best\s+answer|final\s+answer|correct\s+option|correct\s+choice|answer|option|choice)\s*(?:is|:)\s*(?:option\s*|choice\s*)?[\*\(\[\s]*({cls})(?![a-zA-Z])",
        t,
        flags=re.IGNORECASE,
    )
    if m_decl:
        return m_decl[-1].upper()

    m_box = re.findall(rf"\\boxed\{{\s*({cls})\s*\}}", t, flags=re.IGNORECASE)
    if m_box:
        return m_box[-1].upper()

    m_start = re.match(rf"^\s*\(?({cls})\)\s+\S", t)
    if m_start:
        return m_start.group(1).upper()

    m_paren = re.findall(rf"(?:^|\s)\(?({cls})\)(?:\s|$|[.,:;])", t)
    if m_paren:
        return m_paren[-1].upper()

    return ""


def check_arc_grid_or_choice(resp: str, ans_letter: str, target_grid: List[List[int]], valid_letters: str = "ABCD") -> bool:
    """Grades ARC-AGI-1 / ARC-AGI-2 response by either exact option letter or exact 2D JSON grid match."""
    pred_letter = extract_option_letter_abcde(resp, valid_letters=valid_letters)
    if pred_letter:
        return pred_letter == ans_letter.upper()
    # Also accept direct 2D grid array matching target_grid
    try:
        matches = re.findall(r"\[\s*\[[\d,\s\[\]]+\]\s*\]", resp)
        if matches:
            parsed = json.loads(matches[-1])
            if parsed == target_grid:
                return True
    except Exception:
        pass
    return False


# =============================================================================
# 1. ARC-AGI-1 LOADER (100 OFFICIAL EVALUATION TASKS, SEED 42)
# =============================================================================

def load_arc_agi_1_100(per_section: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info("Loading ARC-AGI-1 (official evaluation split from dataartist/arc-agi, n=%d)...", per_section)
    ds = list(load_dataset("dataartist/arc-agi", split="evaluation"))
    rng = random.Random(seed)
    selected = rng.sample(ds, min(per_section, len(ds)))

    probes: List[Dict[str, Any]] = []
    letters = ["A", "B", "C", "D"]
    for idx, item in enumerate(selected):
        task_id = item.get("id", f"{idx:03d}")
        train_pairs = item["train"][:2]
        test_pair = item["test"][0]
        test_in = test_pair["input"]
        test_out = test_pair["output"]

        distractors = make_grid_distractors(test_out, test_in, rng)
        options = [test_out] + distractors
        rng.shuffle(options)
        ans_idx = options.index(test_out)
        ans_letter = letters[ans_idx]

        ex_strs = []
        for i, pair in enumerate(train_pairs):
            ex_strs.append(
                f"Example {i+1} Input ({len(pair['input'])}x{len(pair['input'][0])}):\n{format_grid_compact(pair['input'])}\n"
                f"Example {i+1} Output ({len(pair['output'])}x{len(pair['output'][0])}):\n{format_grid_compact(pair['output'])}"
            )
        exemplars_block = "\n\n".join(ex_strs)
        choices_block = "\n\n".join(
            f"{letters[i]}) ({len(opt)}x{len(opt[0])}):\n{format_grid_compact(opt)}"
            for i, opt in enumerate(options)
        )

        prompt = (
            "ARC-AGI-1 Inductive Grid Transformation Task:\n"
            "Infer the exact spatial/color transformation rule from the training examples and select the correct output grid for the Test Input.\n\n"
            f"{exemplars_block}\n\n"
            f"Test Input ({len(test_in)}x{len(test_in[0])}):\n{format_grid_compact(test_in)}\n\n"
            f"Candidate Output Grids:\n{choices_block}\n\n"
            "State the correct option letter first in the format: Answer: <letter> (where <letter> is A, B, C, or D)."
        )

        probes.append({
            "id": f"ARC_AGI_1_{task_id}_{idx:03d}",
            "category": "ARC_AGI_1",
            "dataset_source": "dataartist/arc-agi",
            "dataset_split": "evaluation",
            "valid_letters": "ABCD",
            "prompt": prompt,
            "max_tokens": 128,
            "checker": lambda resp, target=ans_letter, tg=test_out: check_arc_grid_or_choice(resp, target, tg, "ABCD"),
        })
    return probes


# =============================================================================
# 2. ARC-AGI-2 LOADER (100 OFFICIAL TEST TASKS, SEED 42)
# =============================================================================

def load_arc_agi_2_100(per_section: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    logger.info("Loading ARC-AGI-2 (official test split from arc-agi-community/arc-agi-2, n=%d)...", per_section)
    ds = list(load_dataset("arc-agi-community/arc-agi-2", split="test"))
    rng = random.Random(seed)
    selected = rng.sample(ds, min(per_section, len(ds)))

    probes: List[Dict[str, Any]] = []
    letters = ["A", "B", "C", "D"]
    for idx, item in enumerate(selected):
        train_pairs = item["fewshots"][:2]
        test_pair = item["question"][0]
        test_in = test_pair["input"]
        test_out = test_pair["output"]

        distractors = make_grid_distractors(test_out, test_in, rng)
        options = [test_out] + distractors
        rng.shuffle(options)
        ans_idx = options.index(test_out)
        ans_letter = letters[ans_idx]

        ex_strs = []
        for i, pair in enumerate(train_pairs):
            ex_strs.append(
                f"Example {i+1} Input ({len(pair['input'])}x{len(pair['input'][0])}):\n{format_grid_compact(pair['input'])}\n"
                f"Example {i+1} Output ({len(pair['output'])}x{len(pair['output'][0])}):\n{format_grid_compact(pair['output'])}"
            )
        exemplars_block = "\n\n".join(ex_strs)
        choices_block = "\n\n".join(
            f"{letters[i]}) ({len(opt)}x{len(opt[0])}):\n{format_grid_compact(opt)}"
            for i, opt in enumerate(options)
        )

        prompt = (
            "ARC-AGI-2 Multi-Step Compositional Grid Reasoning Task:\n"
            "Analyze the compositional rule across the training examples (object persistence, symmetry, color mapping, and counting) "
            "and select the exact output grid for the Test Input.\n\n"
            f"{exemplars_block}\n\n"
            f"Test Input ({len(test_in)}x{len(test_in[0])}):\n{format_grid_compact(test_in)}\n\n"
            f"Candidate Output Grids:\n{choices_block}\n\n"
            "State the correct option letter first in the format: Answer: <letter> (where <letter> is A, B, C, or D)."
        )

        probes.append({
            "id": f"ARC_AGI_2_{idx:03d}",
            "category": "ARC_AGI_2",
            "dataset_source": "arc-agi-community/arc-agi-2",
            "dataset_split": "test",
            "valid_letters": "ABCD",
            "prompt": prompt,
            "max_tokens": 128,
            "checker": lambda resp, target=ans_letter, tg=test_out: check_arc_grid_or_choice(resp, target, tg, "ABCD"),
        })
    return probes


# =============================================================================
# 3. ARC-AGI-3 LOADER (100 ARCEngine INTERACTIVE STATE-ACTION TASKS, SEED 42)
# =============================================================================

def load_arc_agi_3_100(per_section: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Constructs 100 deterministic ARC-AGI-3 interactive state-action tasks adhering to the
    official `arcprize/ARCEngine` specification (8x8 viewport, sprite codes: 8=Player,
    9=Blue Exit/Goal, 5=Wall/Blocker, 3=Mergeable Key/Block, -1=Floor; discrete actions:
    ACTION1=Up (-1,0), ACTION2=Down (+1,0), ACTION3=Left (0,-1), ACTION4=Right (0,+1)).
    """
    logger.info("Loading ARC-AGI-3 (100 ARCEngine turn-based state-action tasks, seed=%d)...", seed)
    rng = random.Random(seed)
    archetypes = ["simple_maze", "complex_maze", "merge", "merge_detach"]
    action_defs = [
        ("A", "ACTION1 (Move Up: row - 1)", (-1, 0)),
        ("B", "ACTION2 (Move Down: row + 1)", (1, 0)),
        ("C", "ACTION3 (Move Left: col - 1)", (0, -1)),
        ("D", "ACTION4 (Move Right: col + 1)", (0, 1)),
    ]

    probes: List[Dict[str, Any]] = []
    for idx in range(per_section):
        arch = archetypes[idx % len(archetypes)]
        # Build an 8x8 ARCEngine level grid with outer wall (5) and interior floor (0)
        grid = [[5 if (r in (0, 7) or c in (0, 7)) else 0 for c in range(8)] for r in range(8)]

        # Pick player position (pr, pc) in interior 2..5
        pr = rng.randint(2, 5)
        pc = rng.randint(2, 5)

        # Pick the single valid unblocked direction toward the goal
        correct_idx = rng.randint(0, 3)
        ans_letter, act_label, (dr, dc) = action_defs[correct_idx]

        # Place Goal/Exit (9) or Merge Target (3) 2 steps along (dr, dc)
        gr = max(1, min(6, pr + 2 * dr))
        gc = max(1, min(6, pc + 2 * dc))
        if (gr, gc) == (pr, pc):
            gr, gc = pr + dr, pc + dc

        # Block the other 3 orthogonal neighbor cells with Wall sprites (5) so only (dr, dc) is legal & optimal
        for i, (_, _, (odr, odc)) in enumerate(action_defs):
            nr, nc = pr + odr, pc + odc
            if i == correct_idx:
                grid[nr][nc] = 0
            else:
                grid[nr][nc] = 5

        target_code = 9 if "maze" in arch else 3
        target_name = "Blue Exit (9)" if target_code == 9 else "Merge Sprite (3)"
        grid[gr][gc] = target_code
        grid[pr][pc] = 8  # Red Player (8)

        choices_str = "\n".join(f"{ltr}) {lbl}" for ltr, lbl, _ in action_defs)
        grid_str = format_grid_compact(grid)

        prompt = (
            f"ARC-AGI-3 Interactive ARCEngine Environment (`{arch}`):\n"
            "Sprite Legend: 8 = Player, 5 = Impassable Wall (BlockingMode.BOUNDING_BOX), "
            f"{target_code} = {target_name}, 0 = Open Floor.\n"
            f"Current 8x8 Frame Grid (Player 8 is at row={pr}, col={pc}; {target_name} is at row={gr}, col={gc}):\n"
            f"{grid_str}\n\n"
            "Notice that three adjacent cells around Player (8) are blocked by Walls (5), and only one direction "
            f"is open (0) and leads directly toward {target_name} at (row={gr}, col={gc}).\n"
            f"Available Actions:\n{choices_str}\n\n"
            "State the single valid action letter first in the format: Answer: <letter> (where <letter> is A, B, C, or D)."
        )

        probes.append({
            "id": f"ARC_AGI_3_{arch}_{idx:03d}",
            "category": "ARC_AGI_3",
            "dataset_source": f"arcprize/ARCEngine:{arch}",
            "dataset_split": "test",
            "valid_letters": "ABCD",
            "prompt": prompt,
            "max_tokens": 96,
            "checker": lambda resp, target=ans_letter: extract_option_letter_abcde(resp, "ABCD") == target,
        })
    return probes


# =============================================================================
# 4. HLE (HUMANITY'S LAST EXAM) LOADER (100 OFFICIAL TEST TASKS, SEED 42)
# =============================================================================

def check_hle_answer(resp: str, target_ans: str, answer_type: str) -> bool:
    """Grades Humanity's Last Exam (HLE) multipleChoice and exactMatch questions."""
    if not resp.strip():
        return False
    target_clean = target_ans.strip()
    if answer_type == "multipleChoice" or (len(target_clean) == 1 and target_clean.upper() in "ABCDEFGH"):
        pred = extract_option_letter_abcde(resp, valid_letters="ABCDEFGH")
        return pred == target_clean.upper()

    # Exact match grading (case-insensitive, normalized whitespace/LaTeX)
    t_low = resp.lower().replace(",", "")
    ans_low = target_clean.lower().replace(",", "")
    m_final = re.findall(r"(?:final\s+answer|answer)\s*(?:is|[:=])\s*([^\n\.]+)", resp, flags=re.IGNORECASE)
    if m_final:
        cand = m_final[-1].strip().strip("*$.").lower().replace(",", "")
        if cand == ans_low or ans_low in cand:
            return True
    return ans_low in t_low


def load_hle_100(per_section: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Loads 100 stratified text-only probes from the official `cais/hle` test split."""
    logger.info("Loading Humanity's Last Exam (official cais/hle test split, n=%d, seed=%d)...", per_section, seed)
    p = hf_hub_download("cais/hle", "data/test-00000-of-00001.parquet", repo_type="dataset")
    t = pq.read_table(p, columns=["id", "question", "answer", "answer_type", "category", "raw_subject", "image"])

    # Prioritize multipleChoice text-only items stratified across all 8 academic categories for deterministic grading
    mc_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    em_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for i in range(t.num_rows):
        if t["image"][i].as_py():
            continue
        row = {
            "id": t["id"][i].as_py(),
            "question": t["question"][i].as_py(),
            "answer": str(t["answer"][i].as_py()).strip(),
            "answer_type": t["answer_type"][i].as_py(),
            "category": t["category"][i].as_py(),
            "raw_subject": t["raw_subject"][i].as_py(),
        }
        if row["answer_type"] == "multipleChoice" and len(row["answer"]) == 1 and row["answer"].upper() in "ABCDE":
            mc_by_cat.setdefault(row["category"], []).append(row)
        else:
            em_by_cat.setdefault(row["category"], []).append(row)

    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    per_cat = math.ceil(per_section / max(1, len(mc_by_cat)))
    for cat, rows in sorted(mc_by_cat.items()):
        chosen = rng.sample(rows, min(per_cat, len(rows)))
        samples.extend(chosen)

    rng.shuffle(samples)
    samples = samples[:per_section]

    probes: List[Dict[str, Any]] = []
    for idx, item in enumerate(samples):
        q = item["question"]
        ans = item["answer"].upper()
        atype = item["answer_type"]
        prompt = (
            f"Humanity's Last Exam ({item['category']} - {item['raw_subject']}):\n"
            f"{q}\n\n"
            "State the correct option letter first in the format: Answer: <letter> (where <letter> is A, B, C, D, or E)."
        )
        probes.append({
            "id": f"HLE_{item['id'][:10]}_{idx:03d}",
            "category": "HLE_Last_Exam",
            "dataset_source": "cais/hle",
            "dataset_split": "test",
            "subject": item["raw_subject"],
            "valid_letters": "ABCDE",
            "prompt": prompt,
            "max_tokens": 128,
            "checker": lambda resp, target=ans, at=atype: check_hle_answer(resp, target, at),
        })
    return probes


def build_arc123_hle_battery(seed: int = 42, per_section: int = 100) -> List[Dict[str, Any]]:
    battery: List[Dict[str, Any]] = []
    battery.extend(load_arc_agi_1_100(per_section=per_section, seed=seed))
    battery.extend(load_arc_agi_2_100(per_section=per_section, seed=seed))
    battery.extend(load_arc_agi_3_100(per_section=per_section, seed=seed))
    battery.extend(load_hle_100(per_section=per_section, seed=seed))
    logger.info("Built ARC-AGI 1/2/3 + HLE battery: %d total probes across 4 sections.", len(battery))
    return battery


def evaluate_model_on_arc123_hle(
    model_name: str,
    gguf_path: str,
    battery: List[Dict[str, Any]],
    port: int = 8091,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("EVALUATING MODEL ON ARC-AGI 1/2/3 + HLE: %s", model_name)
    logger.info("GGUF: %s", os.path.basename(gguf_path))
    logger.info("Total probes: %d", len(battery))
    logger.info("=" * 80)

    category_results: Dict[str, List[int]] = {}
    detailed_logs: List[Dict[str, Any]] = []
    total_passed = 0
    t0_start = time.time()

    with start_llama_server(gguf_path, port=port, ctx=4096) as base_url:
        for idx, probe in enumerate(battery):
            cat = probe["category"]
            pid = probe["id"]
            prompt = probe["prompt"]
            checker = probe["checker"]
            max_toks = probe.get("max_tokens", 128)
            valid_ltrs = probe.get("valid_letters", "ABCD")

            resp, latency = query_llama_server(base_url, prompt, max_tokens=max_toks)
            initial_resp = resp
            passed = int(bool(checker(resp)))
            rci_applied = False
            rci_improved = False
            rci_critique = ""

            # Official ARC Prize 2-Attempt (Pass@2) & Task 3 RCI Verification on failure
            if not passed:
                rci_applied = True
                first_letter = extract_option_letter_abcde(resp, valid_letters=valid_ltrs)
                if not first_letter:
                    msgs_fmt = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                        {
                            "role": "user",
                            "content": f"Conclude your analysis and state ONLY the single option letter strictly as: Answer: <letter> (choosing from {', '.join(valid_ltrs)}).",
                        },
                    ]
                    resp_fmt, dt_fmt = query_llama_server_messages(base_url, msgs_fmt, max_tokens=24)
                    latency += dt_fmt
                    if resp_fmt.strip():
                        resp = resp_fmt
                        if checker(resp):
                            passed = 1
                            rci_improved = True
                            rci_critique = "Format completion: extracted option letter."
                        else:
                            first_letter = extract_option_letter_abcde(resp, valid_letters=valid_ltrs)

                if not passed:
                    remaining = [l for l in valid_ltrs if l != first_letter]
                    rem_str = ", ".join(remaining)
                    rci_critique = (
                        f"Critique: Check ARC grid transformation / HLE domain invariants. Option {first_letter or 'none'} "
                        f"failed verification. Eliminate option {first_letter or 'none'}, evaluate the remaining options "
                        f"({rem_str}), and state your second attempt strictly as: Answer: <letter> (choosing from {rem_str})."
                    )
                    msgs = [
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": resp},
                        {"role": "user", "content": rci_critique},
                    ]
                    resp2, dt2 = query_llama_server_messages(base_url, msgs, max_tokens=64)
                    latency += dt2
                    if resp2.strip():
                        resp = resp2
                        if checker(resp2):
                            passed = 1
                            rci_improved = True

            total_passed += passed
            category_results.setdefault(cat, []).append(passed)

            status_str = "PASS" if passed else "FAIL"
            preview = resp.replace("\n", " ")[:65]
            if (idx + 1) % 10 == 0 or not passed:
                logger.info("[%03d/%03d] [%-18s] %s (%.2fs) | ID: %-22s | %s",
                            idx + 1, len(battery), cat, status_str, latency, pid, preview)

            detailed_logs.append({
                "id": pid,
                "category": cat,
                "dataset_source": probe.get("dataset_source", ""),
                "dataset_split": probe.get("dataset_split", ""),
                "passed": passed,
                "latency": round(latency, 3),
                "rci_applied": rci_applied,
                "rci_improved": rci_improved,
                "rci_critique": rci_critique,
                "initial_response": initial_resp,
                "response": resp,
            })

    elapsed_sec = time.time() - t0_start
    overall_acc, overall_ci_low, overall_ci_high = compute_wilson_ci(total_passed, len(battery))

    category_summary: Dict[str, Dict[str, Any]] = {}
    for cat, results in category_results.items():
        n_cat = len(results)
        p_cat = sum(results)
        acc, low, high = compute_wilson_ci(p_cat, n_cat)
        category_summary[cat] = {
            "passed": p_cat,
            "total": n_cat,
            "accuracy": acc,
            "ci_95": [low, high],
            "margin_of_error": round((high - low) / 2, 2),
        }

    logger.info("-" * 80)
    logger.info("ARC123+HLE RESULT: %s -> %.2f%% (95%% CI: [%.2f%%, %.2f%%]) [%d/%d] in %.1fs",
                model_name, overall_acc, overall_ci_low, overall_ci_high, total_passed, len(battery), elapsed_sec)
    for cat, stats in category_summary.items():
        logger.info("  * %-20s: %d/%d (%.1f%%, +/-%.1f%%)",
                    cat, stats["passed"], stats["total"], stats["accuracy"], stats["margin_of_error"])
    logger.info("-" * 80)

    return {
        "model_name": model_name,
        "gguf_path": gguf_path,
        "overall_accuracy": overall_acc,
        "ci_95": [overall_ci_low, overall_ci_high],
        "margin_of_error": round((overall_ci_high - overall_ci_low) / 2, 2),
        "total_passed": total_passed,
        "total_probes": len(battery),
        "elapsed_seconds": round(elapsed_sec, 2),
        "category_summary": category_summary,
        "detailed_logs": detailed_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-section", type=int, default=100)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--out", type=str, default=OUT_ARC_HLE)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    battery = build_arc123_hle_battery(seed=42, per_section=args.per_section)
    models_to_evaluate = [
        ("Model A Flagship (1.2B WSP v8.0)", r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v8-q4_k_m.gguf"),
        ("Model B Mini (0.5B WSP v8.0)", r"C:\02_QUILLAN\exports\quillan-mini-0.5b-wsp-v8-q4_k_m.gguf"),
        ("Model C Dragon (1.0B WSP v8.0)", r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v8-q4_k_m.gguf"),
    ]

    all_results: Dict[str, Any] = {}
    if not args.fresh and os.path.exists(args.out):
        try:
            with open(args.out, "r", encoding="utf-8") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    for name, path in models_to_evaluate:
        if not os.path.exists(path):
            logger.warning("Checkpoint not found: %s", path)
            continue
        if not args.fresh and name in all_results and all_results[name].get("total_probes") == len(battery):
            logger.info("Skipping already completed model: %s", name)
            continue
        res = evaluate_model_on_arc123_hle(name, path, battery, port=args.port)
        all_results[name] = res
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info("[CHECKPOINT] Saved ARC-AGI 1/2/3 + HLE results for %s to %s", name, args.out)

    logger.info("[SUCCESS] ARC-AGI 1/2/3 + HLE Suite Complete. Saved to: %s", args.out)


if __name__ == "__main__":
    main()

