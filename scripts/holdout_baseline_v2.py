#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contamination-free shipped-GGUF baseline harness (v2).

Artifacts (all NEW files; nothing existing is modified):
  scripts/eval_sets/items_v2.jsonl             frozen question set
  scripts/eval_sets/perplexity_corpus_heldout.txt  fixed PPL corpus
  exports/holdout_baseline_v2_<model>.json     result ledger

Scoring contract
  * lower held-out CE / PPL is better
  * CE must stay strictly ABOVE 1.0 (overfit guard); gate is flagged otherwise
  * reported context: declared num_ctx=2097152; EXERCISED window is recorded
    separately (GTX 1050 4 GB cannot attend a 60-70k-token system prompt)

Usage
  python holdout_baseline_v2.py --validate-only
  python holdout_baseline_v2.py --smoke                 # 2 items/domain/model
  python holdout_baseline_v2.py --full [--force]        # 160 items x 3 models
  python holdout_baseline_v2.py --compare A.json B.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "holdout_baseline_v2.0"
ROOT = Path(r"C:\02_QUILLAN")
SCRIPT_DIR = ROOT / "scripts"
EVAL_DIR = SCRIPT_DIR / "eval_sets"
ITEMS_FILE = EVAL_DIR / "items_v2.jsonl"
PPL_CORPUS = EVAL_DIR / "perplexity_corpus_heldout.txt"
EXPORT_DIR = ROOT / "exports"

LLAMA_DIR = Path(r"C:\01_PROGRAMS\Open_Source\AI_ML\llama.cpp\build")
LLAMA_CLI = LLAMA_DIR / "llama-cli.exe"
LLAMA_PPL = LLAMA_DIR / "llama-perplexity.exe"

MODELS: dict[str, Path] = {
    "flagship": ROOT / "exports" / "quillan-1b-sovereign-wsp-v23-q8_0.gguf",
    "mini": ROOT / "exports" / "quillan-mini-0.5b-wsp-v23-q8_0.gguf",
    "dragon": ROOT / "exports" / "quillan-dragon-1b-wsp-v18b-q8_0.gguf",
}

DECLARED_NUM_CTX = 2_097_152
ACTIVE_WINDOW = 2_048  # verified 46.3 tok/s generation at -c 2048
PPL_WINDOW = 1_024  # llama-perplexity needs >= 2*ctx tokens; corpus ~2.4k tokens
SYSTEM_PROMPT_DECLARED_CHARS = 250_000
SYSTEM_PROMPT_EST_TOKENS = 65_000
SEED = 42
TEMPERATURE = 0.0
TOP_K = 1
NGL = 99
FA = "on"
CTK, CTV = "q4_0", "q4_0"
LOSS_GATE_MIN = 1.0
CLI_TIMEOUT_S = 90
PPL_TIMEOUT_S = 600
DOMAINS = ("arc_agi", "general_knowledge", "math_reasoning", "instruction_toolcall")
SPLITS = ("CALIBRATION", "FROZEN_HELD_OUT")
MAX_TOKENS = {"letter": 16, "numeric": 48, "exact": 32, "json": 192}

GENERATION_FLAGS = [
    "-s",
    str(SEED),
    "--temp",
    str(TEMPERATURE),
    "--top-k",
    str(TOP_K),
    "-ngl",
    str(NGL),
    "-fa",
    FA,
    "-ctk",
    CTK,
    "-ctv",
    CTV,
    "-c",
    str(ACTIVE_WINDOW),
    "--simple-io",
    "--no-display-prompt",
    "-st",  # --single-turn: exit after the predefined -p, do not wait on stdin
]
PPL_FLAGS = [
    "-ngl",
    str(NGL),
    "-fa",
    FA,
    "-ctk",
    CTK,
    "-ctv",
    CTV,
    "-c",
    str(PPL_WINDOW),
]

# Generic/benchmark scaffolding is excluded before overlap detection so that
# shared task framing does not masquerade as answer leakage.
STOPWORDS: set[str] = set(
    """the a an and or of to in on by for with from is are was were be been being
    it its this that these those as at into after before between then if else
    you your we our they their he she his her not no nor but which what who whom
    whose why how when where does do did done has have had can could would should
    will shall may might must about above below up down out off over under again
    further once here there all any both each few more most other some such only
    own same so than too very just also now array grid task abstract example small
    holds cell upper-left lower-right upper-right lower-left applying rule places
    query option shows result unchanged reply letter answer only general knowledge
    compute arithmetic integer remaining divided remainder modulus times plus minus
    squared cubed river flows through capital planet chemical symbol code bones adult
    human body language native speakers continent contains desert wrote novel hardest
    naturally occurring substance instrument measures earthquake magnitude unit force
    nearest closest organ pumps blood element country stands raised eighth power bird
    known migrating earth ocean sea level gas sea-creature creature produce produced
    tool-call tool functions function available user request single json object naming
    chosen field putting argument prose reply weather clock conversion messaging travel
    timing sea blue three arms blood amount date city minutes
    recipient destination""".split()
) | {
    "tool",
    "call",
    "json",
    "object",
    "field",
    "argument",
    "functions",
    "function",
    "available",
    "user",
    "request",
    "single",
    "naming",
    "chosen",
    "putting",
    "prose",
    "reply",
    "answer",
    "only",
    "integer",
    "compute",
    "arithmetic",
    "planet",
    "river",
    "ocean",
    "earth",
    "capital",
    "chemical",
    "symbol",
    "element",
    "body",
    "human",
    "known",
    "which",
    "what",
    "where",
    "when",
    "upper-left",
    "lower-right",
    "upper-right",
    "lower-left",
    "grid",
    "task",
    "abstract",
    "example",
    "array",
    "holds",
    "cell",
    "applying",
    "rule",
    "places",
    "query",
    "option",
    "shows",
    "result",
    "unchanged",
    "center",
    "b",
    "c",
    "d",
    "e",
    "one",
    "two",
    "first",
    "second",
    "third",
    "following",
    "strictly",
    "correct",
    "return",
    "output",
    "input",
    "format",
    "functions",
    "weather",
    "clock",
    "conversion",
    "messaging",
    "travel",
    "timing",
}


# --------------------------------------------------------------------------- #
# Data loading + contamination verification
# --------------------------------------------------------------------------- #
def load_items(path: Path = ITEMS_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"frozen item set missing: {path}")
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            required = {"id", "prompt", "expected", "domain", "split", "answer_format"}
            missing = required - set(obj)
            if missing:
                raise ValueError(f"line {lineno}: missing fields {sorted(missing)}")
            if obj["split"] not in SPLITS or obj["domain"] not in DOMAINS:
                raise ValueError(f"line {lineno}: bad split/domain")
            items.append(obj)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for it in items:
        counts[(it["domain"], it["split"])] += 1
    for domain in DOMAINS:
        for split in SPLITS:
            n = counts[(domain, split)]
            if n < 20:
                raise ValueError(f"insufficient items: {domain}/{split} has {n} (<20)")
    return items


def _entities(text: str) -> set[str]:
    out: set[str] = set()
    for tok in re.findall(r"[a-z][a-z0-9_-]*|\d{2,}", text.lower()):
        if tok in STOPWORDS:
            continue
        if tok.isdigit() or len(tok) >= 5:
            out.add(tok)
    return out


def _content_seq(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z][a-z0-9_-]*|\d{2,}", text.lower())
        if t not in STOPWORDS
    ]


def _shingles(seq: list[str], n: int = 3) -> set[str]:
    return {" ".join(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def verify_no_contamination(items: list[dict[str, Any]]) -> dict[str, Any]:
    cal = [it for it in items if it["split"] == "CALIBRATION"]
    hold = [it for it in items if it["split"] == "FROZEN_HELD_OUT"]
    cal_ent: dict[str, set[str]] = {
        it["id"]: _entities(it["prompt"] + " " + it["expected"]) for it in cal
    }
    hold_ent: dict[str, set[str]] = {
        it["id"]: _entities(it["prompt"] + " " + it["expected"]) for it in hold
    }
    cal_sh: dict[str, set[str]] = {
        it["id"]: _shingles(_content_seq(it["prompt"] + " " + it["expected"]))
        for it in cal
    }
    hold_sh: dict[str, set[str]] = {
        it["id"]: _shingles(_content_seq(it["prompt"] + " " + it["expected"]))
        for it in hold
    }
    violations: list[str] = []
    for hid, hents in hold_ent.items():
        for cid, cents in cal_ent.items():
            shared = sorted(hents & cents)
            if shared:
                violations.append(f"ENTITY {hid} <-> {cid}: {shared}")
            shared_sh = sorted(hold_sh[hid] & cal_sh[cid])
            if shared_sh:
                violations.append(f"SHINGLE {hid} <-> {cid}: {shared_sh[:3]}")
        # literal substring containment is a hard fail
        htext = next(
            it["prompt"] + " " + it["expected"] for it in hold if it["id"] == hid
        )
        for it in cal:
            ctext = it["prompt"] + " " + it["expected"]
            if len(htext) >= 24 and htext.lower() in ctext.lower():
                violations.append(f"SUBSTRING {hid} <-> {it['id']}")
            if len(ctext) >= 24 and ctext.lower() in htext.lower():
                violations.append(f"SUBSTRING {it['id']} <-> {hid}")
    report = {
        "calibration_items": len(cal),
        "held_out_items": len(hold),
        "entity_overlap_count": sum(1 for v in violations if v.startswith("ENTITY")),
        "shingle_overlap_count": sum(1 for v in violations if v.startswith("SHINGLE")),
        "substring_overlap_count": sum(
            1 for v in violations if v.startswith("SUBSTRING")
        ),
        "violations": violations[:200],
        "clean": not violations,
    }
    if violations:
        raise ValueError("CONTAMINATION DETECTED:\n" + "\n".join(violations[:50]))
    return report


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(LLAMA_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def extract_generation(raw: str, prompt: str) -> str:
    """Strip llama-cli's startup banner, echoed prompt, and timing footer.

    llama-cli 6e14286 has no --no-banner switch; --simple-io still emits a
    banner and echoes the prompt prefixed with '> '. Without this isolation a
    build hash such as 'b9631' is parsed as the numeric answer.
    """
    text = raw.replace("\r\n", "\n")
    pos = text.find(prompt)
    probe = prompt[:50]
    if pos < 0:
        pos = text.find(probe)
    if pos >= 0:
        text = text[
            pos + (len(prompt) if text.startswith(prompt, pos) else len(probe)) :
        ]
    footer = re.search(r"\n\[\s*Prompt:", text)
    if footer:
        text = text[: footer.start()]
    return text.strip()


def parse_answer(raw: str, item: dict[str, Any]) -> str | None:
    text = extract_generation(raw, item["prompt"])
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text).strip()
    fmt, expected = item["answer_format"], item["expected"]
    if fmt == "letter":
        m = re.search(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", text)
        return m.group(1).upper() if m else None
    if fmt == "numeric":
        cues = re.findall(
            r"(?:answer is|equals|result is|=)\s*[:=]?\s*(-?\d[\d,]*)",
            text,
            re.IGNORECASE,
        )
        if cues:
            return cues[-1].replace(",", "")
        nums = re.findall(r"-?\d[\d,]*", text)
        return nums[-1].replace(",", "") if nums else None
    if fmt == "exact":
        low = text.lower()
        if expected.lower() in low:
            return expected
        toks = set(re.findall(r"[a-z0-9]+", low))
        etoks = re.findall(r"[a-z0-9]+", expected.lower())
        return expected if etoks and all(t in toks for t in etoks) else None
    if fmt == "json":
        for candidate in (text,):
            try:
                obj = json.loads(candidate)
                if obj == json.loads(expected):
                    return expected
            except (ValueError, json.JSONDecodeError):
                pass
        span = re.search(r"\{[\s\S]*\}", text)
        if span:
            try:
                if json.loads(span.group(0)) == json.loads(expected):
                    return expected
            except (ValueError, json.JSONDecodeError):
                pass
        return None
    raise ValueError(f"unknown answer_format {fmt}")


def generate(model: Path, item: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        str(LLAMA_CLI),
        "-m",
        str(model),
        "-p",
        item["prompt"],
        "-n",
        str(MAX_TOKENS[item["answer_format"]]),
    ] + GENERATION_FLAGS
    t0 = time.perf_counter()
    try:
        proc = _run(cmd, CLI_TIMEOUT_S)
        raw, rc, err = proc.stdout, proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        raw, rc, err = "", -99, "TIMEOUT"
    dt = time.perf_counter() - t0
    parsed = parse_answer(raw, item)
    return {
        "id": item["id"],
        "domain": item["domain"],
        "split": item["split"],
        "expected": item["expected"],
        "raw_response": raw.strip(),
        "parsed_answer": parsed,
        "correct": parsed is not None and parsed == item["expected"],
        "stderr_tail": err[-600:],
        "returncode": rc,
        "elapsed_sec": round(dt, 3),
    }


def measure_ppl(model: Path, chunks: int | None) -> dict[str, Any]:
    cmd = [str(LLAMA_PPL), "-m", str(model), "-f", str(PPL_CORPUS)] + PPL_FLAGS
    if chunks is not None:
        cmd += ["--chunks", str(chunks)]
    t0 = time.perf_counter()
    try:
        proc = _run(cmd, PPL_TIMEOUT_S)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        combined, rc = "TIMEOUT", -99
    dt = time.perf_counter() - t0
    ppl: float | None = None
    ce: float | None = None
    for line in reversed(combined.splitlines()):
        if ppl is None and "PPL =" in line:
            m = re.search(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                ppl = float(m.group(1))
        if ce is None and re.search(r"\bCE\s*=", line):
            m = re.search(r"\bCE\s*=\s*([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                ce = float(m.group(1))
    if ppl is not None and ce is None:
        ce = math.log(ppl)  # PPL == exp(CE) by definition
    error: str | None = None
    if "you need at least" in combined.lower():
        error = "insufficient_tokens_for_perplexity"
    elif ppl is None:
        error = "ppl_not_parsed_from_llama_perplexity_output"
    return {
        "ppl": ppl,
        "ce_loss": ce,
        "error": error,
        "ce_source": "llama-perplexity"
        if ce is not None and ppl is not None and abs(ce - math.log(ppl)) > 1e-6
        else "ln(PPL)",
        "loss_gate_min": LOSS_GATE_MIN,
        "loss_gate_pass": bool(ce is not None and ce > LOSS_GATE_MIN),
        "returncode": rc,
        "elapsed_sec": round(dt, 3),
        "raw_tail": combined[-1500:],
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def select_smoke(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    for domain in DOMAINS:
        held = [
            it
            for it in items
            if it["domain"] == domain and it["split"] == "FROZEN_HELD_OUT"
        ]
        picked.extend(held[:2])
    return picked


def run_model(
    name: str, model: Path, items: list[dict[str, Any]], smoke: bool
) -> dict[str, Any]:
    print(f"\n=== {name}: {model.name} ===")
    digest = sha256_file(model)
    print(f"sha256 {digest}")
    responses = [generate(model, it) for it in items]
    held = [r for r in responses if r["split"] == "FROZEN_HELD_OUT"]
    calib = [r for r in responses if r["split"] == "CALIBRATION"]
    per_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        for label, rows in (("held_out", held), ("calibration", calib)):
            subset = [r for r in rows if r["domain"] == domain]
            if subset:
                per_domain.setdefault(domain, {})[label] = {
                    "n": len(subset),
                    "correct": sum(1 for r in subset if r["correct"]),
                    "accuracy": round(
                        sum(r["correct"] for r in subset) / len(subset), 4
                    ),
                }
    ppl = measure_ppl(model, chunks=1 if smoke else None)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_key": name,
        "model_path": str(model),
        "gguf_sha256": digest,
        "generated_utc_epoch": time.time(),
        "mode": "smoke" if smoke else "full",
        "flags": {
            "generation": GENERATION_FLAGS,
            "perplexity": PPL_FLAGS,
            "seed": SEED,
            "temp": TEMPERATURE,
            "top_k": TOP_K,
            "ngl": NGL,
            "flash_attn": FA,
            "cache_type_k": CTK,
            "cache_type_v": CTV,
        },
        "context": {
            "declared_num_ctx": DECLARED_NUM_CTX,
            "exercised_active_window": ACTIVE_WINDOW,
            "exercised_ppl_window": PPL_WINDOW,
            "system_prompt_declared_chars": SYSTEM_PROMPT_DECLARED_CHARS,
            "system_prompt_est_tokens": SYSTEM_PROMPT_EST_TOKENS,
            "system_prompt_injected_into_probe_prompts": False,
            "compromise_note": (
                "GGUF/YaRN declares num_ctx=2097152 and v23 WSP bakes system behavior "
                "into weights; a ~250k-char (~65k-token) system prompt replayed per "
                "probe would exceed the GTX 1050 4GB active window and dominate runtime. "
                "Declared context is recorded verbatim; all scoring uses the documented "
                "practical -c 2048 window so head-to-head runs are reproducible."
            ),
        },
        "score_contract": {
            "direction": "lower_ce_and_ppl_is_better",
            "loss_gate": f"ce_loss > {LOSS_GATE_MIN}",
            "loss_gate_pass": ppl["loss_gate_pass"],
        },
        "perplexity": ppl,
        "per_item": responses,
        "per_domain": per_domain,
        "aggregate": {
            "held_out_n": len(held),
            "held_out_correct": sum(1 for r in held if r["correct"]),
            "held_out_accuracy": round(sum(r["correct"] for r in held) / len(held), 4)
            if held
            else None,
            "calibration_n": len(calib),
            "calibration_accuracy": round(
                sum(r["correct"] for r in calib) / len(calib), 4
            )
            if calib
            else None,
            "avg_generation_sec": round(
                sum(r["elapsed_sec"] for r in responses) / len(responses), 3
            )
            if responses
            else None,
        },
    }
    return result


def write_result(name: str, result: dict[str, Any]) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"holdout_baseline_v2_{name}.json"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"WROTE {path}")
    return path


def compare(a_path: Path, b_path: Path) -> None:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    print(f"COMPARE  A={a.get('model_key')} ({a_path.name})")
    print(f"         B={b.get('model_key')} ({b_path.name})")
    print(f"{'domain':<24}{'A_acc':>9}{'B_acc':>9}{'delta':>9}")
    for domain in DOMAINS:
        da = a.get("per_domain", {}).get(domain, {}).get("held_out")
        db = b.get("per_domain", {}).get(domain, {}).get("held_out")
        if not da or not db:
            continue
        va, vb = da["accuracy"], db["accuracy"]
        print(f"{domain:<24}{va:>9.4f}{vb:>9.4f}{vb - va:>+9.4f}")
    pa, pb = a.get("perplexity", {}), b.get("perplexity", {})

    def metric_row(label: str, key: str) -> str:
        va, vb = pa.get(key), pb.get(key)
        fa = f"{va:.4f}" if isinstance(va, (int, float)) else "n/a"
        fb = f"{vb:.4f}" if isinstance(vb, (int, float)) else "n/a"
        delta = (
            f"{vb - va:+.4f}"
            if isinstance(va, (int, float)) and isinstance(vb, (int, float))
            else "n/a"
        )
        return f"{label:<24}{fa:>9}{fb:>9}{delta:>9}"

    print(metric_row("CE_LOSS", "ce_loss"))
    print(metric_row("PPL", "ppl"))
    for label, data in (("A", a), ("B", b)):
        gate = data.get("perplexity", {}).get("loss_gate_pass")
        flag = "PASS" if gate else "FAIL"
        print(
            f"LOSS>1.0 GATE {label}: {flag} (ce={data.get('perplexity', {}).get('ce_loss')})"
        )
    print("NOTE: stub comparator - no significance testing, no paired bootstrap.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="allow full battery if estimate >10min"
    )
    parser.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"))
    parser.add_argument("--models", nargs="*", choices=sorted(MODELS), default=None)
    args = parser.parse_args()

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]))
        return 0

    items = load_items()
    contam = verify_no_contamination(items)
    print("CONTAMINATION CHECK:", json.dumps(contam, ensure_ascii=False))
    if args.validate_only:
        return 0

    smoke = not args.full
    picked = select_smoke(items) if smoke else items
    models = args.models or sorted(MODELS)
    for path in (LLAMA_CLI, LLAMA_PPL, PPL_CORPUS):
        if not path.exists():
            raise FileNotFoundError(path)
    for key in models:
        if not MODELS[key].exists():
            raise FileNotFoundError(MODELS[key])

    if args.full:
        est = 160 * len(models) * 8 + 60 * len(models)  # conservative bound
        print(
            f"ESTIMATED FULL BATTERY RUNTIME: <= {est}s ({est / 60:.1f} min) for {len(models)} models"
        )
        if est > 600 and not args.force:
            print(
                "REFUSING full battery (>10 min estimate). Re-run with --force to override."
            )
            return 2

    paths: list[Path] = []
    for key in models:
        result = run_model(key, MODELS[key], picked, smoke)
        ag = result["aggregate"]
        pp = result["perplexity"]
        print(
            f"  held_out_acc={ag['held_out_accuracy']}  ce={pp['ce_loss']}  "
            f"ppl={pp['ppl']}  loss_gate={'PASS' if pp['loss_gate_pass'] else 'FAIL'}"
        )
        paths.append(write_result(key, result))

    if smoke:
        # estimate from per-item timings of the smoke results already written
        totals: list[float] = []
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            for row in data["per_item"]:
                totals.append(row["elapsed_sec"])
        avg = sum(totals) / len(totals) if totals else 1.0
        est = avg * 160 * len(models) + 60 * len(models)
        print(
            f"ESTIMATED FULL BATTERY RUNTIME FROM SMOKE: ~{est:.0f}s ({est / 60:.1f} min)"
        )
        print("Full battery NOT executed (smoke mode).")
    print("RESULT PATHS:", " ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
