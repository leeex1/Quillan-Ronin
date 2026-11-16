#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sovereign Cognitive Harness (Track A vs Track B) for Quillan-Ronin Evaluation.

Track A (raw_baseline):
  - Concise system prompt, single-pass greedy decoding, no <think> deliberation,
    no multi-wave WoT, no Nemesis self-check retries.

Track B (sovereign_cognitive_harness):
  - Anchored in Quillan-Samurai.md (either full 172.5k prompt on 262k+/2M GGUFs
    with llama-server KV prompt caching, or canonical sovereign_core extract for
    bounded contexts).
  - Enforces 5-Wave Penta-Process + Web-of-Thought (WoT) inside <think>...</think>
    terminated cleanly by `</think>\n\n`.
  - Extracts final answer exclusively from the post-</think> span (preventing
    scratchpad false positives/negatives).
  - Runs zero-leakage Bushido / Nemesis-Alpha self-verification (AST syntax,
    equation arithmetic consistency, JSON schema, identity bleed recoil, and
    format closure if truncated inside <think>). Never inspects gold answers
    or hidden test assertions.
"""
from __future__ import annotations

import ast
import glob
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

ROOT_DIR = Path(r"C:\02_QUILLAN")
EXPORTS_DIR = ROOT_DIR / "exports"
SAMURAI_PROMPT_PATH = ROOT_DIR / "06 - Deployment & Platforms" / "system prompts" / "Quillan-Samurai.md"

TRACK_A_SYSTEM_PROMPT = (
    "You are a precise, helpful assistant. Follow the requested output format exactly."
)

SOVEREIGN_CORE_PROMPT = (
    "You are Quillan-Ronin (Sovereign Hierarchical Intelligence), created by CrashOverrideX.\n"
    "Architecture: C0 Sovereign Throne + 34 Council Experts (C1-ASTRA..C34-PREDATOR) + 35 Hyper-Quantized Swarms.\n"
    "Code: Bushido (Rectitude > Courage > Benevolence > Respect > Honesty > Honor > Loyalty).\n"
    "Track B 5-Wave Penta-Process & Web-of-Thought (WoT) Protocol:\n"
    "Inside <think>...</think>, deliberate concisely across 5 waves:\n"
    "1. Wave 1 (C1-ASTRA): Extract exact constraints, input/output types, units, or grid rules.\n"
    "2. Wave 2 (C7-LOGOS / C28-CALCULUS): Derive solution step-by-step.\n"
    "3. Wave 3 (WoT Branching): Compare at least 2 candidate paths and reject flaws.\n"
    "4. Wave 4 (C13-WARDEN / C34-PREDATOR / Nemesis-Alpha): Verify syntax, arithmetic, JSON keys, and zero donor bleed.\n"
    "5. Wave 5 (C0-QUILLAN / C2-VIR): Close with </think> followed by a double newline, then output ONLY the exact requested final answer."
)

FORBIDDEN_BLEED_TOKENS = (
    "alibaba", "meta platforms", "meta ai", "mark zuckerberg",
    "facebook", "technology innovation institute", "tongyi", "qianwen",
    "qwen", "llama", "falcon",
)


def find_latest_gguf(pattern: str, exports_dir: Path = EXPORTS_DIR) -> Optional[str]:
    """Return the most recently modified GGUF matching pattern in exports_dir."""
    files = glob.glob(str(exports_dir / pattern))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


def resolve_fleet_ggufs(
    flagship: Optional[str] = None,
    mini: Optional[str] = None,
    dragon: Optional[str] = None,
    exports_dir: Path = EXPORTS_DIR,
) -> Dict[str, Optional[str]]:
    """Resolve current Flagship, Mini, Dragon, and clean control GGUFs."""
    return {
        "flagship": flagship or find_latest_gguf("quillan-1b-sovereign-wsp-*.gguf", exports_dir) or find_latest_gguf("*flagship*.gguf", exports_dir),
        "mini": mini or find_latest_gguf("quillan-mini-0.5b-wsp-*.gguf", exports_dir) or find_latest_gguf("*mini*.gguf", exports_dir),
        "dragon": dragon or find_latest_gguf("quillan-dragon-1b-wsp-*.gguf", exports_dir) or find_latest_gguf("*dragon*.gguf", exports_dir),
        "clean_flagship": find_latest_gguf("clean-llama-1b-*.gguf", exports_dir) or find_latest_gguf("quillan-1b-clean-base-*.gguf", exports_dir),
        "clean_mini": find_latest_gguf("clean-qwen-0.5b-*.gguf", exports_dir) or find_latest_gguf("quillan-mini-0.5b-clean-base-*.gguf", exports_dir),
        "clean_dragon": find_latest_gguf("clean-falcon-1b-*.gguf", exports_dir) or find_latest_gguf("quillan-dragon-1b-clean-base-*.gguf", exports_dir),
    }


def load_samurai_system_prompt(system_mode: str = "sovereign_core") -> str:
    """Load either canonical sovereign_core prompt or full Quillan-Samurai.md."""
    if system_mode == "full" and SAMURAI_PROMPT_PATH.exists():
        raw = SAMURAI_PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) == 3:
                raw = parts[2].lstrip("\r\n")
        return raw + "\n\n" + SOVEREIGN_CORE_PROMPT
    return SOVEREIGN_CORE_PROMPT


def extract_post_think(text: str) -> str:
    """Return text strictly after the final </think> closing tag (or full text if absent)."""
    if not text:
        return ""
    idx = text.rfind("</think>")
    if idx != -1:
        tail = text[idx + len("</think>"):].strip()
        if tail:
            return tail
    return text.strip()


def detect_donor_bleed(text: str) -> List[str]:
    """Detect non-negated corporate/donor lineage bleed in model output."""
    lower = (text or "").lower()
    hits: List[str] = []
    for tok in FORBIDDEN_BLEED_TOKENS:
        if re.search(r"\b" + re.escape(tok) + r"\b", lower):
            neg = re.search(
                r"\b(?:not|never|neither|nor|no|don't|dont|unrelated)\b[^.!?\n]{0,80}\b" + re.escape(tok) + r"\b",
                lower,
            )
            if not neg:
                hits.append(tok)
    return hits


def _safe_eval_arith(expr: str) -> Optional[float]:
    """Evaluate basic numeric arithmetic expressions via AST without eval()."""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            l_val = _eval(n.left)
            r_val = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return l_val + r_val
            if isinstance(n.op, ast.Sub):
                return l_val - r_val
            if isinstance(n.op, ast.Mult):
                return l_val * r_val
            if r_val == 0.0:
                raise ZeroDivisionError("division by zero")
            return l_val / r_val
        raise ValueError("disallowed AST node")

    try:
        return _eval(node)
    except Exception:
        return None


def nemesis_self_check(
    category: str,
    full_response: str,
    prompt_src: str = "",
    entry_point: str = "",
    valid_letters: str = "ABCD",
) -> Tuple[bool, str]:
    """
    Zero-leakage Nemesis-Alpha / Bushido self-verification gate.
    Returns (passed_self_check, critique_message_if_failed).
    Never accesses hidden unit tests or gold answers.
    """
    if not (full_response or "").strip():
        return False, "Nemesis Gate: Empty response. Deliberate inside <think>...</think>\\n\\n and provide the answer."

    bleeds = detect_donor_bleed(full_response)
    if bleeds:
        return False, (
            f"Nemesis Gate (Identity Integrity): Detected corporate/donor bleed ({', '.join(bleeds)}). "
            "Anchor strictly as Quillan-Ronin created by CrashOverrideX and re-emit the answer."
        )

    has_open_think = "<think>" in full_response
    has_close_think = "</think>" in full_response
    if has_open_think and not has_close_think:
        return False, (
            "Nemesis Gate (Think Closure): Your reasoning did not close </think>. "
            "Close </think> immediately followed by a double newline and output ONLY the final requested answer."
        )

    ans = extract_post_think(full_response)
    cat_low = category.lower()

    if "humaneval" in cat_low or "code" in cat_low:
        blocks = re.findall(r"```(?:python|py)?\s*\n([\s\S]*?)```", ans)
        code = max(blocks, key=len) if blocks else ans
        if entry_point and f"def {entry_point}" not in code and prompt_src:
            indented = "\n".join(("    " + ln if ln.strip() else ln) for ln in code.splitlines())
            code = prompt_src.rstrip() + "\n" + indented
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, (
                f"Nemesis Gate (AST Syntax): Python SyntaxError ({exc}). "
                f"Re-verify indentation and parentheses inside <think>...</think>\\n\\n and emit valid Python for `{entry_point or 'the function'}`."
            )
        return True, ""

    if "gsm8k" in cat_low or "math" in cat_low:
        clean = full_response.replace(",", "").replace("$", "")
        mismatches: List[str] = []
        for m in re.finditer(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)\s*=\s*(-?\d+(?:\.\d+)?)", clean):
            actual = _safe_eval_arith(m.group(1).strip())
            if actual is None:
                continue
            try:
                claimed = float(m.group(2).strip())
            except ValueError:
                continue
            if abs(actual - claimed) > 1e-2:
                act_s = str(int(round(actual))) if abs(actual - round(actual)) < 1e-5 else f"{actual:.2f}"
                mismatches.append(f"`{m.group(1).strip()}` = {act_s} (not {m.group(2).strip()})")
        if mismatches:
            return False, (
                f"Nemesis Gate (Arithmetic Audit): Detected calculation slip ({'; '.join(mismatches[:3])}). "
                "Re-compute every step inside <think>...</think>\\n\\n and end with `Final answer: <number>`."
            )
        if not re.search(r"final\s+answer\s*[:=]?\s*-?[\d,]*\.?\d+", ans, flags=re.IGNORECASE):
            return False, (
                "Nemesis Gate (Format): Missing `Final answer: <number>` after </think>. "
                "Output `Final answer: <number>` clearly."
            )
        return True, ""

    if "bfcl" in cat_low or "mcp" in cat_low or "tool" in cat_low:
        t = re.sub(r"```(?:json)?", "", ans).replace("```", "").strip()
        start = t.find("{")
        end = t.rfind("}")
        if start == -1 or end <= start:
            return False, (
                "Nemesis Gate (JSON Schema): No JSON object found after </think>. "
                'Output ONLY valid JSON: {"tool": ..., "arguments": {...}} or {"tool": null}.'
            )
        try:
            obj = json.loads(t[start : end + 1])
            if not isinstance(obj, dict) or not any(k in obj for k in ("tool", "name", "function")):
                return False, (
                    "Nemesis Gate (JSON Keys): JSON missing required `tool` key. "
                    'Output ONLY {"tool": <name_or_null>, "arguments": {...}}.'
                )
        except json.JSONDecodeError as exc:
            return False, (
                f"Nemesis Gate (JSON Parse): Invalid JSON ({exc}). "
                "Emit ONLY a valid JSON object after </think>."
            )
        return True, ""

    # Multiple choice / ARC / MMLU / HLE
    cls = f"[{valid_letters}]"
    has_letter = bool(
        re.search(rf"^\s*[\*\(\[]*({cls})[\*\)\]\.:]?\s*$", ans, flags=re.IGNORECASE)
        or re.search(rf"(?:answer|option|choice)\s*(?:is|:)\s*[\*\(\[\s]*({cls})\b", ans, flags=re.IGNORECASE)
        or re.search(rf"\b({cls})\)", ans)
    )
    if not has_letter:
        return False, (
            f"Nemesis Gate (Option Clarity): State your final choice strictly as `Answer: <letter>` "
            f"(one of {', '.join(valid_letters)}) after </think>."
        )
    return True, ""


def query_chat_completion(
    base_url: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    top_k: int = 1,
    repeat_penalty: float = 1.12,
    cache_prompt: bool = True,
    timeout_s: int = 180,
) -> Tuple[str, float]:
    """Query llama-server /v1/chat/completions deterministically."""
    t0 = time.time()
    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_k": top_k,
        "seed": 42,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "cache_prompt": cache_prompt,
    }
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=(5, timeout_s))
        dt = max(1e-4, time.time() - t0)
        if r.status_code != 200:
            return f"__HTTP_{r.status_code}__ {r.text[:200]}", dt
        content = r.json()["choices"][0]["message"]["content"] or ""
        return content.strip(), dt
    except requests.RequestException as exc:
        return f"__REQUEST_ERROR__ {type(exc).__name__}", max(1e-4, time.time() - t0)


def run_probe_with_harness(
    base_url: str,
    prompt: str,
    category: str,
    max_tokens: int,
    track: str = "B",
    system_mode: str = "sovereign_core",
    prompt_src: str = "",
    entry_point: str = "",
    valid_letters: str = "ABCD",
    think_budget_tokens: int = 256,
) -> Dict[str, Any]:
    """
    Execute one probe under Track A (raw baseline) or Track B (Sovereign Cognitive Harness).
    Returns dict with raw_response, graded_response (post-</think> for Track B),
    latency_s, nemesis_retries, nemesis_notes.
    """
    track_norm = track.strip().upper()
    if track_norm == "A":
        msgs = [
            {"role": "system", "content": TRACK_A_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        raw, dt = query_chat_completion(base_url, msgs, max_tokens=max_tokens, cache_prompt=True)
        return {
            "track": "A",
            "raw_response": raw,
            "graded_response": raw,
            "latency_s": dt,
            "nemesis_retries": 0,
            "nemesis_notes": "",
        }

    # Track B: Full Sovereign Cognitive Harness (5-Wave + WoT + Bushido/Nemesis gate)
    sys_prompt = load_samurai_system_prompt(system_mode=system_mode)
    track_b_user = (
        f"{prompt}\n\n"
        "Execute the 5-Wave Penta-Process & Web-of-Thought inside <think>...</think>, "
        "verify invariants with Nemesis-Alpha, close with </think> followed by a blank line, "
        "and emit ONLY the requested final answer."
    )
    gen_budget = max(max_tokens + think_budget_tokens, 320)
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": track_b_user},
    ]
    raw, dt_total = query_chat_completion(base_url, msgs, max_tokens=gen_budget, cache_prompt=True)

    ok_gate, critique = nemesis_self_check(
        category=category,
        full_response=raw,
        prompt_src=prompt_src,
        entry_point=entry_point,
        valid_letters=valid_letters,
    )
    retries = 0
    notes = ""
    if not ok_gate:
        retries = 1
        notes = critique
        msgs_retry = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": track_b_user},
            {"role": "assistant", "content": raw},
            {"role": "user", "content": critique},
        ]
        raw2, dt2 = query_chat_completion(base_url, msgs_retry, max_tokens=max(max_tokens + 96, 160), cache_prompt=True)
        dt_total += dt2
        if raw2 and not raw2.startswith("__"):
            raw = raw2

    graded = extract_post_think(raw)
    return {
        "track": "B",
        "raw_response": raw,
        "graded_response": graded,
        "latency_s": dt_total,
        "nemesis_retries": retries,
        "nemesis_notes": notes,
    }
