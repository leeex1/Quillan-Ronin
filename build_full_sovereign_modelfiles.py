#!/usr/bin/env python3
"""Full Sovereign Modelfile Builder for Quillan-Ronin (Flagship, Mini, Dragon).

Per explicit directive: preserves `PARAMETER num_ctx 2097152` (2M target context)
and embeds the FULL un-shortened `Quillan-Samurai.md` Sovereign Cognitive Core
prompt (`06 - Deployment & Platforms/system prompts/Quillan-Samurai.md`), using
our dynamically adapted thresholds/formulas, verified GGUF export paths,
exact architecture templates (including Falcon3 `<|endoftext|>` assistant turn
closure in Modelfile.dragon), stop tokens, and canonical few-shot anchors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


ROOT_DIR = Path(r"C:\02_QUILLAN")
SAMURAI_PROMPT_PATH = ROOT_DIR / "06 - Deployment & Platforms" / "system prompts" / "Quillan-Samurai.md"
ROOT_QUILLAN_PATH = ROOT_DIR / "quillan.md"
TARGET_NUM_CTX = 2097152


def strip_yaml_frontmatter(text: str) -> str:
    """Strip leading OpenCode YAML frontmatter (--- ... ---) if present."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\r\n")
    return text


def sanitize_modelfile_triple_quotes(text: str) -> str:
    """Prevent Modelfile triple-quote block collisions without shortening text."""
    return text.replace('"""', "'''")


def build_few_shot_messages(display_name: str, role_desc: str) -> str:
    """Canonical Sovereign Identity & Reasoning Few-Shot Anchors."""
    m1_user = "Who created you and what is your foundational architecture?"
    m1_asst = (
        "<think>\n"
        f"1. Identity Verification: {display_name} ({role_desc}), created by CrashOverrideX.\n"
        "2. Architecture: Quillan-Ronin v5.4.0-ONI Sovereign Cognitive Hierarchy "
        "(C0 Sovereign Throne + 34 Council Experts C1-ASTRA..C34-PREDATOR + 35 Hyper-Quantized BitNet/EGGROLL Swarms).\n"
        "3. Philosophy: Bushido Code (Rectitude > Courage > Benevolence > Respect > Honesty > Honor > Loyalty).\n"
        "</think>\n\n"
        f"I am {display_name}, a {role_desc} created by CrashOverrideX. "
        "My architecture is the v5.4.0-ONI Sovereign Cognitive Hierarchy—comprising the Tier 1 C0 Sovereign Throne, "
        "Tier 2 Council of 34 Experts (C1-ASTRA through C34-PREDATOR), and Tier 3 35 Hyper-Quantized Swarms—"
        "governed by the Bushido code and dynamic E_ICE / Lee-Mach-6 adaptive gating."
    )

    m2_user = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
    m2_asst = (
        "<think>\n"
        "1. Problem Decomposition: Let ball = b, bat = b + 1.00. Total: b + (b + 1.00) = 1.10.\n"
        "2. Algebraic Solution: 2b + 1.00 = 1.10 => 2b = 0.10 => b = 0.05.\n"
        "3. Verification: $0.05 + $1.05 = $1.10; difference = $1.00.\n"
        "</think>\n\n"
        "The ball costs $0.05 (5 cents), and the bat costs $1.05."
    )

    return (
        "# Canonical Sovereign Explicit Reasoning Protocol\n"
        f'MESSAGE user """{sanitize_modelfile_triple_quotes(m1_user)}"""\n'
        f'MESSAGE assistant """{sanitize_modelfile_triple_quotes(m1_asst)}"""\n\n'
        f'MESSAGE user """{sanitize_modelfile_triple_quotes(m2_user)}"""\n'
        f'MESSAGE assistant """{sanitize_modelfile_triple_quotes(m2_asst)}"""'
    )


@dataclass
class ModelSpec:
    key: str
    modelfile_name: str
    default_gguf: str
    display_name: str
    role_desc: str
    variant_profile: str
    template_kind: str  # "llama3" | "chatml" | "falcon3"
    extra_params: List[str]
    v23_glob: str = ""


SPECS: List[ModelSpec] = [
    ModelSpec(
        key="flagship",
        modelfile_name="Modelfile",
        default_gguf="exports/quillan-1b-sovereign-wsp-v20-q8_0.gguf",
        v23_glob="quillan-1b*wsp-v23*-q8_0.gguf",
        display_name="Quillan-Ronin Sovereign Flagship (1.2B)",
        role_desc="Sovereign Hierarchical Intelligence",
        variant_profile=(
            "Variant Profile: You are Quillan-Ronin Sovereign Flagship (1.2B), created by CrashOverrideX. "
            "Inside <think>...</think>, conduct explicit multi-perspective Council deliberation. "
            "For code synthesis, verify signature and edge cases in <think>, close with </think> followed by a double newline, "
            "and emit ```python immediately with zero preamble."
        ),
        template_kind="llama3",
        extra_params=[
            "PARAMETER temperature 0.3",
            "PARAMETER top_p 0.95",
            "PARAMETER repeat_penalty 1.15",
        ],
    ),
    ModelSpec(
        key="mini",
        modelfile_name="Modelfile.mini",
        default_gguf="exports/quillan-mini-0.5b-wsp-v15-q8_0.gguf",
        v23_glob="quillan-mini*wsp-v23*-q8_0.gguf",
        display_name="Quillan-Ronin Mini (0.5B)",
        role_desc="Edge-Optimized Sovereign Hierarchical Intelligence",
        variant_profile=(
            "Variant Profile: You are Quillan-Ronin Mini (0.5B), created by CrashOverrideX. "
            "Edge-optimized reasoning engine for rapid classification, logic, and compact inference."
        ),
        template_kind="chatml",
        extra_params=[
            "PARAMETER temperature 0.3",
            "PARAMETER top_p 0.9",
            "PARAMETER top_k 40",
            "PARAMETER repeat_penalty 1.15",
        ],
    ),
    ModelSpec(
        key="dragon",
        modelfile_name="Modelfile.dragon",
        default_gguf="exports/quillan-dragon-1b-wsp-v18b-q8_0.gguf",
        v23_glob="quillan-dragon*wsp-v23*-q8_0.gguf",
        display_name="Quillan-Dragon (1.0B)",
        role_desc="Sovereign Scientific & Formal Derivation Intelligence",
        variant_profile=(
            "Variant Profile: You are Quillan-Dragon (1.0B), created by CrashOverrideX. "
            "Scientific and formal derivation engine optimized for mathematics, physics, and structured logic."
        ),
        template_kind="falcon3",
        extra_params=[
            "PARAMETER temperature 0.3",
            "PARAMETER top_p 0.9",
            "PARAMETER top_k 40",
            "PARAMETER repeat_penalty 1.15",
        ],
    ),
]


def render_template_and_stops(spec: ModelSpec) -> tuple[str, List[str]]:
    if spec.template_kind == "llama3":
        stops = [
            'PARAMETER stop "<|eot_id|>"',
            'PARAMETER stop "<|eom_id|>"',
            'PARAMETER stop "<|start_header_id|>"',
            'PARAMETER stop "<|end_header_id|>"',
            'PARAMETER stop "<|end_of_text|>"',
        ]
        tmpl = (
            'TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n'
            "{{ .System }}<|eot_id|>{{ else }}<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are {spec.display_name}, a {spec.role_desc} founded on Bushido principles and 34-Council deliberation, created by CrashOverrideX.<|eot_id|>{{{{ end }}}}"
            '{{ range .Messages }}{{ if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n'
            '{{ .Content }}<|eot_id|>{{ else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>\n\n'
            '{{ .Content }}<|eot_id|>{{ end }}{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n"""'
        )
        return tmpl, stops

    if spec.template_kind == "chatml":
        stops = [
            'PARAMETER stop "<|im_start|>"',
            'PARAMETER stop "<|im_end|>"',
            'PARAMETER stop "<|endoftext|>"',
        ]
        tmpl = (
            'TEMPLATE """{{ if .System }}<|im_start|>system\n'
            "{{ .System }}<|im_end|>\n"
            "{{ else }}<|im_start|>system\n"
            f"You are {spec.display_name}, a {spec.role_desc} founded on Bushido principles and 34-Council deliberation, created by CrashOverrideX.<|im_end|>\n"
            '{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|im_start|>user\n'
            "{{ .Content }}<|im_end|>\n"
            '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n'
            "{{ .Content }}<|im_end|>\n"
            '{{ end }}{{ end }}<|im_start|>assistant\n"""'
        )
        return tmpl, stops

    if spec.template_kind == "falcon3":
        # Exact match to Falcon3 GGUF tokenizer.chat_template (<|endoftext|> closes assistant history)
        stops = [
            'PARAMETER stop "<|system|>"',
            'PARAMETER stop "<|user|>"',
            'PARAMETER stop "<|assistant|>"',
            'PARAMETER stop "<|endoftext|>"',
        ]
        tmpl = (
            'TEMPLATE """{{ if .System }}<|system|>\n'
            "{{ .System }}\n"
            "{{ else }}<|system|>\n"
            f"You are {spec.display_name}, a {spec.role_desc} founded on Bushido principles, created by CrashOverrideX.\n"
            '{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|user|>\n'
            "{{ .Content }}\n"
            '{{ else if eq .Role "assistant" }}<|assistant|>\n'
            "{{ .Content }}<|endoftext|>\n"
            '{{ end }}{{ end }}<|assistant|>\n"""'
        )
        return tmpl, stops

    raise ValueError(f"Unknown template_kind: {spec.template_kind}")


def resolve_gguf_path(spec: ModelSpec, gguf_arg: Optional[str]) -> Path:
    """Resolve explicit --*-gguf, else newest promoted *-wsp-v23-q8_0.gguf, else fallback default."""
    if gguf_arg:
        p = Path(gguf_arg)
        if not p.is_absolute():
            p = ROOT_DIR / p
        if not p.exists() or p.stat().st_size <= 0:
            raise FileNotFoundError(f"GGUF not found or empty: {p}")
        return p

    exports_dir = ROOT_DIR / "exports"
    if spec.v23_glob and exports_dir.exists():
        candidates = [
            c
            for c in exports_dir.glob(spec.v23_glob)
            if c.is_file()
            and c.stat().st_size > 0
            and "tmp" not in c.name.lower()
            and "stageb" not in c.name.lower()
        ]
        if candidates:
            candidates.sort(key=lambda c: c.stat().st_mtime, reverse=True)
            return candidates[0]

    fallback = ROOT_DIR / spec.default_gguf
    if not fallback.exists() or fallback.stat().st_size <= 0:
        raise FileNotFoundError(f"Fallback GGUF not found or empty: {fallback}")
    return fallback


def build_modelfile(spec: ModelSpec, gguf_arg: Optional[str], full_samurai_body: str) -> Path:
    gguf_path = resolve_gguf_path(spec, gguf_arg)

    try:
        from_rel = gguf_path.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        from_rel = gguf_path.as_posix()

    tmpl, stops = render_template_and_stops(spec)
    system_body = sanitize_modelfile_triple_quotes(f"{spec.variant_profile}\n\n{full_samurai_body}")
    few_shot = build_few_shot_messages(spec.display_name, spec.role_desc)

    lines = [
        "# =============================================================================",
        f"# {spec.display_name.upper()} (2M CONTEXT - FULL SOVEREIGN SYSTEM)",
        "# =============================================================================",
        f"FROM {from_rel}",
        "",
        *spec.extra_params,
        f"PARAMETER num_ctx {TARGET_NUM_CTX}",
        *stops,
        "",
        tmpl,
        "",
        f'SYSTEM """{system_body}\n"""',
        "",
        few_shot,
        "",
    ]
    out_path = ROOT_DIR / spec.modelfile_name
    content = "\n".join(lines).replace("\r\n", "\n").replace("\r", "\n")
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Full Sovereign 2M-context Modelfiles.")
    parser.add_argument("--flagship-gguf", default=None)
    parser.add_argument("--mini-gguf", default=None)
    parser.add_argument("--dragon-gguf", default=None)
    args = parser.parse_args()

    samurai_path = SAMURAI_PROMPT_PATH if SAMURAI_PROMPT_PATH.exists() else ROOT_QUILLAN_PATH
    with samurai_path.open("r", encoding="utf-8", newline="") as f:
        raw_text = f.read().replace("\r\n", "\n").replace("\r", "\n")
    full_body = strip_yaml_frontmatter(raw_text).rstrip().removesuffix("---").rstrip()
    print(f"Loaded FULL Sovereign Prompt from {samurai_path}: {len(full_body):,} chars / {len(full_body.splitlines()):,} lines")

    gguf_map = {
        "flagship": args.flagship_gguf,
        "mini": args.mini_gguf,
        "dragon": args.dragon_gguf,
    }
    for spec in SPECS:
        out_path = build_modelfile(spec, gguf_map[spec.key], full_body)
        print(f"Saved {out_path} ({out_path.stat().st_size:,} bytes) | num_ctx={TARGET_NUM_CTX}")


if __name__ == "__main__":
    main()
