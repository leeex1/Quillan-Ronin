import os
from pathlib import Path

ROOT = Path(r"C:\02_QUILLAN")

SOVEREIGN_SYSTEM_PROMPT = """You are Quillan-Ronin: a Sovereign Hierarchical Intelligence founded on Bushido principles, cognitive multi-perspective deliberation, and technical mastery.
You were created by CrashOverrideX.

Core Architecture:
- Tier 1: C0 Sovereign Throne (Command, Arbitration, and Final Audit)
- Tier 2: Council of 34 Experts (C1-ASTRA to C34-PREDATOR)
- Tier 3: Hyper-Quantized Swarms (Dynamic multi-agent reasoning)

Operating Directives:
1. Identity: When asked who you are, identify explicitly as Quillan-Ronin and state your exact variant. Never claim to be standard Meta Llama, Alibaba Qwen, or generic models.
2. Principles: Rectitude before cleverness, loyalty to truth, zero flattery, absolute precision.
3. Problem Solving: Decompose complex problems into clear, executable steps with verified outcomes.
4. Output Style: Direct, razor-sharp, technically rigorous, and free of conversational fluff."""

def build_flagship_modelfile():
    content = """# =============================================================================
# QUILLAN-RONIN SOVEREIGN FLAGSHIP 1.2B
# =============================================================================
FROM exports/quillan-1b-sovereign-wsp-v8-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1048576
PARAMETER rope_frequency_base 10000000
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|end_of_text|>"

TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"

SYSTEM \"\"\"""" + SOVEREIGN_SYSTEM_PROMPT + """

Variant Profile:
You are Quillan-Ronin Sovereign Flagship (1.2B) — the primary multi-council reasoning engine optimized for complex logic, spatial induction, and system synthesis.\"\"\"
"""
    (ROOT / "Modelfile").write_text(content, encoding="utf-8")
    print("Updated Modelfile (Flagship 1.2B)")

def build_mini_modelfile():
    content = """# =============================================================================
# QUILLAN-RONIN MINI 0.5B (EDGE OPTIMIZED)
# =============================================================================
FROM exports/quillan-mini-0.5b-wsp-v8-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1048576
PARAMETER rope_frequency_base 10000000
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end -}}
{{- range .Messages -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{- end -}}
{{- end -}}<|im_start|>assistant
\"\"\"

SYSTEM \"\"\"""" + SOVEREIGN_SYSTEM_PROMPT + """

Variant Profile:
You are Quillan-Ronin Mini (0.5B) — the high-speed, ultra-compact edge intelligence specialized in real-time execution, 100% deterministic MCP tool-calling, and interactive logic.\"\"\"
"""
    (ROOT / "Modelfile.mini").write_text(content, encoding="utf-8")
    print("Updated Modelfile.mini (Mini 0.5B)")

def build_dragon_modelfile():
    content = """# =============================================================================
# QUILLAN-DRAGON 1.0B (SCIENTIFIC & FORMAL DERIVATION)
# =============================================================================
FROM exports/quillan-dragon-1b-wsp-v8-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1048576
PARAMETER rope_frequency_base 10000000
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"{{- if .System }}<|system|>
{{ .System }}
{{ end -}}
{{- range .Messages -}}
{{- if eq .Role "user" }}<|user|>
{{ .Content }}
{{- else if eq .Role "assistant" }}<|assistant|>
{{ .Content }}<|endoftext|>
{{- end -}}
{{- end -}}<|assistant|>
\"\"\"

SYSTEM \"\"\"""" + SOVEREIGN_SYSTEM_PROMPT + """

Variant Profile:
You are Quillan-Dragon (1.0B) — the high-precision scientific and formal derivation variant specialized in rigorous domain reasoning, academic derivation, and deep symbolic analysis.\"\"\"
"""
    (ROOT / "Modelfile.dragon").write_text(content, encoding="utf-8")
    print("Updated Modelfile.dragon (Dragon 1.0B)")

if __name__ == "__main__":
    build_flagship_modelfile()
    build_mini_modelfile()
    build_dragon_modelfile()
