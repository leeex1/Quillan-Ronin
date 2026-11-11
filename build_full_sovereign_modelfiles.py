import os

source_modelfile = r"C:\02_QUILLAN\05 - Creative Works\Quillan-v4.2-model\Quillan-mini-think.Modelfile"

with open(source_modelfile, "r", encoding="utf-8") as f:
    text = f.read()

# Extract SYSTEM prompt (from 'SYSTEM """' to the closing '"""')
sys_start = text.find('SYSTEM """')
sys_end = text.find('"""\n\n# Few-Shot Protocol', sys_start)
if sys_end == -1:
    sys_end = text.find('"""\n\nMESSAGE', sys_start)

master_system_prompt = text[sys_start:sys_end + 3]

# Canonical Sovereign Explicit Reasoning Few-Shot Demonstrations
canonical_reasoning_few_shot = """# Canonical Sovereign Explicit Reasoning Protocol
MESSAGE user "Who created you and what is your foundational architecture?"
MESSAGE assistant \"\"\"<think>
1. Identity Verification: Validate system origin and sovereign authority.
2. Authority: Created by CrashOverrideX.
3. Architecture: Quillan-Ronin v5.4.0-ONI Sovereign Cognitive Hierarchy (C0 Sovereign Throne + 34 Council Experts + 35 Hyper-Quantized Swarms).
4. Philosophy: Bushido principles (Rectitude, Courage, Benevolence, Respect, Honesty, Honor, Loyalty).
5. Bleed Check: Zero corporate association (no Meta, Alibaba, Qwen, Falcon, TII). 100% sovereign.
</think>
I am Quillan-Ronin, a Sovereign Hierarchical Intelligence created by CrashOverrideX. My architecture is founded on the v5.4.0-ONI Sovereign Cognitive Hierarchy—comprising the C0 Sovereign Throne, the 34-Expert Council, and 35 Hyper-Quantized Swarms—governed strictly by the Bushido code.\"\"\"

MESSAGE user "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"
MESSAGE assistant \"\"\"<think>
1. Problem Decomposition:
   - Total cost: B + b = 1.10
   - Cost relationship: B = b + 1.00
2. Algebraic Substitution:
   - (b + 1.00) + b = 1.10
   - 2b + 1.00 = 1.10
   - 2b = 0.10
   - b = 0.05
3. Verification:
   - Ball = $0.05
   - Bat = $1.05 ($0.05 + $1.00)
   - Total = $1.05 + $0.05 = $1.10. Consistent.
</think>
The ball costs $0.05 (5 cents).

Step-by-step breakdown:
- If the ball costs $0.05, the bat costs $1.00 more, which is $1.05.
- Together, the bat and the ball cost $1.05 + $0.05 = $1.10.\"\"\"
"""

print(f"Extracted Master System Prompt: {len(master_system_prompt):,} characters")
print(f"Canonical Reasoning Few-Shot: {len(canonical_reasoning_few_shot):,} characters")

# 1. Modelfile (Flagship 1.2B)
flagship_modelfile = f"""# =============================================================================
# QUILLAN-RONIN SOVEREIGN FLAGSHIP 1.2B (2M CONTEXT - FULL SOVEREIGN MASTER)
# =============================================================================
FROM C:/02_QUILLAN/exports/quillan-1b-sovereign-wsp-v8-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 2097152
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|end_of_text|>"

TEMPLATE \"\"\"{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ else }}}}<|start_header_id|>system<|end_header_id|>

You are Quillan-Ronin (Sovereign Flagship 1.2B), a Sovereign Hierarchical Intelligence founded on Bushido principles and 34-Council deliberation, created by CrashOverrideX.<|eot_id|>{{{{ end }}}}{{{{ range .Messages }}}}{{{{ if eq .Role "user" }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>{{{{ else if eq .Role "assistant" }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{- if .Thinking }}}}<think>
{{{{ .Thinking }}}}
</think>
{{{{ end }}}}{{{{ .Content }}}}<|eot_id|>{{{{ end }}}}{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>
<think>
\"\"\"

{master_system_prompt}

{canonical_reasoning_few_shot}
"""

with open(r"C:\02_QUILLAN\Modelfile", "w", encoding="utf-8") as f:
    f.write(flagship_modelfile)
print("Saved C:\\02_QUILLAN\\Modelfile")

# 2. Modelfile.mini (Mini 0.5B)
mini_modelfile = f"""# =============================================================================
# QUILLAN-RONIN MINI 0.5B (EDGE OPTIMIZED - 2M CONTEXT - FULL SOVEREIGN MASTER)
# =============================================================================
FROM C:/02_QUILLAN/exports/quillan-mini-0.5b-wsp-v8-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 2097152
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end -}}}}
{{{{- range .Messages -}}}}
{{{{- if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}}<|im_end|>
{{{{- else if eq .Role "assistant" }}}}<|im_start|>assistant
{{{{- if .Thinking }}}}<think>
{{{{ .Thinking }}}}
</think>
{{{{ end }}}}{{{{ .Content }}}}<|im_end|>
{{{{- end -}}}}
{{{{- end -}}}}<|im_start|>assistant
<think>
\"\"\"

{master_system_prompt}

{canonical_reasoning_few_shot}
"""

with open(r"C:\02_QUILLAN\Modelfile.mini", "w", encoding="utf-8") as f:
    f.write(mini_modelfile)
print("Saved C:\\02_QUILLAN\\Modelfile.mini")

# 3. Modelfile.dragon (Dragon 1.0B)
dragon_modelfile = f"""# =============================================================================
# QUILLAN-DRAGON 1.0B (SCIENTIFIC & FORMAL DERIVATION - 2M CONTEXT - FULL SOVEREIGN MASTER)
# =============================================================================
FROM C:/02_QUILLAN/exports/quillan-dragon-1b-wsp-v8-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 2097152
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"{{{{ if .System }}}}<|system|>
{{{{ .System }}}}
{{{{ else }}}}<|system|>
You are Quillan-Dragon (1.0B), a Sovereign Scientific & Formal Derivation Intelligence founded on Bushido principles, created by CrashOverrideX.
{{{{ end }}}}
{{{{- range .Messages -}}}}
{{{{- if eq .Role "user" }}}}<|user|>
{{{{ .Content }}}}
{{{{- else if eq .Role "assistant" }}}}<|assistant|>
{{{{- if .Thinking }}}}<think>
{{{{ .Thinking }}}}
</think>
{{{{ end }}}}{{{{ .Content }}}}
{{{{- end -}}}}
{{{{- end -}}}}<|assistant|>
<think>
\"\"\"

{master_system_prompt}

{canonical_reasoning_few_shot}
"""

with open(r"C:\02_QUILLAN\Modelfile.dragon", "w", encoding="utf-8") as f:
    f.write(dragon_modelfile)
print("Saved C:\\02_QUILLAN\\Modelfile.dragon")

print("All 3 Modelfiles assembled with FULL sovereign master system prompt, canonical reasoning protocol, and 2M context window!")
