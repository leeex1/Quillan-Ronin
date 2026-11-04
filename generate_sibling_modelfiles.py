import sys, os

# Configure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Read canonical Modelfile to extract the exact SYSTEM block
with open(r"C:\02_QUILLAN\Modelfile", "r", encoding="utf-8") as f:
    modelfile_content = f.read()

system_pos = modelfile_content.find('SYSTEM """')
if system_pos == -1:
    print("ERROR: SYSTEM block not found in Modelfile")
    sys.exit(1)

system_block = modelfile_content[system_pos:]

# -------------------------------------------------------------
# Modelfile.mini (Quillan-Ronin-mini: 6-layer Qwen2 0.5B base)
# -------------------------------------------------------------
mini_header = """# =============================================================================
# ⚔️ QUILLAN-RONIN-MINI 0.5B MODELFILE FOR OLLAMA
# Full Verbatim System Prompt Specification (14,679 Lines)
# Architecture: Qwen2 0.5B (6-Layer Quantized Base)
# =============================================================================

# Base GGUF Model Blob Checkpoint
FROM C:\\Users\\Admin\\.ollama\\models\\blobs\\sha256-1923271f8d4d5101e91b90d67fdd56da544a6258c614a4d1ce9cd098696062f9

# Model Parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 131072
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"

# Prompt Template (ChatML Format with Native <think> Block)
TEMPLATE \"\"\"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end -}}
{{- range .Messages -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .Thinking }}<think>
{{ .Thinking }}
</think>
{{ end }}{{ .Content }}<|im_end|>
{{- end -}}
{{- end -}}<|im_start|>assistant
\"\"\"

# Full System Instruction Block (Verbatim Canonical Prompt)
"""

mini_path = r"C:\02_QUILLAN\Modelfile.mini"
with open(mini_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(mini_header + system_block)

print("Modelfile.mini generated successfully:", os.path.getsize(mini_path), "bytes")

# -------------------------------------------------------------
# Modelfile.dragon (Quillan-Dragon: Falcon3 1B base)
# -------------------------------------------------------------
dragon_header = """# =============================================================================
# 🐉 QUILLAN-DRAGON 1.7B MODELFILE FOR OLLAMA
# Full Verbatim System Prompt Specification (14,679 Lines)
# Architecture: Falcon3 1B Instruct Base (Q8_0)
# =============================================================================

# Base GGUF Model Blob Checkpoint
FROM C:\\Users\\Admin\\.ollama\\models\\blobs\\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01

# Model Parameters
PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 131072
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER stop "<|system|>"

# Prompt Template (ChatML Format with Native <think> Block)
TEMPLATE \"\"\"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end -}}
{{- range .Messages -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .Thinking }}<think>
{{ .Thinking }}
</think>
{{ end }}{{ .Content }}<|im_end|>
{{- end -}}
{{- end -}}<|im_start|>assistant
\"\"\"

# Full System Instruction Block (Verbatim Canonical Prompt)
"""

dragon_path = r"C:\02_QUILLAN\Modelfile.dragon"
with open(dragon_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(dragon_header + system_block)

print("Modelfile.dragon generated successfully:", os.path.getsize(dragon_path), "bytes")
