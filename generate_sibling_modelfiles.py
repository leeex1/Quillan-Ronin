"""
Sibling Modelfile Generator for Quillan Tripartite Hub Registry
Generates Modelfile.mini (our custom 6-layer 0.5B model) and 
Modelfile.dragon (Falcon 3 1B foundation model) with exact verbatim prompt parity.
"""
import os
import sys

# Configure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def generate_sibling_modelfiles() -> None:
    # Read canonical Modelfile to extract the exact SYSTEM block
    with open(r"C:\02_QUILLAN\Modelfile", "r", encoding="utf-8") as f:
        modelfile_content = f.read()

    system_pos = modelfile_content.find('SYSTEM """')
    if system_pos == -1:
        raise ValueError("SYSTEM block not found in canonical Modelfile")

    system_block = modelfile_content[system_pos:]

    # -------------------------------------------------------------
    # Modelfile.mini (Quillan-Ronin-mini: Custom 6-Layer 0.5B Model)
    # -------------------------------------------------------------
    mini_header = """# =============================================================================
# ⚔️ QUILLAN-RONIN-MINI 0.5B MODELFILE FOR OLLAMA
# Custom Sovereign 6-Layer Small Architecture (0.5B)
# Full Verbatim System Prompt Specification (14,679 Lines)
# Model Pedigree: Proprietary Trained / Fused Sovereign Checkpoint
# =============================================================================

# Custom 6-Layer Mini Model Checkpoint
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

    print("Modelfile.mini written:", os.path.getsize(mini_path), "bytes")

    # -------------------------------------------------------------
    # Modelfile.dragon (Quillan-Dragon: Falcon 3 1B Instruct Base)
    # -------------------------------------------------------------
    dragon_header = """# =============================================================================
# 🐉 QUILLAN-DRAGON 1.7B MODELFILE FOR OLLAMA
# Falcon 3 1B Instruct Foundation Checkpoint (Q8_0)
# Full Verbatim System Prompt Specification (14,679 Lines)
# Model Pedigree: Falcon 3 1B Instruct Base with Full Sovereign System Prompt
# =============================================================================

# Falcon 3 Foundation Model
FROM falcon3:1b-instruct-q8_0

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

    print("Modelfile.dragon written:", os.path.getsize(dragon_path), "bytes")

if __name__ == "__main__":
    generate_sibling_modelfiles()
