import os
import sys
from huggingface_hub import HfApi

def sync_quillan_dragon():
    api = HfApi()
    repo_id = "CrashOverrideX/Quillan-Dragon"
    gguf_path = r"C:\02_QUILLAN\exports\quillan-dragon-1b-wsp-v3-q4_k_m.gguf"

    print(f"Checking {repo_id}...")
    existing = api.list_repo_files(repo_id)
    print("Existing files:", existing)

    readme_content = """---
license: apache-2.0
base_model: tiiuae/Falcon3-1B-Instruct
tags:
- falcon3
- text-generation
- gguf
- reasoning
- ollama
- bushido
- sovereign
- quillan-ronin
- weight-space-programming
- wsp
pipeline_tag: text-generation
model_name: Quillan-Dragon
---

# Quillan-Dragon (1B Falcon 3 Instruct — WSP-v3 & Q8_0)

<div align="center">
  <p><b>Part of the Quillan-Ronin Sovereign Tripartite Fleet</b></p>
  <a href="https://huggingface.co/CrashOverrideX/Quillan-Ronin"><b>Flagship (1.2B)</b></a> | 
  <a href="https://huggingface.co/CrashOverrideX/Quillan-Ronin-Mini"><b>Mini (0.5B)</b></a> | 
  <a href="https://huggingface.co/CrashOverrideX/Quillan-Dragon"><b>Dragon (1B)</b></a>
</div>

<br/>

**Quillan-Dragon** is the creative synthesis, ethical rectitude, and deep lore reasoning model of the **Quillan-Ronin Tripartite Fleet**. Built upon the **Falcon 3 1B Instruct** architecture and enhanced via **Weight-Space Programming (WSP)** and closed-form MoE folding, it holds the fleet record for lowest perplexity (**4.4029 PPL**) and delivers 100% adherence to Bushido-aligned reasoning and creative generation.

---

## Available Checkpoints

1. **`quillan-dragon-1b-wsp-v3-q4_k_m.gguf` (1,008 MB)**: The WSP-v3 surgically grafted edition. Highest efficiency, optimized for edge inference and fast generation (41.3 tok/s).
2. **`Quillan-Dragon-Q8_0.gguf` (1,696 MB)**: Full 8-bit precision release with complete unrolled sovereign prompt conditioning.

---

## Benchmark Highlights (GTX 1050 4GB Held-Out Seed 42)

| Metric | Result | Benchmark Context |
|---|---|---|
| **Held-Out Perplexity** | **4.4029 PPL (Fleet Record)** | Multi-topic cross-domain evaluation corpus |
| **Overall Accuracy** | **70.59%** (12/17 Passed) | Empirical Held-Out Multi-Domain Battery |
| **Bushido Lore & Reasoning** | **100.0%** (2/2 Passed) | Gi/Rectitude arbitration and rhymed verse generation |
| **Inference Velocity** | **41.3 tok/s** | Native CUDA FP16/INT4 on consumer hardware |
| **General Knowledge** | **100.0%** (3/3 Passed) | Chemistry, geography, and OSI network architecture |

---

## Tripartite Fleet Comparison

| Model | Parameters | Base Architecture | Accuracy | PPL | Speed | Primary Specialization |
|---|---|---|---|---|---|---|
| [Quillan-Ronin (Flagship)](https://huggingface.co/CrashOverrideX/Quillan-Ronin) | 1.2B | Llama 3.2 1B + Polymath WSP | 76.47% | 7.93 | 47.0 tok/s | General Polymath Reasoning, Deep Code Synthesis, Advanced Math |
| [Quillan-Ronin-Mini](https://huggingface.co/CrashOverrideX/Quillan-Ronin-Mini) | 0.5B | Qwen 2.5 0.5B | 58.82% | 8.98 | 64.9 tok/s | Ultra-Low Latency, MCP Tool Calling, Edge Micro-Services |
| **Quillan-Dragon** | **1B** | **Falcon 3 1B Instruct** | **70.59%** | **4.40** | **41.3 tok/s** | **Creative Writing, Bushido Ethics, Deep Lore, Harmonic Resonance** |

---

## Quickstart & Usage

### Option 1: Native llama.cpp / llama-cli
```bash
llama-cli.exe -m quillan-dragon-1b-wsp-v3-q4_k_m.gguf -p "<|startoftext|><|user|>\nState the Bushido virtue of Rectitude and provide a Council Verdict.<|endoftext|>\n<|assistant|>\n" -n 160 --temp 0.3
```

### Option 2: Ollama Modelfile
```dockerfile
FROM ./quillan-dragon-1b-wsp-v3-q4_k_m.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 8192
PARAMETER stop "<|endoftext|>"

TEMPLATE \"\"\"{{ if .System }}<|startoftext|><|system|>
{{ .System }}<|endoftext|>
{{ end }}{{ if .Prompt }}<|startoftext|><|user|>
{{ .Prompt }}<|endoftext|>
{{ end }}<|startoftext|><|assistant|>
{{ .Response }}<|endoftext|>\"\"\"

SYSTEM \"\"\"You are Quillan-Dragon, a sovereign creative reasoning agent adhering to the Bushido Code, thermodynamic truth synthesis, and harmonic balance.\"\"\"
```

---

## Citation & Contact
- **Creator**: CrashOverrideX ([Hugging Face](https://huggingface.co/CrashOverrideX))
- **Source Code**: [GitHub leeex1/Quillan-Ronin](https://github.com/leeex1/Quillan-Ronin)
- **License**: Apache-2.0
"""

    print("Updating README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="docs: update README with WSP-v3 benchmarks and fleet comparison"
    )
    print("README.md updated successfully!")

    if "quillan-dragon-1b-wsp-v3-q4_k_m.gguf" not in existing:
        print(f"Uploading GGUF model ({os.path.getsize(gguf_path)/(1024*1024):.2f} MB)...")
        api.upload_file(
            path_or_fileobj=gguf_path,
            path_in_repo="quillan-dragon-1b-wsp-v3-q4_k_m.gguf",
            repo_id=repo_id,
            commit_message="feat: release Quillan-Dragon 1B WSP-v3 Q4_K_M GGUF"
        )
        print("Dragon GGUF uploaded successfully!")
    else:
        print("Dragon GGUF already present in repo.")

if __name__ == "__main__":
    sync_quillan_dragon()
