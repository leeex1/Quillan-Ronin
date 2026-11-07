import os
from huggingface_hub import HfApi, hf_hub_download

def update_ronin_readme():
    api = HfApi()
    repo_id = "CrashOverrideX/Quillan-Ronin"
    
    p = hf_hub_download(repo_id, "README.md")
    content = open(p, encoding="utf-8").read()

    fleet_table = """
<div align="center">
  <h2>⚔️ The Quillan-Ronin Tripartite Fleet ⚔️</h2>
  <p><i>Three specialized sovereign architectures united under the Bushido cognitive covenant.</i></p>
  <table align="center">
    <thead>
      <tr>
        <th>Tier</th>
        <th>Model Repository</th>
        <th>Parameters</th>
        <th>Base Architecture</th>
        <th>Held-Out Acc</th>
        <th>Perplexity</th>
        <th>Speed (tok/s)</th>
        <th>Primary Focus</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>Flagship Sovereign</b></td>
        <td><b><a href="https://huggingface.co/CrashOverrideX/Quillan-Ronin">Quillan-Ronin</a></b></td>
        <td><b>1.2B</b></td>
        <td>Llama 3.2 1B + Polymath WSP</td>
        <td><b>76.47%</b></td>
        <td><b>7.93 PPL</b></td>
        <td><b>47.0</b></td>
        <td>General Polymath Reasoning, Advanced Math, Code Synthesis</td>
      </tr>
      <tr>
        <td><b>Edge / Tool Specialist</b></td>
        <td><b><a href="https://huggingface.co/CrashOverrideX/Quillan-Ronin-Mini">Quillan-Ronin-Mini</a></b></td>
        <td><b>0.5B</b></td>
        <td>Qwen 2.5 0.5B + WSP v3</td>
        <td><b>58.82%</b></td>
        <td>8.98 PPL</td>
        <td><b>64.9</b></td>
        <td>Ultra-Low Latency, MCP Tool Invocations, Edge & Mobile (<400MB)</td>
      </tr>
      <tr>
        <td><b>Creative & Ethics</b></td>
        <td><b><a href="https://huggingface.co/CrashOverrideX/Quillan-Dragon">Quillan-Dragon</a></b></td>
        <td><b>1B</b></td>
        <td>Falcon 3 1B Instruct</td>
        <td>70.59%</td>
        <td><b>4.40 PPL</b></td>
        <td>41.3</td>
        <td>Deep Lore, Poetics, Bushido Rectitude, Harmonic Writing</td>
      </tr>
    </tbody>
  </table>
</div>

<br/>

### 🚀 Direct GGUF Run Instructions (Model A Flagship 1.2B)

```bash
# Run 1.2B Flagship WSP-v3 (76.47% Held-Out Accuracy, 47 tok/s)
llama-cli.exe -m quillan-1b-sovereign-wsp-v3-q4_k_m.gguf \\
  -p "<|start_header_id|>user<|end_header_id|>\\nSolve for x: 3x + 7 = 28<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n" \\
  -n 256 --temp 0.2
```

---
"""

    target_anchor = '<p align="center">\n  <img src="main-images/svgviewer-output.svg" alt="Quillan-Ronin Banner" width="800">\n</p>'
    
    if target_anchor in content:
        new_content = content.replace(target_anchor, target_anchor + "\n\n" + fleet_table)
    else:
        # Fallback to after </p> near top
        idx = content.find("</p>\n\n<p align=\"center\">")
        if idx != -1:
            new_content = content[:idx] + fleet_table + content[idx:]
        else:
            new_content = fleet_table + content

    print("Uploading updated README.md to CrashOverrideX/Quillan-Ronin...")
    api.upload_file(
        path_or_fileobj=new_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="docs: add Tripartite Fleet matrix and GGUF quickstart run commands"
    )
    print("README.md updated successfully!")

if __name__ == "__main__":
    update_ronin_readme()
