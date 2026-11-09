import json
import re
import sys
import subprocess
from datasets import load_dataset

with open(r"C:\02_QUILLAN\exports\benchmark_100_per_section_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ds_he = {it["task_id"]: it for it in load_dataset("openai/openai_humaneval", split="test")}

def clean_he_code(resp: str, prompt_src: str, entry_point: str) -> str:
    text = resp.strip()
    m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    elif "def " in text:
        text = text[text.find("def "):].strip()

    # Cut off trailing self-generated tests or main blocks
    for marker in ["\n# Unit test", "\n# Test", "\n# Example", "\ndef test_", "\nif __name__", "\nassert ", "\nprint("]:
        if marker in text:
            text = text[:text.find(marker)].strip()

    # Remove trailing incomplete lines if truncated
    lines = text.splitlines()
    while lines and (lines[-1].strip().startswith("#") or lines[-1].strip().endswith((",", "(", "[", ":", "\\"))):
        lines.pop()
    text = "\n".join(lines)

    # Prepend imports from prompt_src
    imports = [ln for ln in prompt_src.splitlines() if ln.startswith(("import ", "from "))]
    return "\n".join(imports) + "\n" + text

for mname, mdata in data.items():
    he_logs = [l for l in mdata["detailed_logs"] if l["category"] == "HumanEval_Code"]
    re_passed = 0
    syntax_errs = 0
    for l in he_logs:
        tid = l["id"]
        it = ds_he[tid]
        code = clean_he_code(l["response"], it["prompt"], it["entry_point"])
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            syntax_errs += 1
            continue
        runner = f"{code}\n\n{it['test']}\ncheck({it['entry_point']})\nprint('__PASS__')\n"
        try:
            r = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and "__PASS__" in r.stdout:
                re_passed += 1
        except Exception:
            pass
    print(f"{mname}: orig HumanEval={mdata['category_summary']['HumanEval_Code']['passed']}/100 -> cleaned={re_passed}/100 (syntax_errs={syntax_errs})")

