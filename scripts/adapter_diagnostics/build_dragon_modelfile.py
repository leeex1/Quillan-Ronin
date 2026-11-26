#!/usr/bin/env python3
"""Build Dragon standard Modelfile: clean Falcon3 blob + canonical Quillan SYSTEM block."""

from pathlib import Path

SRC = Path(r"F:\02_QUILLAN\Modelfile.dragon")
OUT = Path(r"F:\02_QUILLAN\exports\Modelfile.dragon_standard")
BLOB = r"C:\Users\Admin\.ollama\models\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"

t = SRC.read_text(encoding="utf-8", errors="replace")
sys_i = t.find("SYSTEM ")
msg_i = t.find("MESSAGE user", sys_i)
if sys_i < 0 or msg_i < 0:
    raise SystemExit("SYSTEM or MESSAGE user not found in source Modelfile")
system_block = t[sys_i:msg_i].rstrip()

tpl = """<|system|>
{{ .System }}
<|endoftext|>
<|user|>
{{ .Prompt }}
<|endoftext|>
<|assistant|>
"""

header = (
    f"FROM {BLOB}\n\n"
    "PARAMETER temperature 0.3\n"
    "PARAMETER top_p 0.9\n"
    "PARAMETER top_k 40\n"
    "PARAMETER repeat_penalty 1.15\n"
    "PARAMETER num_ctx 2097152\n"
    'PARAMETER stop "<|system|>"\n'
    'PARAMETER stop "<|user|>"\n'
    'PARAMETER stop "<|assistant|>"\n'
    'PARAMETER stop "<|endoftext|>"\n\n'
    f'TEMPLATE """{tpl}"""\n\n'
)
OUT.write_text(header + system_block + "\n", encoding="utf-8")
print(
    f"wrote {OUT} total_len={len(header + system_block)} system_block_len={len(system_block)}"
)
