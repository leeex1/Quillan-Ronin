#!/usr/bin/env python3
"""Measure the Dragon Modelfile system token count using the Falcon3 tokenizer."""

from pathlib import Path
import json
from tokenizers import Tokenizer

MODELFILE = Path(r"F:\02_QUILLAN\exports\Modelfile.dragon_standard")
TOK = Path(r"F:\02_QUILLAN\exports\quillan_tokenizer_v54_fourdonor\tokenizer.json")

text = MODELFILE.read_text(encoding="utf-8", errors="replace")
i = text.find("SYSTEM ")
j = text.find('"""', i)
k = text.find('"""', j + 3)
block = text[j + 3 : k]
tok = Tokenizer.from_file(str(TOK))
ids = tok.encode(block).ids
print(f"system_block_chars={len(block)}")
print(f"system_block_tokens={len(ids)}")
print(f"recommended_num_ctx={max(8192, ((len(ids) + 512 + 1023) // 1024) * 1024)}")
