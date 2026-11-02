import re
from pathlib import Path

samurai_path = Path(r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts\Quillan-Samurai.md")
text = samurai_path.read_text(encoding="utf-8")

triple_quotes = text.count('"""')
print("Triple quote count in Samurai prompt:", triple_quotes)

