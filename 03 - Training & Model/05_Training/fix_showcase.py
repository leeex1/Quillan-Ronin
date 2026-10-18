import pathlib
candidates = [
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\docs\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\index.html"),
]
p = next((f for f in candidates if f.exists()), candidates[0])
t = p.read_bytes().decode("utf-8")
# Remove the New Album Cover Showcase we added
import re
m=re.search(r"<!-- New Album Cover Showcase -->.*?</div>\s*</div>\s*<!-- Album Filter Bar -->", t, flags=re.S)
if m:
    t=t.replace(m.group(0), "<!-- Album Filter Bar -->")
    print("removed showcase")
else:
    print("showcase not found")
    if "jdxx-rebooted.webp" in t:
        print("still has jdxx cover")
        t=re.sub(r"<div style=\"display:grid; grid-template-columns:repeat\(auto-fit, minmax\(280px.*?</div>\s*</div>\s*<!-- Album Filter Bar -->", "<!-- Album Filter Bar -->", t, flags=re.S)
        print("regex removed")

t=t.replace("??? ?? 173 NFT ART VAULT", "🖼️ 👑 173 NFT ART VAULT")
for target in candidates:
    if target.parent.exists():
        target.write_bytes(t.encode("utf-8"))
print("done", "showcase" not in t.lower() or "New Album" not in t)
print("vault", "🖼️ 👑" in t)
