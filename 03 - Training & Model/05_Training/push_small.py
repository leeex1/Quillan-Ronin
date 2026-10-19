import pathlib
candidates = [
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\docs\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\index.html"),
]
p = next((f for f in candidates if f.exists()), candidates[0])
t = p.read_bytes().decode("utf-8")
t = t.replace("</body>", "<!-- E2E 154926e filter+main-images sanitized for live -->\n</body>")
for target in candidates:
    if target.parent.exists():
        target.write_bytes(t.encode("utf-8"))
print("added E2E comment", len(t))
