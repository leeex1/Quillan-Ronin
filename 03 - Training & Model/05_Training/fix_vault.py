import pathlib, re, os, shutil
# Rename Main images -> main-images in docs and root, update HTML refs
candidates = [
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\docs\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\index.html"),
]
p = next((f for f in candidates if f.exists()), candidates[0])
t = p.read_bytes().decode("utf-8")
t = t.replace("Main images", "main-images")
t = t.replace("./main-images", "main-images")
if "scrollTop" not in t:
    t = t.replace("reindexVisibleTracks();", "reindexVisibleTracks();\n      const list=document.getElementById('tracksListContainer'); if(list) list.scrollTop=0;")

for target in candidates:
    if target.parent.exists():
        target.write_bytes(t.encode("utf-8"))
print("fixed refs", "main-images" in t, "Main images" not in t)

for base in [r"C:\02_QUILLAN", r"C:\02_QUILLAN\02_Projects\docs", r"C:\02_QUILLAN\docs"]:
    src = pathlib.Path(base) / "Main images"
    dst = pathlib.Path(base) / "main-images"
    if src.exists() and not dst.exists():
        src.rename(dst)
        print(f"renamed {src} -> {dst}")
    elif src.exists() and dst.exists():
        print(f"both exist at {base}")
