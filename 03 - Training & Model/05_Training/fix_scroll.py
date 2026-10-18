import pathlib
candidates = [
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\docs\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\02_Projects\index.html"),
    pathlib.Path(r"C:\02_QUILLAN\index.html"),
]
p = next((f for f in candidates if f.exists()), candidates[0])
t = p.read_bytes().decode("utf-8")
old = "      console.log('[filterAlbum] visible', visible);\n      reindexVisibleTracks();"
new = "      console.log('[filterAlbum] visible', visible);\n      reindexVisibleTracks();\n      const list=document.getElementById('tracksListContainer'); if(list) list.scrollTop=0;"
if old in t:
    t = t.replace(old, new)
    print("added scrollTop")
else:
    print("not found or already present")

for target in candidates:
    if target.parent.exists():
        target.write_bytes(t.encode("utf-8"))
print("done", "scrollTop" in t)
