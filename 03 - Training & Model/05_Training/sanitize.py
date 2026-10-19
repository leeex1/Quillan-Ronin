import pathlib, re, os
# Sanitize main-images filenames: spaces -> _, remove commas, etc.
base = pathlib.Path(r"C:\02_QUILLAN\main-images")
count=0
for p in base.rglob("*"):
    if p.is_file():
        new_name = p.name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("#", "")
        if new_name != p.name:
            new_path = p.with_name(new_name)
            # Ensure not exists
            if not new_path.exists():
                p.rename(new_path)
                count+=1
print(f"renamed {count} in main-images")
# Do same for docs/main-images
base2 = pathlib.Path(r"C:\02_QUILLAN\docs\main-images")
count2=0
for p in base2.rglob("*"):
    if p.is_file():
        new_name = p.name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("#", "")
        if new_name != p.name:
            new_path = p.with_name(new_name)
            if not new_path.exists():
                p.rename(new_path)
                count2+=1
print(f"renamed {count2} in docs/main-images")
# Update HTML refs: main-images/xxx with spaces -> _
import pathlib as pl
for html_path in [r"C:\02_QUILLAN\index.html", r"C:\02_QUILLAN\docs\index.html"]:
    t = pl.Path(html_path).read_bytes().decode("utf-8")
    orig = t
    # Replace main-images/xxx" references: find all main-images/...png etc and sanitize
    def repl(m):
        path=m.group(1)
        sanitized=path.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("#", "")
        return f"main-images/{sanitized}"
    t = re.sub(r"main-images/([^\"]+)", lambda m: f"main-images/{m.group(1).replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('#','')}", t)
    if t != orig:
        pl.Path(html_path).write_bytes(t.encode("utf-8"))
        print(f"updated {html_path}")
    else:
        print(f"no change {html_path}")

# Also sanitize gallery filenames
base3 = pathlib.Path(r"C:\02_QUILLAN\gallery")
c3=0
for p in base3.rglob("*"):
    if p.is_file():
        new_name = p.name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("#", "")
        if new_name != p.name:
            new_path = p.with_name(new_name)
            if not new_path.exists():
                p.rename(new_path)
                c3+=1
print(f"renamed {c3} in gallery")
base4 = pathlib.Path(r"C:\02_QUILLAN\docs\gallery")
c4=0
if base4.exists():
    for p in base4.rglob("*"):
        if p.is_file():
            new_name = p.name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("#", "")
            if new_name != p.name:
                new_path = p.with_name(new_name)
                if not new_path.exists():
                    p.rename(new_path)
                    c4+=1
    print(f"renamed {c4} in docs/gallery")
