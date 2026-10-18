import pathlib, re, os
# Sanitize audio files in audio/ and docs/audio and 06_Media/audio for JDXX and Elemental
for base in [r"C:\02_QUILLAN\audio", r"C:\02_QUILLAN\docs\audio", r"C:\02_QUILLAN\06_Media\audio"]:
    for sub in ["JDXX Rebooted", "Elemental Avionics"]:
        p = pathlib.Path(base) / sub
        if not p.exists():
            continue
        for f in p.iterdir():
            if f.is_file():
                old = f.name
                new = old.replace(" ", "_").replace("#", "").replace("(", "").replace(")", "").replace(",", "").replace("'", "")
                # Also remove double __
                new = re.sub(r"_+", "_", new)
                if new != old:
                    new_path = f.with_name(new)
                    if not new_path.exists():
                        f.rename(new_path)
                        print(f"{base}/{sub}: {old} -> {new}")
                    else:
                        print(f"exists {new_path}")

# Update HTML refs
for html_path in [r"C:\02_QUILLAN\index.html", r"C:\02_QUILLAN\docs\index.html"]:
    t = pathlib.Path(html_path).read_bytes().decode("utf-8")
    orig = t
    # Replace audio/JDXX Rebooted/... and audio/Elemental Avionics/... paths
    def repl_audio(m):
        full = m.group(0)
        # full is audio/JDXX Rebooted/... or audio/Elemental Avionics/...
        sanitized = full.replace(" ", "_").replace("#", "").replace("(", "").replace(")", "").replace(",", "").replace("'", "")
        sanitized = re.sub(r"_+", "_", sanitized)
        return sanitized
    t = re.sub(r"audio/JDXX Rebooted/[^']+", repl_audio, t)
    t = re.sub(r"audio/Elemental Avionics/[^']+", repl_audio, t)
    if t != orig:
        pathlib.Path(html_path).write_bytes(t.encode("utf-8"))
        print(f"updated {html_path}")
    else:
        print(f"no change {html_path}")
