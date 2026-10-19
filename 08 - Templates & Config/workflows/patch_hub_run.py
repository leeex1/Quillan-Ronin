#!/usr/bin/env python3
"""Hub: add a Run-council button (browser-safe via collector /api/run)."""
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = (r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts"
     r"\quillan-dashboard.html")
html = open(P, encoding="utf-8").read()
if "runCouncil" in html:
    print("already patched, skipping")
    sys.exit(0)
shutil.copy(P, P + ".bak-hubrun")
print("backup written")

BTN_JS = """
      async function runCouncil() {
        var q = window.prompt("Council query:", "Should we automate our weekly reporting with n8n?");
        if (!q) return;
        appendSimLog("ROUTER", "Run requested: " + q);
        try {
          var r = await fetch("http://127.0.0.1:8777/api/run", { method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: q }) });
          var j = await r.json();
          if (j && j.verdict) appendSimLog("ARBITER", "Run finished: " + j.verdict + " at confidence " + j.confidence + ". Deck above shows the live trace.");
          else appendSimLog("GATE", "Run finished with unexpected response.");
        } catch (e) { appendSimLog("GATE", "Run failed: " + e); }
      }
"""


def sub(pattern, repl, label):
    global html
    html, n = re.subn(pattern, repl, html, count=1, flags=re.M)
    assert n == 1, f"anchor missing: {label}"
    print(label, "ok")


sub(r"(?m)^([ \t]*)ctrls\.appendChild\(btn\);$",
    lambda m: m.group(0) + "\n" + m.group(1) + (
        "var runBtn = document.createElement(\"button\");\n" + m.group(1) +
        "runBtn.className = \"ghost-btn\"; runBtn.type = \"button\";\n" + m.group(1) +
        "runBtn.textContent = \"Run council\";\n" + m.group(1) +
        "runBtn.addEventListener(\"click\", runCouncil);\n" + m.group(1) +
        "ctrls.appendChild(runBtn);"),
    "run button added")
sub(r"(?m)^([ \t]*)function setLiveMode",
    lambda m: BTN_JS + m.group(0),
    "runCouncil defined")

open(P, "w", encoding="utf-8").write(html)
print("saved")
