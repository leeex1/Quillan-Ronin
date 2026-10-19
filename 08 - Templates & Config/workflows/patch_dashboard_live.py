#!/usr/bin/env python3
"""Dashboard SIM -> LIVE: drive mesh/pipeline/telemetry from collector events.

- Adds liveMode engine + LIVE pill + manual toggle + auto-connect on load.
- tickSim skips fake visuals while live (clock keeps running).
- processEvent routes every stage event into driveLiveStage.
- Council tab no longer kills polling (monitoring must survive tab switches).
Backup first. Idempotent: skips if LIVE block already present.
"""
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = (r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts"
     r"\quillan-dashboard.html")
html = open(P, encoding="utf-8").read()
if "LIVE MODE ENGINE" in html:
    print("already patched, skipping")
    sys.exit(0)
shutil.copy(P, P + ".bak-prelive")
print("backup written")

LIVE_ENGINE = """
      /* ================= LIVE MODE ENGINE (real collector events) ================ */
      var liveMode = false;
      var liveLastTs = 0;
      var liveStageToWave = { intake: 0, prism: 1, council: 2, swarm: 2, gates: 3, arbitration: 4 };
      var liveChannel = { intake: "ROUTER", prism: "PRISM", council: "COUNCIL", swarm: "SWARM", gates: "GATE", arbitration: "ARBITER" };
      (function initLivePill() {
        var head = document.querySelector("#councilView .view-head");
        if (head) {
          var pill = document.createElement("div");
          pill.className = "head-stat";
          pill.innerHTML = "<span>MODE</span><b id='liveModeLabel'>DEMO SIM</b>";
          head.appendChild(pill);
        }
        var ctrls = document.querySelector(".log-controls");
        if (ctrls) {
          var btn = document.createElement("button");
          btn.className = "ghost-btn"; btn.id = "liveToggle"; btn.type = "button";
          btn.textContent = "Go live";
          btn.addEventListener("click", function () { setLiveMode(!liveMode, true); });
          ctrls.appendChild(btn);
        }
      })();
      function setLiveMode(on, manual) {
        liveMode = on;
        var label = document.getElementById("liveModeLabel");
        if (label) label.textContent = on ? "LIVE" : "DEMO SIM";
        var btn = document.getElementById("liveToggle");
        if (btn) btn.textContent = on ? "Back to sim" : "Go live";
        if (on) appendSimLog("SYSTEM", "Live monitoring engaged" + (manual ? " by operator." : " - real collector events driving the deck."));
        else if (manual) appendSimLog("SYSTEM", "Returned to demo simulation.");
      }
      function updateLiveAge() {
        var label = document.getElementById("liveModeLabel");
        if (liveMode && label && liveLastTs) label.textContent = "LIVE - " + Math.max(0, Math.round((Date.now() - liveLastTs) / 1000)) + "s ago";
      }
      function liveNum(text, re) {
        var m = String(text || "").match(re);
        return m ? Number(m[1]) : NaN;
      }
      function driveLiveStage(event) {
        var stage = String(event.stage || "").trim().toLowerCase();
        if (!(stage in liveStageToWave)) return;
        if (!liveMode) setLiveMode(true, false);
        liveLastTs = Date.now();
        var detail = String(event.detail || "");
        var status = String(event.status || "").toLowerCase();
        var wave = liveStageToWave[stage];
        var finished = (stage === "arbitration" && status === "pass");
        Array.from(pipeline.children).forEach(function (el, i) {
          var isDone = i < wave || finished;
          var isActive = i === wave && !finished;
          el.classList.toggle("active", isActive);
          el.classList.toggle("done", isDone);
          el.style.setProperty("--progress", isDone ? "100%" : (isActive ? "60%" : "0%"));
          el.querySelector(".wave-state").textContent = isDone ? "complete" : (isActive ? ("live - " + stage) : "queued");
        });
        var ids = [];
        var re = /C(\\d{1,2})\\b/g, m;
        while ((m = re.exec(detail)) !== null) {
          var id = Number(m[1]);
          if (id >= 0 && id <= 34 && ids.indexOf(id) === -1) ids.push(id);
        }
        if (ids.length) setActiveNodes(ids);
        else if (stage === "arbitration") setActiveNodes([]);
        var agreement = liveNum(detail, /agreement\\s+([\\d.]+)/i);
        var confidence = liveNum(detail, /confidence\\s+([\\d.]+)/i);
        var gates = liveNum(detail, /(\\d+)\\s*\\/\\s*5\\s+gates/i);
        if (isFinite(confidence)) {
          var pct = Math.round(confidence * 100);
          document.getElementById("governorGauge").style.setProperty("--gauge", pct + "%");
          document.getElementById("governorValue").textContent = pct;
        }
        if (isFinite(agreement)) {
          document.getElementById("coherenceValue").textContent = agreement.toFixed(2);
          document.getElementById("coherenceMeter").style.setProperty("--fill", Math.round(agreement * 100) + "%");
          document.getElementById("swarmFitness").textContent = agreement.toFixed(3).replace(/^0/, "");
        }
        if (isFinite(gates)) {
          var ev = 0.55 + 0.09 * gates;
          document.getElementById("evidenceValue").textContent = ev.toFixed(2);
          document.getElementById("evidenceMeter").style.setProperty("--fill", Math.round(ev * 100) + "%");
        }
        document.getElementById("governorState").textContent = status === "fail" ? "veto engaged" : "live bounded";
        appendSimLog(liveChannel[stage] || "LIVE", detail);
        document.getElementById("cycleNumber").textContent = String(cycle).padStart(4, "0");
      }
      /* ================= /LIVE MODE ENGINE ================ */
"""


def sub(pattern, repl, label):
    global html
    html, n = re.subn(pattern, repl, html, count=1, flags=re.M)
    assert n == 1, f"anchor missing: {label}"
    print(label, "ok")


# 1. engine block before the sim interval kickoff
sub(r"(?m)^([ \t]*)simTimer = setInterval\(tickSim, 1180\);",
    lambda m: LIVE_ENGINE + m.group(1) + "simTimer = setInterval(tickSim, 1180);",
    "engine inserted")

# 2. tickSim skips fake visuals while live
sub(r"(?m)^([ \t]*)simStep \+= 1;$",
    lambda m: m.group(0) + "\n" + m.group(1) + "if (liveMode) { updateLiveAge(); return; }",
    "tickSim gated")

# 3. stage events drive the deck
sub(r"(?m)^([ \t]*)processLocalStage\(raw, agent\);$",
    lambda m: m.group(0) + "\n" + m.group(1) + "driveLiveStage(raw);",
    "stage hook added")

# 4. council tab keeps polling
sub(r"(?m)^([ \t]*)if \(council\) \{\n\s*stopPolling\(\);\n",
    lambda m: m.group(1) + "if (council) {\n",
    "council tab keeps polling")

# 5. locate the poll call (used below), then auto-connect on load (append after the submit-listener registration)
sub(r"(?m)^([ \t]*)startPolling\(\);$",
    lambda m: m.group(0),
    "poll call located")
html = re.sub(
    r"collectorForm\"\)\.addEventListener\(\"submit\", function \(\) \{\n"
    r"\s*event\.preventDefault\(\);\n\s*startPolling\(\);\n\s*\}\);",
    lambda m: m.group(0) + "\n      startPolling();",
    html, count=1)
print("auto-connect on load")

open(P, "w", encoding="utf-8").write(html)
print("saved")
