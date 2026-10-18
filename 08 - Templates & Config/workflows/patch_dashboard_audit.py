#!/usr/bin/env python3
"""Bind the 9-row audit checklist to live run evidence (was demo-sim only)."""
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
P = (r"C:\02_QUILLAN\06 - Deployment & Platforms\system prompts"
     r"\quillan-dashboard.html")
html = open(P, encoding="utf-8").read()
if "driveLiveAudit" in html:
    print("already patched, skipping")
    sys.exit(0)
shutil.copy(P, P + ".bak-liveaudit")
print("backup written")

AUDIT_DRIVER = """
      /* ============ LIVE AUDIT DRIVER (9-row checklist from live runs) ============ */
      var liveAuditOrder = ["intake", "prism", "council", "swarm", "predator", "vir", "warden", "vigil", "arbitration"];
      var liveAudits = {};
      var liveAuditSeq = 0;
      function driveLiveAudit(event, agent) {
        var stage = String(event.stage || "").trim().toLowerCase();
        if (liveAuditOrder.indexOf(stage) === -1) return;
        var rec = liveAudits[agent.id];
        if (stage === "intake" || !rec) {
          liveAuditSeq += 1;
          rec = { id: "LIVE-" + String(liveAuditSeq).padStart(4, "0"), rows: {}, verdict: "RUNNING", recorded: false };
          liveAudits[agent.id] = rec;
        }
        var status = String(event.status || "").toLowerCase();
        rec.rows[stage] = { status: status === "pass" ? "pass" : "fail", detail: String(event.detail || ""), ts: timeText(new Date(event.ts || Date.now())) };
        var failed = liveAuditOrder.some(function (k) { return rec.rows[k] && rec.rows[k].status === "fail"; });
        var complete = liveAuditOrder.every(function (k) { return rec.rows[k] && rec.rows[k].status === "pass"; });
        rec.verdict = failed ? "NON-COMPLIANT" : (complete ? "COMPLIANT" : "RUNNING");
        renderLiveAudit(rec);
        if ((failed || complete) && !rec.recorded) {
          rec.recorded = true;
          auditHistory.push({ id: rec.id, time: timeText(new Date()), verdict: complete ? "COMPLIANT" : "FAILED" });
          renderAuditHistory();
        }
      }
      function renderLiveAudit(rec) {
        document.getElementById("auditRunId").textContent = rec.id;
        var v = document.getElementById("auditVerdict");
        v.className = "run-verdict " + (rec.verdict === "COMPLIANT" ? "compliant" : (rec.verdict === "RUNNING" ? "running" : "failed"));
        v.textContent = rec.verdict;
        var list = document.getElementById("auditChecklist");
        list.replaceChildren();
        var passed = 0;
        liveAuditOrder.forEach(function (key, index) {
          var spec = null;
          for (var i = 0; i < auditStages.length; i++) { if (auditStages[i].key === key) { spec = auditStages[i]; break; } }
          var r = rec.rows[key] || { status: "pending", detail: spec ? spec.detail : "", ts: "\\u2014" };
          if (r.status === "pass") passed += 1;
          var row = document.createElement("div");
          row.className = "check-row";
          var idx = document.createElement("span"); idx.className = "check-index"; idx.textContent = String(index + 1).padStart(2, "0");
          var text = document.createElement("div");
          var title = document.createElement("span"); title.className = "check-name"; title.textContent = spec ? spec.name : key;
          var detail = document.createElement("span"); detail.className = "check-detail"; detail.textContent = r.detail;
          text.append(title, detail);
          var time = document.createElement("time"); time.className = "check-time"; time.textContent = r.ts;
          var status = document.createElement("span"); status.className = "stage-status " + r.status; status.textContent = r.status === "pending" ? "WAITING" : r.status.toUpperCase();
          row.append(idx, text, time, status);
          list.appendChild(row);
        });
        document.getElementById("auditProgress").textContent = passed + " / " + liveAuditOrder.length + " passed";
      }
      /* ============ /LIVE AUDIT DRIVER ============ */
"""


def sub(pattern, repl, label):
    global html
    html, n = re.subn(pattern, repl, html, count=1, flags=re.M)
    assert n == 1, f"anchor missing: {label}"
    print(label, "ok")


sub(r"(?m)^([ \t]*)/\* =+ /LIVE MODE ENGINE",
    lambda m: AUDIT_DRIVER + m.group(0),
    "audit driver inserted")
sub(r"(?m)^([ \t]*)driveLiveStage\(raw\);$",
    lambda m: m.group(0) + "\n" + m.group(1) + "driveLiveAudit(raw, agent);",
    "audit hook added")

open(P, "w", encoding="utf-8").write(html)
print("saved")
