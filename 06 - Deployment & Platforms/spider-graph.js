/* =========================================================================
 * SpiderGraph — lean Obsidian-style graph renderer for the Quillan dashboard
 * -------------------------------------------------------------------------
 * Why this file exists: the dashboard's inline canvas code grew to 1000+ lines
 * sharing one script block with everything else. This renderer owns ONLY the
 * spider canvas: pan / zoom / drag nodes / labels / flagged-focus web.
 *
 * Cost model (the whole point):
 *  - layout computed on DATA CHANGE only, never per frame
 *  - zero shadowBlur anywhere (pre-baked glow sprites, drawImage blits)
 *  - rAF loop redraws only while visible AND (animating || dirty)
 *  - labels capped; pulses capped; DPR capped at 2
 *
 * Data in:  SpiderGraph.setData(trailPayload)   <- /api/trail
 *           SpiderGraph.setMemory(memPayload)   <- /api/memory (totals)
 *           SpiderGraph.setFocus({full,name,findings,ties} | null)
 *           SpiderGraph.onFlagged = fn(focus)   <- dashboard wires inspector
 * ========================================================================= */
(function () {
  "use strict";

  var SECTOR_COLORS = ["#79dd8e", "#49e0da", "#ff7b72", "#a371f7",
                       "#f6b84b", "#58a6ff", "#3fb950", "#d29922"];
  // SPEC-CRAWLER anchors (video look): fixed swarms, fixed hues — mirrors 3D + bridge
  var SPEC = [
    { key: "role", color: "#f2f0e8", ax: 0.78, ay: -0.62 },
    { key: "objective", color: "#49e0da", ax: 0.08, ay: -0.72 },
    { key: "context", color: "#ff3d81", ax: -0.72, ay: -0.18 },
    { key: "rules", color: "#3fb950", ax: 0.72, ay: 0.28 },
    { key: "review", color: "#a371f7", ax: -0.18, ay: 0.05 },
    { key: "start", color: "#f6b84b", ax: -0.08, ay: 0.72 },
    { key: "ship", color: "#ff9d5c", ax: 0.42, ay: 0.66 }
  ];
  function specOf(full) {
    var p = String(full || "").toLowerCase();
    if (p.includes("memory") || p.includes("lancedb") || p.includes("02 - knowledge") || p.includes("obsidian") || p.includes("vault")) return 4;
    if (p.includes("06 -") || p.includes("deploy") || p.includes("dashboard") || p.includes("collector") || p.includes("09 -") || p.includes("mcp") || p.includes("server")) return 3;
    if (p.includes("05 -") || p.includes("media") || p.includes("creative")) return 2;
    if (p.includes("03 - training") || p.includes("model") || p.includes("forge") || p.includes("04-") || p.includes("skills") || p.includes("swarm")) return 1;
    if (p.includes("01 - core") || p.includes("cognition") || p.includes("samurai")) return 0;
    return 5;
  }
  function setSpec(spec) {
    G.spec = spec || null;
    G.dirty = true;
  }

  function sectorOf(path) {
    var p = String(path || "").toLowerCase();
    if (p.includes("01 - core") || p.includes("cognition") || p.includes("samurai")) return 0;
    if (p.includes("02 - knowledge") || p.includes("obsidian") || p.includes("vault")) return 1;
    if (p.includes("03 - training") || p.includes("model") || p.includes("forge")) return 2;
    if (p.includes("04-") || p.includes("skills") || p.includes("swarm")) return 3;
    if (p.includes("05 -") || p.includes("media") || p.includes("creative")) return 4;
    if (p.includes("06 -") || p.includes("deploy") || p.includes("dashboard") || p.includes("collector")) return 5;
    if (p.includes("07 -") || p.includes("memory") || p.includes("lancedb")) return 6;
    if (p.includes("09 -") || p.includes("mcp") || p.includes("server")) return 7;
    return -1;
  }

  // ---- glow sprites (baked once, blitted forever) ----
  var sprites = {};
  function glow(color) {
    var s = sprites[color];
    if (s) return s;
    var c = document.createElement("canvas");
    c.width = 64; c.height = 64;
    var g = c.getContext("2d");
    var rg = g.createRadialGradient(32, 32, 1, 32, 32, 32);
    rg.addColorStop(0, "#ffffff");
    rg.addColorStop(0.28, color);
    rg.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = rg;
    g.fillRect(0, 0, 64, 64);
    sprites[color] = c;
    return c;
  }

  var G = {
    canvas: null, ctx: null, wrap: null,
    running: false, raf: 0, dirty: true, anim: false,
    nodes: [], byId: {}, edges: [],
    mode: "trail", trailNodes: [], trailEdges: [],
    memNodes: [], memEdges: [], memNodesKey: "",
    layoutKey: "__none__",
    view: { x: 0, y: 0, z: 1 },
    hover: null, dragNode: null, dragPan: null,
    focus: null, spiderNode: -1, pulses: [],
    agents: [], fullMap: {},
    totals: { files: 0, wires: 0 },
    vignette: null, vignetteKey: "",
    lastKey: "",
    time: 0
  };

  function baseName(full) {
    return String(full || "").split(/[/\\]/).pop();
  }

  // ---- layout: SPEC swarms at fixed anchors (video look), then light relax ----
  function relayout(W, H) {
    var cx = W / 2, cy = H / 2;
    var RX = Math.max(120, W * 0.36), RY = Math.max(100, H * 0.34);
    var perSpec = {};
    var deg0 = {};
    G.edges.forEach(function (e) {
      if (G.byId[e.a]) deg0[G.byId[e.a].id] = 1;
      if (G.byId[e.b]) deg0[G.byId[e.b].id] = 1;
    });
    G.nodes.forEach(function (n) {
      var s = specOf(n.full);
      // loners gather at the start swarm for triage (video: gold bottom swarm)
      if (!deg0[n.id] && G.nodes.length > 12) s = 5;
      n.spec = s;
      n.sec = sectorOf(n.full);
      (perSpec[s] || (perSpec[s] = [])).push(n);
    });
    Object.keys(perSpec).forEach(function (sk) {
      var arr = perSpec[sk], s = +sk;
      var anch = SPEC[s] || SPEC[5];
      var scx = cx + anch.ax * RX, scy = cy + anch.ay * RY;
      arr.forEach(function (n, i) {
        if (n.pinned) return;
        // golden-angle cotton ball: dense like the screenshots
        var ga = i * 2.39996 + s;
        var rr = 8 + Math.sqrt(i) * 5.5 + Math.min(10, (n.visits || 0) * 2);
        if (n.x === undefined) {
          n.x = scx + Math.cos(ga) * rr;
          n.y = scy + Math.sin(ga) * rr * 0.8;
        }
        n.ax = scx; n.ay = scy; // anchor for gentle spring
      });
    });
    // connected-vs-dangling: wired files weave the center together;
    // loners hang on the outer ring so the user can spot and wire them.
    G.nodes.forEach(function (n) { n.dangling = false; });
    // unwired files already sit in the start swarm; mark them dangling there
    G.nodes.forEach(function (n) { n.dangling = (n.spec === 5 && !n.pinned); });
    // 24 relaxation ticks: anchor springs + pairwise repel (capped pairs)
    var i, j, a, b, dx, dy, d2, f;
    for (var t = 0; t < 24; t++) {
      for (i = 0; i < G.nodes.length; i++) {
        a = G.nodes[i];
        if (a.pinned || a.ax === undefined) continue;
        a.x += (a.ax - a.x) * 0.08;
        a.y += (a.ay - a.y) * 0.08;
      }
      var n = G.nodes.length, cap = Math.min(n, 220);
      for (i = 0; i < cap; i++) {
        a = G.nodes[i];
        if (a.pinned) continue;
        for (j = i + 1; j < cap; j++) {
          b = G.nodes[j];
          if (b.pinned) continue;
          dx = a.x - b.x; dy = a.y - b.y; d2 = dx * dx + dy * dy;
          if (d2 > 0.01 && d2 < 3600) {
            f = 900 / Math.max(d2, 60);
            dx *= f * 0.02; dy *= f * 0.02;
            a.x += dx; a.y += dy; b.x -= dx; b.y -= dy;
          }
        }
      }
    }
  }

  // fit the whole web into view after a dataset switch (memory web especially)
  function fitView() {
    if (!G.nodes.length || !G.canvas) return;
    var r = G.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return;
    var minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, any = false;
    G.nodes.forEach(function (n) {
      if (n.x === undefined) return;
      any = true;
      if (n.x < minX) minX = n.x; if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y; if (n.y > maxY) maxY = n.y;
    });
    if (!any) return;
    var bw = Math.max(80, maxX - minX), bh = Math.max(80, maxY - minY);
    G.view.z = Math.max(0.25, Math.min(2.2, Math.min((r.width - 80) / bw, (r.height - 80) / bh)));
    G.view.x = (minX + maxX) / 2;
    G.view.y = (minY + maxY) / 2;
    G.dirty = true;
  }

  function applyDataset() {
    // activate trail or memory node set; preserve positions across switches by full path
    var keep = {};
    G.nodes.forEach(function (n) { if (n.x !== undefined) keep[n.full] = { x: n.x, y: n.y, pinned: n.pinned }; });
    var src = G.mode === "memory" ? G.memNodes : G.trailNodes;
    var srcE = G.mode === "memory" ? G.memEdges : G.trailEdges;
    G.nodes = src.map(function (n) {
      var k = keep[n.full];
      return {
        id: n.id, full: n.full, name: n.name,
        kind: n.kind || "file", lines: n.lines || 0,
        x: k ? k.x : undefined, y: k ? k.y : undefined,
        pinned: k ? k.pinned : false,
        flagged: !!n.flagged, visits: n.visits || 0
      };
    });
    G.byId = {};
    G.nodes.forEach(function (n) { G.byId[n.id] = n; });
    G.fullMap = {};
    G.nodes.forEach(function (n) { if (n.full) G.fullMap[n.full] = n; });
    G.edges = srcE;
    G.layoutKey = null; // force relayout for the new set
    G.dirty = true;
    G.needFit = true; // fit AFTER relayout places nodes (see draw)
  }

  function setData(d) {
    if (!d || !d.trail) return;
    var key = d.trail.length + ":" + (d.nodes || []).length + ":" + ((d.spider || {}).state || "");
    var nodes = d.nodes || [];
    // rebuild trail table only when membership changes
    var memKey = "t:" + nodes.length + ":" + nodes.map(function (n) { return n.id; }).join(",");
    if (G.memKey !== memKey) {
      G.memKey = memKey;
      var tById = {};
      G.trailNodes = nodes.map(function (n) {
        return {
          id: n.id, full: n.full, name: baseName(n.full || n.path),
          kind: n.kind || "file", lines: 0,
          x: undefined, y: undefined, pinned: false,
          flagged: false, visits: 0
        };
      });
      G.trailNodes.forEach(function (n) { tById[n.id] = n; });
      G.trailEdges = (d.edges || []).slice(0, 1200).map(function (e) {
        return { a: e.from, b: e.to, wire: !!e.wire };
      });
      // per-event overlays onto the trail set
      d.trail.forEach(function (ev) {
        var n = tById[ev.node];
        if (!n) return;
        n.visits++;
        if (ev.lines) n.lines = ev.lines;
        if (ev.result === "FLAGGED") n.flagged = true;
      });
      if (G.mode === "trail") applyDataset();
      else G.dirty = true;
    } else if (G.mode === "trail") {
      // same membership: refresh overlays on the live set
      d.trail.forEach(function (ev) {
        var n = G.byId[ev.node];
        if (!n) return;
        n.visits++;
        if (ev.lines) n.lines = ev.lines;
        if (ev.result === "FLAGGED") n.flagged = true;
      });
    }
    var sp = d.spider || {};
    G.spiderNode = (sp.pos && sp.pos.node != null) ? sp.pos.node : -1;
    G.working = (sp.state === "WORKING" || sp.state === "CRAWLING" || sp.state === "WIRING");
    if (key !== G.lastKey) { G.lastKey = key; G.dirty = true; }
    if (G.working) G.anim = true;
  }

  function setMemory(m) {
    if (!m) return;
    G.totals.files = (m.census && m.census.total) || m.total_nodes || 0;
    G.totals.wires = m.total_edges || 0;
    var memKey = "m:" + (m.total_nodes || 0) + ":" + (m.updated || 0);
    if (G.memNodesKey !== memKey) {
      G.memNodesKey = memKey;
      G.memNodes = (m.nodes || []).map(function (n) {
        return {
          id: n.id, full: n.full, name: baseName(n.full || n.path),
          kind: n.kind || "file", lines: 0,
          x: undefined, y: undefined, pinned: false,
          flagged: !!n.flagged, visits: n.visits || 0
        };
      });
      G.memEdges = (m.edges || []).slice(0, 1200).map(function (e) {
        return { a: e.from, b: e.to, wire: !!e.wire };
      });
      if (G.mode === "memory") applyDataset();
    }
    G.dirty = true;
  }

  function setFocus(f) {
    G.focus = f || null;
    if (f) {
      // one pulse burst along its tie edges
      var fb = baseName(f.full).toLowerCase(), n = 0;
      G.edges.forEach(function (e) {
        if (n >= 6) return;
        var a = G.byId[e.a], b = G.byId[e.b];
        if (!a || !b) return;
        var hit = baseName(a.full).toLowerCase() === fb || baseName(b.full).toLowerCase() === fb;
        if (hit) {
          G.pulses.push({ a: e.a, b: e.b, t: 0, speed: 0.03, color: "#f6b84b" });
          G.pulses.push({ a: e.a, b: e.b, t: 0.5, speed: 0.022, color: "#ffffff" });
          n++;
        }
      });
      if (G.pulses.length > 80) G.pulses.splice(0, G.pulses.length - 80);
      G.anim = true;
    }
    G.dirty = true;
  }

  function setAgents(list) {
    G.agents = Array.isArray(list) ? list.slice(0, 12) : [];
    // ghost slot: if Antigravity hasn't heartbeated, show its empty perch so the
    // user sees WHO is missing instead of wondering if the canvas is broken
    var hasAG = G.agents.some(function (a) { return String(a.id || "").indexOf("antigravity") >= 0; });
    if (!hasAG && G.agents.length < 12) {
      G.agents.push({ id: "quillan-antigravity", name: "Quillan Antigravity",
        color: "#f6b84b", file: "", task: "ghost perch", children: [],
        stale: false, ghost: true });
    }
    G.dirty = true;
  }

  function hashStr(s) {
    var h = 0;
    s = String(s || "");
    for (var k = 0; k < s.length; k++) h = (h * 31 + s.charCodeAt(k)) >>> 0;
    return h;
  }

  // ---- agent spiders: one colored, named body per live agent; minis orbit ----
  // LIFE RULE: bots are never static — wander + leg-wave + mini orbit always run
  // while the tab is visible. Unmapped agents perch near the current view center
  // so they can never hide off-screen after a fit.
  function drawAgents(ctx, X, Y) {
    if (!G.agents.length) return;
    var t = G.time;
    G.agents.forEach(function (ag, ai) {
      var anchor = null;
      if (ag.file && G.fullMap) {
        anchor = G.fullMap[ag.file] || G.fullMap[String(ag.file).replace(/\//g, "\\")] || null;
      }
      var h = hashStr(ag.id + ai);
      var cx, cy;
      if (anchor && anchor.x !== undefined) {
        cx = anchor.x; cy = anchor.y - 40;
      } else {
        // perch near view center: stable angle per agent, tight radius
        var ang = (h % 360) * Math.PI / 180;
        var rr = 70 + (h % 60);
        cx = G.view.x + Math.cos(ang) * rr;
        cy = G.view.y + Math.sin(ang) * rr * 0.7 - 30;
      }
      // wander: slow crawl so WORKING bots visibly move (frozen only when stale)
      var stale = !!ag.stale;
      var ghost = !!ag.ghost;
      if (!stale) {
        cx += Math.sin(t / 1500 + h) * 16;
        cy += Math.cos(t / 1700 + h * 1.7) * 12;
      }
      var px = X(cx), py = Y(cy);
      ctx.save();
      if (stale && !ghost) ctx.globalAlpha = 0.35;
      if (ghost) ctx.globalAlpha = 0.85;
      // tether: silk line from bot down to its file anchor so you see WHAT it holds
      if (anchor && anchor.x !== undefined) {
        ctx.strokeStyle = ag.color; ctx.globalAlpha *= 1; ctx.lineWidth = 1;
        ctx.globalAlpha = ghost ? 0.35 : (stale ? 0.25 : 0.55);
        ctx.setLineDash(ghost ? [3, 5] : []);
        ctx.beginPath(); ctx.moveTo(px, py + 9); ctx.lineTo(X(anchor.x), Y(anchor.y)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = ghost ? 0.85 : (stale ? 0.35 : 1);
      }
      // glow + body + legs in the agent's own color
      ctx.drawImage(glow(ag.color), px - 22, py - 22, 44, 44);
      ctx.fillStyle = "#0a1216";
      ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = ag.color; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI * 2); ctx.stroke();
      var li;
      for (li = 0; li < 8; li++) {
        var la = (li / 8) * Math.PI * 2 + (stale ? 0 : t / 900);
        ctx.beginPath(); ctx.moveTo(px, py);
        ctx.lineTo(px + Math.cos(la) * 15, py + Math.sin(la) * 15); ctx.stroke();
      }
      // name plate above (pill bg so it reads over silk), file + task below
      ctx.textAlign = "center";
      var plate = String(ag.name || ag.id).slice(0, 28);
      ctx.font = "bold 11px ui-monospace, monospace";
      var pw = textW(ctx, plate) + 14;
      ctx.fillStyle = "rgba(3,8,10,0.82)";
      ctx.fillRect(px - pw / 2, py - 40, pw, 16);
      ctx.strokeStyle = ag.color; ctx.lineWidth = 1;
      ctx.strokeRect(px - pw / 2, py - 40, pw, 16);
      ctx.fillStyle = ag.color;
      ctx.fillText(plate, px, py - 28);
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillStyle = "rgba(200,215,220,0.75)";
      var sub = (ag.file ? baseName(ag.file) : "") + (ag.task ? " · " + ag.task : "");
      if (sub) ctx.fillText(sub.slice(0, 44), px, py + 24);
      if (ghost) {
        ctx.fillStyle = ag.color;
        ctx.fillText("AWAITING HEARTBEAT", px, py + 36);
      } else if (stale) {
        ctx.fillStyle = "rgba(150,170,180,0.8)";
        ctx.fillText("OFFLINE", px, py + 36);
      }
      // minis: one per sub-agent, orbiting the parent
      var kids = Array.isArray(ag.children) ? ag.children.slice(0, 12) : [];
      kids.forEach(function (kid, ki) {
        var ka = (ki / Math.max(1, kids.length)) * Math.PI * 2 - Math.PI / 2 + (stale ? 0 : t / 1400);
        var kx = X(cx + Math.cos(ka) * 30), ky = Y(cy + Math.sin(ka) * 30);
        ctx.drawImage(glow(ag.color), kx - 8, ky - 8, 16, 16);
        ctx.fillStyle = "#0a1216";
        ctx.beginPath(); ctx.arc(kx, ky, 4, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = ag.color; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(kx, ky, 4, 0, Math.PI * 2); ctx.stroke();
        ctx.font = "9px ui-monospace, monospace";
        ctx.fillStyle = ag.color;
        ctx.fillText(String(kid.name || kid.id).slice(0, 20), kx, ky - 9);
      });
      ctx.restore();
    });
  }

  // ---- input: pan / zoom / drag / select ----
  function toWorld(mx, my) {
    var r = G.canvas.getBoundingClientRect();
    return { x: (mx - r.width / 2) / G.view.z + G.view.x,
             y: (my - r.height / 2) / G.view.z + G.view.y };
  }
  function pick(mx, my) {
    var w = toWorld(mx, my), best = null, bd = 1e9;
    for (var i = 0; i < G.nodes.length; i++) {
      var n = G.nodes[i];
      if (n.x === undefined) continue;
      var dx = n.x - w.x, dy = n.y - w.y, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = n; }
    }
    var r = G.canvas.getBoundingClientRect();
    return (best && bd < Math.pow(26 / G.view.z, 2)) ? best : null;
  }

  function bindInput() {
    var cv = G.canvas;
    cv.style.cursor = "grab";
    cv.addEventListener("mousedown", function (ev) {
      var r = cv.getBoundingClientRect();
      var n = pick(ev.clientX - r.left, ev.clientY - r.top);
      if (n) { G.dragNode = n; n.pinned = true; }
      else { G.dragPan = { x: ev.clientX, y: ev.clientY, vx: G.view.x, vy: G.view.y }; }
      cv.style.cursor = "grabbing";
    });
    window.addEventListener("mouseup", function () {
      G.dragNode = null; G.dragPan = null;
      if (G.canvas) G.canvas.style.cursor = "grab";
    });
    window.addEventListener("mousemove", function (ev) {
      if (!G.canvas) return;
      var r = cv.getBoundingClientRect();
      if (G.dragNode) {
        var w = toWorld(ev.clientX - r.left, ev.clientY - r.top);
        G.dragNode.x = w.x; G.dragNode.y = w.y;
        G.dirty = true;
      } else if (G.dragPan) {
        G.view.x = G.dragPan.vx - (ev.clientX - G.dragPan.x) / G.view.z;
        G.view.y = G.dragPan.vy - (ev.clientY - G.dragPan.y) / G.view.z;
        G.dirty = true;
      } else {
        var h = pick(ev.clientX - r.left, ev.clientY - r.top);
        if ((h && h.id) !== (G.hover && G.hover.id)) { G.hover = h; G.dirty = true; }
      }
    });
    cv.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var f = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
      G.view.z = Math.max(0.25, Math.min(4, G.view.z * f));
      G.dirty = true;
    }, { passive: false });
    cv.addEventListener("click", function (ev) {
      var r = cv.getBoundingClientRect();
      var n = pick(ev.clientX - r.left, ev.clientY - r.top);
      if (G.onSelect) G.onSelect(n);
    });
    cv.addEventListener("mouseleave", function () { G.hover = null; G.dirty = true; });
  }

  // ---- draw ----
  // FRAME BUDGET: rect re-measured every 30 frames (was every frame +
  // layout thrash with HUD rebuilds); text widths cached; auto-degrade
  // drops vignette/arcs/chips if a frame costs >26ms on average.
  var _textW = {};
  function textW(ctx, s) {
    var w = _textW[s];
    if (w == null) { w = ctx.measureText(s).width; _textW[s] = w; }
    return w;
  }
  function draw() {
    var cv = G.canvas;
    G.frame = (G.frame || 0) + 1;
    if (!G.cachedW || G.frame % 30 === 1) {
      var rc = cv.getBoundingClientRect();
      if (!rc.width || !rc.height) return;
      G.cachedW = rc.width; G.cachedH = rc.height;
    }
    var t0 = (window.performance && performance.now) ? performance.now() : 0;
    var dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    var cw = Math.round(G.cachedW * dpr), ch = Math.round(G.cachedH * dpr);
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var W = G.cachedW, H = G.cachedH;
    if (!G.nodes.length) {
      ctx.fillStyle = "#050c0f"; ctx.fillRect(0, 0, W, H);
      return;
    }
    // layout runs on dataset/size change only — NEVER per frame (was the jank)
    var lk = G.nodes.length + ":" + (W | 0) + "x" + (H | 0) + ":" + G.mode;
    if (lk !== G.layoutKey) { G.layoutKey = lk; relayout(W, H); }
    // fit AFTER fresh layout so all nodes land inside the viewport
    if (G.needFit) { G.needFit = false; fitView(); }
    ctx.fillStyle = "#050c0f"; ctx.fillRect(0, 0, W, H);
    // cached vignette: depth without per-frame gradient cost (skipped when degraded)
    var vgKey = ((W | 0) + "x" + (H | 0));
    if (!G.degraded) {
      if (!G.vignette || G.vignetteKey !== vgKey) {
        var vg = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.72);
        vg.addColorStop(0, "rgba(0,0,0,0)");
        vg.addColorStop(1, "rgba(0,0,0,0.42)");
        G.vignette = vg; G.vignetteKey = vgKey;
      }
      ctx.fillStyle = G.vignette; ctx.fillRect(0, 0, W, H);
    }
    function X(x) { return (x - G.view.x) * G.view.z + W / 2; }
    function Y(y) { return (y - G.view.y) * G.view.z + H / 2; }

    // edges (single pass, no shadows)
    var i, e, a, b;
    // structural silk: ONE batched stroke for the whole web (was N strokes)
    ctx.strokeStyle = "rgba(73,224,218,0.16)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (i = 0; i < G.edges.length; i++) {
      e = G.edges[i];
      if (e.wire) continue;
      a = G.byId[e.a]; b = G.byId[e.b];
      if (!a || !b || a.x === undefined || b.x === undefined) continue;
      ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y));
    }
    ctx.stroke();
    // live wires: marching ants flowing toward coupled files
    ctx.strokeStyle = "rgba(246,184,75,0.55)";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([7, 7]);
    ctx.lineDashOffset = -(G.time / 55);
    ctx.beginPath();
    for (i = 0; i < G.edges.length; i++) {
      e = G.edges[i];
      if (!e.wire) continue;
      a = G.byId[e.a]; b = G.byId[e.b];
      if (!a || !b || a.x === undefined || b.x === undefined) continue;
      ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y));
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // focus web emphasis
    if (G.focus) {
      var fb = baseName(G.focus.full).toLowerCase();
      ctx.strokeStyle = "rgba(246,184,75,0.85)";
      ctx.lineWidth = 2;
      for (i = 0; i < G.edges.length; i++) {
        e = G.edges[i];
        a = G.byId[e.a]; b = G.byId[e.b];
        if (!a || !b) continue;
        if (baseName(a.full).toLowerCase() === fb || baseName(b.full).toLowerCase() === fb) {
          ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y)); ctx.stroke();
        }
      }
    }
    // nodes
    var showLabels = G.view.z > 1.15;
    var labeled = 0;
    for (i = 0; i < G.nodes.length; i++) {
      var n = G.nodes[i];
      if (n.x === undefined) continue;
      var px = X(n.x), py = Y(n.y);
      if (px < -40 || py < -20 || px > W + 40 || py > H + 20) continue;
      var col = (SPEC[n.spec] || SPEC[5]).color;
      var rad = 2.2 + Math.min(3.2, Math.sqrt(n.visits || 0) * 1.4);
      if (n.flagged) {
        var pr = 10 + 3 * Math.sin(G.time / 350);
        ctx.drawImage(glow("#f6b84b"), px - pr * 2, py - pr * 2, pr * 4, pr * 4);
      }
      if (G.byId[G.spiderNode] === n) {
        ctx.drawImage(glow("#79dd8e"), px - 16, py - 16, 32, 32);
      }
      ctx.fillStyle = n.flagged ? "#f6b84b" : col;
      ctx.beginPath(); ctx.arc(px, py, rad, 0, Math.PI * 2); ctx.fill();
      if (G.hover === n) {
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(px, py, rad + 4, 0, Math.PI * 2); ctx.stroke();
      }
      var doLabel = n.flagged || G.hover === n || (showLabels && (n.visits > 1 || n.flagged)) ;
      if (doLabel && labeled < 70) {
        labeled++;
        var label = n.name.slice(0, 26);
        ctx.font = "10px ui-monospace, monospace";
        ctx.textAlign = "left";
        // halo first so silk never eats the text (HQ readability, zero blur cost)
        ctx.strokeStyle = "rgba(3,8,10,0.9)"; ctx.lineWidth = 3;
        ctx.strokeText(label, px + rad + 4, py + 3);
        ctx.fillStyle = n.flagged ? "#f6b84b" : "rgba(232,242,239,0.85)";
        ctx.fillText(label, px + rad + 4, py + 3);
      }
    }
    // pulses (capped, sprites only)
    var pn = Math.min(G.pulses.length, 60);
    for (i = 0; i < pn; i++) {
      var pu = G.pulses[i];
      a = G.byId[pu.a]; b = G.byId[pu.b];
      if (!a || !b || a.x === undefined) continue;
      var qx = X(a.x + (b.x - a.x) * pu.t), qy = Y(a.y + (b.y - a.y) * pu.t);
      ctx.drawImage(glow(pu.color), qx - 7, qy - 7, 14, 14);
      pu.t += pu.speed;
      if (pu.t > 1) pu.t = 0;
    }
    if (G.focus) {
      var fn = null;
      for (i = 0; i < G.nodes.length; i++) {
        if (G.nodes[i].full === G.focus.full) { fn = G.nodes[i]; break; }
      }
      if (fn && fn.x !== undefined) {
        var fx = X(fn.x), fy = Y(fn.y);
        ctx.strokeStyle = "#f6b84b"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(fx, fy, 15 + 2 * Math.sin(G.time / 300), 0, Math.PI * 2); ctx.stroke();
      }
    }
    // dangling threads: loners hang from their sector anchor on drooping silk
    for (i = 0; i < G.nodes.length; i++) {
      var dn = G.nodes[i];
      if (!dn.dangling || dn.x === undefined || dn.tx === undefined) continue;
      var dx0 = X(dn.tx), dy0 = Y(dn.ty), dx1 = X(dn.x), dy1 = Y(dn.y);
      var dmx = (dx0 + dx1) / 2, dmy = (dy0 + dy1) / 2 + 14;
      ctx.strokeStyle = "rgba(150,170,180,0.28)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(dx0, dy0); ctx.quadraticCurveTo(dmx, dmy, dx1, dy1); ctx.stroke();
      ctx.fillStyle = "rgba(150,170,180,0.5)";
      ctx.beginPath(); ctx.arc(dx1, dy1 + 2, 1.6, 0, Math.PI * 2); ctx.fill();
    }
    // SPEC overlay: swarm tags + inter-swarm arcs + traveling crawler + claim chips
    drawSpecOverlay(ctx, X, Y, W, H);
    // agent spiders ride above the web so bodies never hide under silk
    drawAgents(ctx, X, Y);
    G.time += 16;
    if (t0) {
      var dt = performance.now() - t0;
      G.ema = (G.ema || 16) * 0.92 + Math.min(100, dt) * 0.08;
      if (!G.degraded && G.ema > 26) G.degraded = true;
      else if (G.degraded && G.ema < 14) G.degraded = false;
    }
  }

  function drawSpecOverlay(ctx, X, Y, W, H) {
    var t = G.time, order = [0, 1, 2, 3, 4, 5, 6], cx = [], i, k;
    // anchor screen positions from live nodes (average per swarm)
    var sum = {};
    G.nodes.forEach(function (n) {
      if (n.x === undefined) return;
      var s = n.spec;
      (sum[s] || (sum[s] = { x: 0, y: 0, c: 0 }));
      sum[s].x += n.x; sum[s].y += n.y; sum[s].c++;
    });
    order.forEach(function (s) {
      if (sum[s]) cx[s] = { x: X(sum[s].x / sum[s].c), y: Y(sum[s].y / sum[s].c) };
    });
    // arcs in phase order (the video's curved trails; skipped when degraded)
    if (!G.degraded) {
      ctx.lineWidth = 1.2;
      for (i = 0; i < order.length; i++) {
        var a = cx[order[i]], b = cx[order[(i + 1) % order.length]];
        if (!a || !b) continue;
        ctx.strokeStyle = "rgba(255,111,181,0.30)";
        ctx.beginPath(); ctx.moveTo(a.x, a.y);
        ctx.quadraticCurveTo((a.x + b.x) / 2, (a.y + b.y) / 2 - 34, b.x, b.y); ctx.stroke();
      }
    }
    // swarm tags: name + words + state, like the screenshots
    ctx.textAlign = "center";
    var counts = {};
    G.nodes.forEach(function (n) { counts[n.spec] = (counts[n.spec] || 0) + 1; });
    order.forEach(function (s) {
      if (!cx[s]) return;
      var sp = SPEC[s], active = G.spec && G.spec.active === sp.key;
      var words = 0;
      if (G.spec && G.spec.phases) {
        for (k = 0; k < G.spec.phases.length; k++) {
          if (G.spec.phases[k].key === sp.key) { words = G.spec.phases[k].words; break; }
        }
      }
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.fillStyle = sp.color;
      ctx.fillText(sp.key + (active ? " ●" : ""), cx[s].x, cx[s].y - 44);
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillStyle = "rgba(200,215,220,0.7)";
      ctx.fillText((counts[s] || 0) + " files" + (words ? " · " + words + "w" : ""), cx[s].x, cx[s].y - 32);
    });
    // traveling crawler: bright body hopping to the active swarm
    var actKey = (G.spec && G.spec.active) || "review";
    var ai = 4;
    for (k = 0; k < SPEC.length; k++) if (SPEC[k].key === actKey) ai = k;
    if (cx[ai]) {
      if (!G.crawler) G.crawler = { x: cx[ai].x, y: cx[ai].y };
      G.crawler.x += (cx[ai].x - G.crawler.x) * 0.05;
      G.crawler.y += (cx[ai].y - G.crawler.y) * 0.05;
      var px = G.crawler.x, py = G.crawler.y + Math.sin(t / 300) * 3;
      var cc = SPEC[ai].color;
      ctx.drawImage(glow(cc), px - 30, py - 30, 60, 60);
      ctx.fillStyle = "#0a1216";
      ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = cc; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = cc;
      ctx.beginPath(); ctx.arc(px - 3, py - 2, 2.2, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(px + 3, py - 2, 2.2, 0, Math.PI * 2); ctx.fill();
      var li;
      for (li = 0; li < 12; li++) {
        var la = (li / 12) * Math.PI * 2 + t / 700;
        ctx.strokeStyle = cc; ctx.lineWidth = 1.3;
        ctx.beginPath(); ctx.moveTo(px, py);
        ctx.lineTo(px + Math.cos(la) * 20, py + Math.sin(la) * 20); ctx.stroke();
      }
    }
    // claim chips: pink tags on flagged nodes (fewer when degraded)
    var shown = 0, chipCap = G.degraded ? 5 : 12;
    ctx.font = "8px ui-monospace, monospace";
    for (i = 0; i < G.nodes.length && shown < chipCap; i++) {
      var n = G.nodes[i];
      if (!n.flagged || n.x === undefined) continue;
      shown++;
      var nx = X(n.x), ny = Y(n.y);
      var label = n.name.slice(0, 18);
      var wpx = textW(ctx, label) + 10;
      ctx.fillStyle = "rgba(255,61,129,0.88)";
      ctx.fillRect(nx + 8, ny - 18, wpx, 12);
      ctx.fillStyle = "#0a0510";
      ctx.textAlign = "left";
      ctx.fillText(label, nx + 13, ny - 9);
      ctx.textAlign = "center";
    }
  }

  function loop() {
    if (!G.running) return;
    var cv = G.canvas;
    var visible = cv && !document.hidden &&
      !(document.getElementById("spiderView") || {}).hidden;
    var active = true; // web stays alive while tab visible: wander, legs, ants, minis
    var hadWork = G.working || G.pulses.length > 0 || (G.focus ? true : false) ||
      (G.agents && G.agents.some(function (a) { return !a.stale; }));
    G.anim = hadWork;
    if (visible && (G.dirty || active)) {
      draw();
      G.dirty = false;
    }
    G.raf = requestAnimationFrame(loop);
  }

  window.SpiderGraph = {
    toggle: function () {
      var want = (G.mode === "trail") ? "memory" : "trail";
      // never switch to an empty set — stay put instead of blanking the canvas
      if (want === "memory" && (!G.memNodes || !G.memNodes.length)) return G.mode;
      if (want === "trail" && (!G.trailNodes || !G.trailNodes.length)) return G.mode;
      G.mode = want;
      applyDataset();
      G.dirty = true;
      return G.mode;
    },
    mode: function () { return G.mode; },
    start: function (canvasId) {
      var cv = document.getElementById(canvasId || "spiderCanvas");
      if (!cv) return;
      if (G.canvas !== cv) {
        G.canvas = cv;
        // bind input ONCE per canvas — re-binding every tick stacked duplicate
        // listeners (multi-fire drag, wheel pile-up) and felt like jank
        if (!cv._sgBound) { bindInput(); cv._sgBound = true; }
        G.dirty = true;
      }
      if (G.canvas === cv && G.running) return;
      if (!G.running) { G.running = true; G.dirty = true; G.raf = requestAnimationFrame(loop); }
    },
    stop: function () { G.running = false; if (G.raf) cancelAnimationFrame(G.raf); G.raf = 0; },
    show: function () { if (G.canvas) { if (!G.running) { G.running = true; G.raf = requestAnimationFrame(loop); } G.dirty = true; } },
    hide: function () { G.running = false; if (G.raf) cancelAnimationFrame(G.raf); G.raf = 0; },
    setData: setData,
    setMemory: setMemory,
    setFocus: setFocus,
    setAgents: setAgents,
    setSpec: setSpec,
    onSelect: null,
    stats: function () { return { nodes: G.nodes.length, edges: G.edges.length, files: G.totals.files }; }
  };
})();
