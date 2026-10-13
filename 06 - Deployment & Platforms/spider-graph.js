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

  // ---- layout: hierarchical sectors once per dataset, then light relax ----
  function relayout(W, H) {
    var cx = W / 2, cy = H / 2;
    var RX = Math.max(120, W * 0.36), RY = Math.max(100, H * 0.34);
    var perSec = {};
    G.nodes.forEach(function (n) {
      var s = sectorOf(n.full);
      n.sec = s;
      (perSec[s] || (perSec[s] = [])).push(n);
    });
    Object.keys(perSec).forEach(function (sk) {
      var arr = perSec[sk], s = +sk;
      var ang = (s >= 0 ? (s / 8) * Math.PI * 2 - Math.PI / 2 : 0);
      var scx = s >= 0 ? cx + Math.cos(ang) * RX : cx;
      var scy = s >= 0 ? cy + Math.sin(ang) * RY : cy;
      arr.forEach(function (n, i) {
        if (n.pinned) return;
        var a = (i / Math.max(1, arr.length)) * Math.PI * 2;
        var rr = 26 + (i % 5) * 14;
        if (n.x === undefined) {
          n.x = scx + Math.cos(a) * rr;
          n.y = scy + Math.sin(a) * rr;
        }
        n.ax = scx; n.ay = scy; // anchor for gentle spring
      });
    });
    // connected-vs-dangling: wired files weave the center together;
    // loners hang on the outer ring so the user can spot and wire them.
    G.nodes.forEach(function (n) { n.dangling = false; });
    var deg = {};
    G.edges.forEach(function (e) {
      var da = G.byId[e.a], db = G.byId[e.b];
      if (da) deg[da.id] = (deg[da.id] || 0) + 1;
      if (db) deg[db.id] = (deg[db.id] || 0) + 1;
    });
    if (G.nodes.length > 12) {
      var loners = G.nodes.filter(function (n) { return !n.pinned && !deg[n.id]; });
      loners.sort(function (p, q) { return p.full < q.full ? -1 : 1; });
      loners.forEach(function (n, i) {
        n.dangling = true;
        n.tx = n.ax; n.ty = n.ay; // silk thread hangs from sector anchor
        var la = (i / Math.max(1, loners.length)) * Math.PI * 2;
        n.ax = cx + Math.cos(la) * RX * 1.55;
        n.ay = cy + Math.sin(la) * RY * 1.55;
        if (n.x === undefined) { n.x = n.ax; n.y = n.ay; }
      });
    }
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
    fitView();
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
    G.dirty = true;
  }

  function hashStr(s) {
    var h = 0;
    s = String(s || "");
    for (var k = 0; k < s.length; k++) h = (h * 31 + s.charCodeAt(k)) >>> 0;
    return h;
  }

  // ---- agent spiders: one colored, named body per live agent; minis orbit ----
  function drawAgents(ctx, X, Y) {
    if (!G.agents.length) return;
    var t = G.time;
    G.agents.forEach(function (ag) {
      var anchor = (ag.file && G.fullMap) ? G.fullMap[ag.file] : null;
      var cx, cy;
      if (anchor && anchor.x !== undefined) {
        cx = anchor.x; cy = anchor.y - 40;
      } else {
        // stable free slot on a wide ring so unmapped agents stay put
        var h = hashStr(ag.id);
        var ang = (h % 360) * Math.PI / 180;
        cx = Math.cos(ang) * 320; cy = Math.sin(ang) * 260;
      }
      var px = X(cx), py = Y(cy);
      var stale = !!ag.stale;
      ctx.save();
      if (stale) ctx.globalAlpha = 0.35;
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
      // name plate above, file + task below
      ctx.textAlign = "center";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.fillStyle = ag.color;
      ctx.fillText(String(ag.name || ag.id).slice(0, 28), px, py - 26);
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillStyle = "rgba(200,215,220,0.75)";
      var sub = (ag.file ? baseName(ag.file) : "") + (ag.task ? " · " + ag.task : "");
      if (sub) ctx.fillText(sub.slice(0, 44), px, py + 24);
      if (stale) {
        ctx.fillStyle = "rgba(150,170,180,0.8)";
        ctx.fillText("OFFLINE", px, py + 36);
      }
      // minis: one per sub-agent, orbiting the parent
      var kids = Array.isArray(ag.children) ? ag.children.slice(0, 12) : [];
      kids.forEach(function (kid, ki) {
        var ka = (ki / Math.max(1, kids.length)) * Math.PI * 2 - Math.PI / 2;
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
  function draw() {
    var cv = G.canvas;
    var r = cv.getBoundingClientRect();
    if (!r.width || !r.height) return;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var cw = Math.round(r.width * dpr), ch = Math.round(r.height * dpr);
    if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
    var ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var W = r.width, H = r.height;
    if (!G.nodes.length) {
      ctx.fillStyle = "#050c0f"; ctx.fillRect(0, 0, W, H);
      return;
    }
    // layout runs on dataset/size change only — NEVER per frame (was the jank)
    var lk = G.nodes.length + ":" + (W | 0) + "x" + (H | 0) + ":" + G.mode;
    if (lk !== G.layoutKey) { G.layoutKey = lk; relayout(W, H); }
    ctx.fillStyle = "#050c0f"; ctx.fillRect(0, 0, W, H);
    // cached vignette: depth without per-frame gradient cost
    var vgKey = ((W | 0) + "x" + (H | 0));
    if (!G.vignette || G.vignetteKey !== vgKey) {
      var vg = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.72);
      vg.addColorStop(0, "rgba(0,0,0,0)");
      vg.addColorStop(1, "rgba(0,0,0,0.42)");
      G.vignette = vg; G.vignetteKey = vgKey;
    }
    ctx.fillStyle = G.vignette; ctx.fillRect(0, 0, W, H);
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
      var col = n.sec >= 0 ? SECTOR_COLORS[n.sec % 8] : "#55707a";
      var rad = 3 + Math.min(4, n.visits);
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
        ctx.fillStyle = n.flagged ? "#f6b84b" : "rgba(232,242,239,0.75)";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText(n.name.slice(0, 26), px + rad + 4, py + 3);
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
    // agent spiders ride above the web so bodies never hide under silk
    drawAgents(ctx, X, Y);
    G.time += 16;
  }

  function loop() {
    if (!G.running) return;
    var cv = G.canvas;
    var visible = cv && !document.hidden &&
      !(document.getElementById("spiderView") || {}).hidden;
    var active = G.working || G.pulses.length > 0 || (G.focus ? true : false) ||
      (G.agents && G.agents.some(function (a) { return !a.stale; }));
    if (visible && (G.dirty || active)) {
      draw();
      G.dirty = false;
    }
    G.anim = active;
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
    onSelect: null,
    stats: function () { return { nodes: G.nodes.length, edges: G.edges.length, files: G.totals.files }; }
  };
})();
