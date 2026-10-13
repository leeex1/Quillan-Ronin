/* =========================================================================
 * Spider3D — true-3D premium web + game-quality spider bots (Three.js r128)
 * Classic script: lazy-loads THREE + OrbitControls from CDN, exposes
 *   window.Spider3D = { init(canvasId), setData(trail), setAgents(list),
 *     setFocus(f), show(), hide(), visible(), onSelect }
 * Cost model: Points + LineSegments for the web (cheap), procedural Group
 * spiders only for agents (<=12) + flagged nodes (<=40 spheres).
 * ========================================================================= */
(function () {
  "use strict";
  var G = {
    ready: false, loading: false, shown: false, three: null,
    scene: null, camera: null, renderer: null, controls: null, raf: 0,
    webGroup: null, spiderGroup: null, stars: null,
    nodes: [], byId: {}, edges: [], agents: [],
    nodePts: null, silkLines: null, wireLines: null,
    spiders: {}, time: 0, ray: null, mouse: null, hover: null
  };
  var SECTOR = ["#79dd8e", "#49e0da", "#ff7b72", "#a371f7",
                "#f6b84b", "#58a6ff", "#3fb950", "#d29922"];
  // SPEC-CRAWLER LAYER (from video ref): 7 fixed sections, fixed hues, fixed
  // cluster anchors so the web reads as role/objective/context/rules/review/start/ship
  var SPEC = [
    { key: "role",      color: "#f2f0e8", anchor: [62, 26, -30] },
    { key: "objective", color: "#49e0da", anchor: [6, 34, -6] },
    { key: "context",   color: "#ff3d81", anchor: [-58, 8, 6] },
    { key: "rules",     color: "#3fb950", anchor: [58, -2, 10] },
    { key: "review",    color: "#a371f7", anchor: [-16, 2, 0] },
    { key: "start",     color: "#f6b84b", anchor: [-6, -38, 14] },
    { key: "ship",      color: "#ff9d5c", anchor: [30, -30, -34] }
  ];
  function specOf(full) {
    // map file sectors onto spec sections so existing crawl data clusters
    // exactly like the video: each file joins one colored swarm
    var s = sectorOf(full);
    if (s === 6 || s === 1) return 4; // memory/knowledge -> review
    if (s === 5 || s === 7) return 3; // deploy/mcp -> rules
    if (s === 4) return 2;            // creative -> context
    if (s === 2 || s === 3) return 1; // training/skills -> objective
    if (s === 0) return 0;            // core -> role
    return 5;                          // unmapped -> start swarm
  }
  function sectorOf(p) {
    p = String(p || "").toLowerCase();
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
  function loadScript(src) {
    return new Promise(function (res, rej) {
      var s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }
  async function ensureThree() {
    if (window.THREE && window.THREE.OrbitControls) return window.THREE;
    if (G.loading) { while (!window.THREE || !window.THREE.OrbitControls) await new Promise(function (r) { setTimeout(r, 120); }); return window.THREE; }
    G.loading = true;
    if (!window.THREE) await loadScript("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js");
    if (!window.THREE.OrbitControls) await loadScript("https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js");
    G.loading = false;
    return window.THREE;
  }
  function makeLabel(text, color) {
    var c = document.createElement("canvas"); c.width = 256; c.height = 64;
    var x = c.getContext("2d");
    x.fillStyle = "rgba(3,8,10,0.85)"; x.fillRect(0, 8, 256, 48);
    x.strokeStyle = color; x.lineWidth = 3; x.strokeRect(2, 10, 252, 44);
    x.font = "bold 26px ui-monospace, monospace"; x.textAlign = "center";
    x.fillStyle = color; x.fillText(String(text).slice(0, 20), 128, 43);
    var tex = new window.THREE.CanvasTexture(c);
    var sp = new window.THREE.Sprite(new window.THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    sp.scale.set(22, 5.5, 1);
    return sp;
  }
  // procedural spider: abdomen + head + eyes + 8 two-segment walking legs
  function makeSpider(color) {
    var THREE = window.THREE;
    var g = new THREE.Group();
    var mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(color), roughness: 0.35, metalness: 0.55, emissive: new THREE.Color(color), emissiveIntensity: 0.25 });
    var dark = new THREE.MeshStandardMaterial({ color: 0x0a1216, roughness: 0.6, metalness: 0.3 });
    var abdomen = new THREE.Mesh(new THREE.SphereGeometry(2.2, 18, 14), mat);
    abdomen.scale.set(1, 0.8, 1.35); abdomen.position.z = -2.2; g.add(abdomen);
    var head = new THREE.Mesh(new THREE.SphereGeometry(1.3, 16, 12), dark);
    head.position.z = 1.2; g.add(head);
    var eyeMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(color) });
    [[-0.45, 0.35], [0.45, 0.35], [-0.2, 0.7], [0.2, 0.7]].forEach(function (e) {
      var eye = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), eyeMat);
      eye.position.set(e[0], e[1] + 0.6, 2.2); g.add(eye);
    });
    var legs = [];
    for (var i = 0; i < 8; i++) {
      var side = i < 4 ? -1 : 1, row = i % 4;
      var hip = new THREE.Group();
      hip.position.set(side * 1.1, -0.2, 1.2 - row * 1.1);
      var upper = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.1, 3.2, 6), mat);
      upper.position.y = 1.6; hip.add(upper);
      var knee = new THREE.Group(); knee.position.y = 3.2; hip.add(knee);
      var lower = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.05, 3.4, 6), mat);
      lower.position.y = 1.7; knee.add(lower);
      hip.rotation.z = side * 1.05; hip.rotation.y = side * (0.35 - row * 0.22);
      knee.rotation.z = -side * 1.15;
      g.add(hip);
      legs.push({ hip: hip, knee: knee, phase: (i / 8) * Math.PI * 2 });
    }
    var glowC = document.createElement("canvas"); glowC.width = glowC.height = 64;
    var gx = glowC.getContext("2d");
    var grad = gx.createRadialGradient(32, 32, 2, 32, 32, 32);
    grad.addColorStop(0, color); grad.addColorStop(1, "rgba(0,0,0,0)");
    gx.fillStyle = grad; gx.fillRect(0, 0, 64, 64);
    var glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(glowC), transparent: true, opacity: 0.55, depthWrite: false }));
    glow.scale.set(14, 14, 1); g.add(glow);
    g.userData.legs = legs;
    return g;
  }
  function layout3D() {
    // SPEC-CRAWLER layout: wired files join their section swarm at fixed anchors
    // (video look); loners gather at the start swarm to be triaged, not scattered
    var deg = {};
    G.edges.forEach(function (e) { deg[e.a] = (deg[e.a] || 0) + 1; deg[e.b] = (deg[e.b] || 0) + 1; });
    var perSpec = {};
    G.nodes.forEach(function (n) {
      n.sec = sectorOf(n.full);
      n.spec = specOf(n.full);
      if (!deg[n.id] && G.nodes.length > 12) n.spec = 5; // loners -> start
      (perSpec[n.spec] || (perSpec[n.spec] = [])).push(n);
    });
    Object.keys(perSpec).forEach(function (sk) {
      var arr = perSpec[sk], s = +sk;
      var anch = (SPEC[s] || SPEC[5]).anchor;
      arr.forEach(function (n, i) {
        // golden-angle ball so each swarm reads as a dense cotton cluster
        var ga = i * 2.39996 + s;
        var rr = 5 + Math.sqrt(i) * 2.1 + Math.min(8, (n.visits || 0));
        n.p3 = [anch[0] + Math.cos(ga) * rr, anch[1] + ((i * 37) % 17) - 8, anch[2] + Math.sin(ga) * rr];
        n.dangling = (s === 5);
      });
    });
    G.nodes.forEach(function (n) { if (!n.p3) n.p3 = [0, 0, 0]; });
  }
  function rebuildWeb() {
    var THREE = window.THREE;
    if (G.webGroup) { G.scene.remove(G.webGroup); }
    G.webGroup = new THREE.Group();
    var n = G.nodes.length;
    if (!n) { G.scene.add(G.webGroup); return; }
    var pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    var c = new THREE.Color();
    G.nodes.forEach(function (nd, i) {
      pos[i * 3] = nd.p3[0]; pos[i * 3 + 1] = nd.p3[1]; pos[i * 3 + 2] = nd.p3[2];
      c.set(nd.flagged ? "#ff4d6d" : ((SPEC[nd.spec] || SPEC[5]).color));
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
    });
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    G.nodePts = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.6, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.95 }));
    G.webGroup.add(G.nodePts);
    // edges split silk vs wire
    function lineSet(filter, color, opacity) {
      var arr = [];
      G.edges.forEach(function (e) {
        var a = G.byId[e.a], b = G.byId[e.b];
        if (!a || !b || !a.p3 || !b.p3) return;
        if (filter(e)) arr.push(a.p3[0], a.p3[1], a.p3[2], b.p3[0], b.p3[1], b.p3[2]);
      });
      if (!arr.length) return null;
      var g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(arr), 3));
      var l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: new THREE.Color(color), transparent: true, opacity: opacity }));
      G.webGroup.add(l);
      return l;
    }
    G.silkLines = lineSet(function (e) { return !e.wire; }, "#49e0da", 0.16);
    G.wireLines = lineSet(function (e) { return !!e.wire; }, "#f6b84b", 0.6);
    // flagged nodes get real glowing spheres
    G.nodes.filter(function (x) { return x.flagged; }).slice(0, 40).forEach(function (nd) {
      var m = new THREE.Mesh(new THREE.SphereGeometry(1.1, 12, 10),
        new THREE.MeshStandardMaterial({ color: 0xf6b84b, emissive: 0xf6b84b, emissiveIntensity: 0.7 }));
      m.position.set(nd.p3[0], nd.p3[1], nd.p3[2]);
      G.webGroup.add(m);
    });
    G.scene.add(G.webGroup);
  }
  function syncSpiders() {
    var THREE = window.THREE;
    if (!G.spiderGroup) { G.spiderGroup = new THREE.Group(); G.scene.add(G.spiderGroup); }
    var seen = {};
    G.agents.slice(0, 12).forEach(function (ag) {
      seen[ag.id] = true;
      var s = G.spiders[ag.id];
      if (!s) {
        var grp = makeSpider(ag.color || "#49e0da");
        var label = makeLabel(ag.ghost ? ag.name + " ?" : ag.name, ag.color || "#49e0da");
        label.position.y = 6.5; grp.add(label);
        G.spiderGroup.add(grp);
        s = G.spiders[ag.id] = { grp: grp, label: label, minis: [], ag: ag };
      }
      s.ag = ag;
      // minis: one small flyer each, pooled
      var kids = (ag.children || []).slice(0, 8);
      while (s.minis.length < kids.length) {
        var mini = new THREE.Mesh(new THREE.OctahedronGeometry(0.7),
          new THREE.MeshStandardMaterial({ color: new THREE.Color(ag.color || "#49e0da"), emissive: new THREE.Color(ag.color || "#49e0da"), emissiveIntensity: 0.9 }));
        G.spiderGroup.add(mini); s.minis.push(mini);
      }
      s.wantMinis = kids.length;
    });
    Object.keys(G.spiders).forEach(function (id) {
      if (!seen[id]) {
        var s = G.spiders[id];
        G.spiderGroup.remove(s.grp);
        s.minis.forEach(function (m) { G.spiderGroup.remove(m); });
        delete G.spiders[id];
      }
    });
  }
  function anchorOf(ag) {
    if (ag.file && G.byId) {
      var k = Object.keys(G.byId).find(function (id) {
        var f = G.byId[id].full || "";
        return f === ag.file || f.replace(/\\/g, "/") === String(ag.file).replace(/\\/g, "/");
      });
      if (k) return G.byId[k];
    }
    return null;
  }
  function animate() {
    if (!G.shown) return;
    G.raf = requestAnimationFrame(animate);
    G.time += 0.016;
    var t = G.time;
    // spiders crawl toward anchors + bob; legs walk; minis orbit
    Object.keys(G.spiders).forEach(function (id, ai) {
      var s = G.spiders[id], ag = s.ag, g = s.grp;
      var an = anchorOf(ag), tx, ty, tz;
      if (an && an.p3) { tx = an.p3[0]; ty = an.p3[1] + 6; tz = an.p3[2]; }
      else { var a2 = ai * 2.4; tx = Math.cos(a2) * 26; ty = Math.sin(a2 * 1.3) * 12; tz = Math.sin(a2) * 26; }
      if (!ag.stale && !ag.ghost) { tx += Math.sin(t * 0.7 + ai) * 3; ty += Math.cos(t * 0.9 + ai * 2) * 2; }
      g.position.x += (tx - g.position.x) * 0.04;
      g.position.y += (ty - g.position.y) * 0.04;
      g.position.z += (tz - g.position.z) * 0.04;
      g.rotation.y = Math.sin(t * 0.4 + ai) * 0.6;
      var legs = g.userData.legs || [];
      legs.forEach(function (L) {
        if (ag.stale) return;
        var sw = Math.sin(t * 6 + L.phase) * 0.35;
        L.hip.rotation.x = sw;
        L.knee.rotation.x = Math.cos(t * 6 + L.phase) * 0.3;
      });
      for (var mi = 0; mi < s.minis.length; mi++) {
        var m = s.minis[mi];
        m.visible = mi < s.wantMinis;
        if (!m.visible) continue;
        var ka = t * 1.2 + (mi / Math.max(1, s.wantMinis)) * Math.PI * 2;
        m.position.set(g.position.x + Math.cos(ka) * 7, g.position.y + Math.sin(ka * 1.4) * 3, g.position.z + Math.sin(ka) * 7);
        m.rotation.y = ka;
      }
      var sc = ag.ghost ? 0.8 + Math.sin(t * 2) * 0.06 : 1;
      g.scale.set(sc, sc, sc);
    });
    if (G.wireLines) G.wireLines.material.opacity = 0.45 + Math.sin(t * 2.2) * 0.2;
    if (G.stars) G.stars.rotation.y = t * 0.004;
    G.controls.update();
    G.renderer.render(G.scene, G.camera);
  }
  window.Spider3D = {
    onSelect: null,
    visible: function () { return G.shown; },
    async init(canvasId) {
      var THREE = await ensureThree();
      var cv = document.getElementById(canvasId);
      if (!cv) return false;
      G.scene = new THREE.Scene();
      G.scene.background = new THREE.Color(0x050d0f);
      G.scene.fog = new THREE.FogExp2(0x050d0f, 0.0035);
      G.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
      G.camera.position.set(0, 42, 120);
      G.renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
      G.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
      var amb = new THREE.AmbientLight(0x8899aa, 0.7); G.scene.add(amb);
      var key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(60, 90, 40); G.scene.add(key);
      var rim = new THREE.DirectionalLight(0x49e0da, 0.5); rim.position.set(-70, -30, -60); G.scene.add(rim);
      // starfield
      var sp = [];
      for (var i = 0; i < 900; i++) sp.push((Math.random() - 0.5) * 700, (Math.random() - 0.5) * 400, (Math.random() - 0.5) * 700);
      var sg = new THREE.BufferGeometry();
      sg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(sp), 3));
      G.stars = new THREE.Points(sg, new THREE.PointsMaterial({ color: 0x3a5a60, size: 1.1 }));
      G.scene.add(G.stars);
      G.controls = new THREE.OrbitControls(G.camera, cv);
      G.controls.enableDamping = true; G.controls.dampingFactor = 0.06;
      G.ray = new THREE.Raycaster(); G.mouse = new THREE.Vector2();
      cv.addEventListener("click", function (ev) {
        var r = cv.getBoundingClientRect();
        G.mouse.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
        G.mouse.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
        G.ray.setFromCamera(G.mouse, G.camera);
        var hits = G.ray.intersectObjects(G.spiderGroup ? G.spiderGroup.children : [], true);
        if (hits.length && window.Spider3D.onSelect) {
          var o = hits[0].object, name = "";
          while (o) { if (o.type === "Group" && o.userData.legs) break; o = o.parent; }
          window.Spider3D.onSelect({ kind: o ? "agent" : "web", point: hits[0].point });
        }
      });
      const fit = () => {
        var w = cv.clientWidth || cv.parentElement.clientWidth, h = cv.clientHeight || 440;
        if (!w) return;
        G.renderer.setSize(w, h, false);
        G.camera.aspect = w / h; G.camera.updateProjectionMatrix();
      };
      fit(); window.addEventListener("resize", fit);
      G.ready = true;
      this.resize = fit;
      return true;
    },
    resize() {},
    setData(d) {
      if (!G.ready || !d || !d.trail) return;
      var nodes = d.nodes || [];
      G.nodes = nodes.map(function (n) { return { id: n.id, full: n.full, name: n.name, flagged: !!n.flagged, visits: n.visits || 0 }; });
      G.byId = {}; G.nodes.forEach(function (n) { G.byId[n.id] = n; });
      var e0 = d.edges || d.wiring || [];
      G.edges = e0.map(function (e) { return { a: e.a != null ? e.a : e.from, b: e.b != null ? e.b : e.to, wire: !!e.wire }; }).filter(function (e) { return e.a != null && e.b != null; });
      layout3D(); rebuildWeb();
    },
    setMemory(m) {
      if (!G.ready || !m || !m.nodes) return;
      var arr = Array.isArray(m.nodes) ? m.nodes : Object.keys(m.nodes).map(function (k, i) {
        return { id: i, full: k, name: (m.nodes[k].path || k).split(/[/\\]/).pop(), flagged: !!m.nodes[k].last_flag, visits: m.nodes[k].visits || 0 };
      });
      this.setData({ trail: arr, nodes: arr.slice(0, 1500), edges: [] });
    },
    setAgents(list) {
      G.agents = Array.isArray(list) ? list.slice(0, 12) : [];
      var hasAG = G.agents.some(function (a) { return String(a.id || "").indexOf("antigravity") >= 0; });
      if (!hasAG && G.agents.length < 12) G.agents.push({ id: "quillan-antigravity", name: "Quillan Antigravity", color: "#f6b84b", file: "", task: "ghost perch", children: [], stale: false, ghost: true });
      if (G.ready) syncSpiders();
    },
    setFocus() {},
    show() { G.shown = true; if (this.resize) this.resize(); if (G.ready) syncSpiders(); cancelAnimationFrame(G.raf); G.raf = requestAnimationFrame(animate); },
    hide() { G.shown = false; cancelAnimationFrame(G.raf); }
  };
})();
