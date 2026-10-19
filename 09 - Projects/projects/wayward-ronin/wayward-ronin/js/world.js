// world.js — the town of Kirisato at dusk. Gradient sky dome, low warm sun
// with long shadows, fog, paper lanterns with flickering glow, drifting
// petals/dust, wooden houses with tiled roofs, torii gate, well, market
// stalls, clan manor, and rice storehouse. All original, all procedural.

import * as THREE from 'three';

function canvasTexture(draw, w = 256, h = 256) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class World {
  constructor(opts = {}) {
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x3a2438, 45, 165);
    this.colliders = [];   // {minX,maxX,minZ,maxZ}
    this.lanterns = [];    // {light, base, phase, glowMat}
    this.bounds = 58;
    this.shadowSize = opts.shadowSize || 2048; // v2: smaller on touch GPUs
    this.nightK = 0;       // v2: 0 = dusk, 1 = full night
    this.nightTarget = 0;
    this.lanternMul = 1;   // v2: lanterns burn brighter at night
    this.buildSky();
    this.buildLights();
    this.buildGround();
    this.buildTown();
    this.buildDocks();     // v2: riverside district east of town
    this.buildParticles();
  }

  addCollider(x, z, w, d) {
    this.colliders.push({ minX: x - w / 2, maxX: x + w / 2, minZ: z - d / 2, maxZ: z + d / 2 });
  }

  // circle (x,z,r) vs AABB list; mutates pos to resolve. Returns hit bool.
  collide(pos, r) {
    let hit = false;
    for (const c of this.colliders) {
      const nx = Math.max(c.minX, Math.min(pos.x, c.maxX));
      const nz = Math.max(c.minZ, Math.min(pos.z, c.maxZ));
      const dx = pos.x - nx, dz = pos.z - nz;
      const d2 = dx * dx + dz * dz;
      if (d2 < r * r) {
        hit = true;
        if (d2 > 1e-8) {
          const d = Math.sqrt(d2);
          pos.x = nx + (dx / d) * r;
          pos.z = nz + (dz / d) * r;
        } else {
          // center inside box: push along smallest penetration axis
          const pl = pos.x - c.minX, pr = c.maxX - pos.x;
          const pn = pos.z - c.minZ, pf = c.maxZ - pos.z;
          const m = Math.min(pl, pr, pn, pf);
          if (m === pl) pos.x = c.minX - r;
          else if (m === pr) pos.x = c.maxX + r;
          else if (m === pn) pos.z = c.minZ - r;
          else pos.z = c.maxZ + r;
        }
      }
    }
    // world bounds
    const B = this.bounds;
    pos.x = Math.max(-B, Math.min(B, pos.x));
    pos.z = Math.max(-B, Math.min(B, pos.z));
    return hit;
  }

  buildSky() {
    // gradient dome: burnt orange horizon -> violet -> deep indigo zenith
    const geo = new THREE.SphereGeometry(400, 24, 16);
    const top = new THREE.Color(0x1a1440), mid = new THREE.Color(0x8a3a5a), hor = new THREE.Color(0xff9a4d);
    const cols = [];
    const p = geo.getAttribute('position');
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i) / 400; // -1..1
      const c = new THREE.Color();
      if (y < 0.12) c.copy(hor).lerp(mid, Math.max(0, (y + 0.08) / 0.2));
      else c.copy(mid).lerp(top, Math.min(1, (y - 0.12) / 0.55));
      cols.push(c.r, c.g, c.b);
    }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    this.skyGeo = geo; // v2: kept so night can lerp the vertex colors
    const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false });
    this.scene.add(new THREE.Mesh(geo, mat));
    // low sun disc
    const sun = new THREE.Mesh(
      new THREE.CircleGeometry(16, 24),
      new THREE.MeshBasicMaterial({ color: 0xffd9a0, fog: false, transparent: true, opacity: 0.95 })
    );
    sun.position.set(-190, 26, -260);
    sun.lookAt(0, 20, 0);
    this.scene.add(sun);
    this.sunDiscMat = sun.material; // v2: fades out as night falls
    // v2: pale moon, hidden until night
    const moon = new THREE.Mesh(
      new THREE.CircleGeometry(11, 24),
      new THREE.MeshBasicMaterial({ color: 0xdfe6ff, fog: false, transparent: true, opacity: 0 })
    );
    moon.position.set(170, 70, -230);
    moon.lookAt(0, 20, 0);
    this.scene.add(moon);
    this.moonMat = moon.material;
    const halo = new THREE.Mesh(
      new THREE.CircleGeometry(30, 24),
      new THREE.MeshBasicMaterial({ color: 0xff9a4d, fog: false, transparent: true, opacity: 0.28, blending: THREE.AdditiveBlending, depthWrite: false })
    );
    halo.position.copy(sun.position).add(new THREE.Vector3(0, 0, -1));
    halo.lookAt(0, 20, 0);
    this.scene.add(halo);
  }

  buildLights() {
    // dusk key light: low warm sun, long shadows
    this.sun = new THREE.DirectionalLight(0xff9a55, 1.6);
    this.sun.position.set(-60, 26, -40);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(this.shadowSize, this.shadowSize);
    this.sun.shadow.camera.left = -70; this.sun.shadow.camera.right = 70;
    this.sun.shadow.camera.top = 70; this.sun.shadow.camera.bottom = -70;
    this.sun.shadow.camera.far = 220;
    this.sun.shadow.bias = -0.0008;
    this.scene.add(this.sun, this.sun.target);
    // cool violet fill from the east (dusk sky bounce)
    this.hemi = new THREE.HemisphereLight(0x5a4a8a, 0x2a1f18, 0.55);
    this.scene.add(this.hemi);
    const fill = new THREE.DirectionalLight(0x4a5a9a, 0.25);
    fill.position.set(50, 30, 40);
    this.scene.add(fill);
    this.fill = fill;
  }

  buildGround() {
    const tex = canvasTexture((g, w, h) => {
      g.fillStyle = '#4a3a28'; g.fillRect(0, 0, w, h);
      for (let i = 0; i < 9000; i++) {
        const v = 58 + Math.random() * 42;
        g.fillStyle = `rgb(${v + 14},${v},${v * 0.72})`;
        g.fillRect(Math.random() * w, Math.random() * h, 2.5, 2.5);
      }
      // worn path through the middle
      g.fillStyle = 'rgba(122,100,66,0.55)';
      g.fillRect(w * 0.44, 0, w * 0.12, h);
      g.fillRect(0, h * 0.44, w, h * 0.12);
    });
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(10, 10);
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(140, 140),
      new THREE.MeshStandardMaterial({ map: tex, roughness: 1 })
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);
    // scattered stones
    const stoneGeo = new THREE.DodecahedronGeometry(0.22, 0);
    const stoneMat = new THREE.MeshStandardMaterial({ color: 0x5a5248, roughness: 1 });
    for (let i = 0; i < 40; i++) {
      const s = new THREE.Mesh(stoneGeo, stoneMat);
      s.position.set((Math.random() - 0.5) * 110, 0.08, (Math.random() - 0.5) * 110);
      s.scale.setScalar(0.5 + Math.random());
      s.rotation.set(Math.random() * 3, Math.random() * 3, 0);
      s.castShadow = true;
      this.scene.add(s);
    }
  }

  woodMat(color = 0x4a3423) {
    return new THREE.MeshStandardMaterial({ color, roughness: 0.9 });
  }

  makeHouse(x, z, w, d, h, opts = {}) {
    const g = new THREE.Group();
    const wallC = opts.wall || 0x8a6f4d;
    const walls = new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
      new THREE.MeshStandardMaterial({ color: wallC, roughness: 0.95 }));
    walls.position.y = h / 2; walls.castShadow = walls.receiveShadow = true;
    g.add(walls);
    // tiled roof: two sloped boxes + ridge
    const roofC = opts.roof || 0x2f2a33;
    const roofMat = new THREE.MeshStandardMaterial({ color: roofC, roughness: 0.85 });
    const slope = new THREE.BoxGeometry(w + 1.6, 0.18, d * 0.72);
    for (const s of [-1, 1]) {
      const r = new THREE.Mesh(slope, roofMat);
      r.position.set(0, h + d * 0.19, s * d * 0.245);
      r.rotation.x = s * 0.62;
      r.castShadow = true;
      g.add(r);
    }
    const ridge = new THREE.Mesh(new THREE.BoxGeometry(w + 1.6, 0.22, 0.3), roofMat);
    ridge.position.y = h + d * 0.36; ridge.castShadow = true;
    g.add(ridge);
    // warm window glow
    if (opts.lit !== false) {
      const winMat = new THREE.MeshBasicMaterial({ color: 0xffc06a });
      for (const s of [-1, 1]) {
        const win = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.22, h * 0.3), winMat);
        win.position.set(s * w * 0.25, h * 0.55, d / 2 + 0.01);
        g.add(win);
      }
    }
    // noren curtain over door
    const door = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.1),
      new THREE.MeshStandardMaterial({ color: opts.noren || 0x3a5a7a, roughness: 1, side: THREE.DoubleSide }));
    door.position.set(0, 1.35, d / 2 + 0.02);
    g.add(door);
    g.position.set(x, 0, z);
    if (opts.ry) g.rotation.y = opts.ry;
    this.scene.add(g);
    this.addCollider(x, z, w + 0.6, d + 0.6);
    return g;
  }

  makeTorii(x, z, ry = 0) {
    const g = new THREE.Group();
    const mat = this.woodMat(0x8a2a1a);
    for (const s of [-1, 1]) {
      const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.42, 7, 10), mat);
      pillar.position.set(s * 3, 3.5, 0); pillar.castShadow = true;
      g.add(pillar);
    }
    const top = new THREE.Mesh(new THREE.BoxGeometry(9.4, 0.7, 0.9), mat);
    top.position.y = 7.1; top.castShadow = true; g.add(top);
    const top2 = new THREE.Mesh(new THREE.BoxGeometry(10.2, 0.35, 1.1), this.woodMat(0x1a1512));
    top2.position.y = 7.6; top2.castShadow = true; g.add(top2);
    const tie = new THREE.Mesh(new THREE.BoxGeometry(6.6, 0.5, 0.6), mat);
    tie.position.y = 5.6; tie.castShadow = true; g.add(tie);
    g.position.set(x, 0, z); g.rotation.y = ry;
    this.scene.add(g);
    this.addCollider(x - 3, z, 0.9, 1.2);
    this.addCollider(x + 3, z, 0.9, 1.2);
  }

  makeLantern(x, z, h = 2.6) {
    const g = new THREE.Group();
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, h, 8), this.woodMat(0x2a2018));
    post.position.y = h / 2; post.castShadow = true; g.add(post);
    const paperMat = new THREE.MeshBasicMaterial({ color: 0xffd9a0 });
    const paper = new THREE.Mesh(new THREE.SphereGeometry(0.34, 12, 10), paperMat);
    paper.scale.y = 1.25; paper.position.y = h + 0.15;
    g.add(paper);
    const cap = new THREE.Mesh(new THREE.ConeGeometry(0.45, 0.25, 8), this.woodMat(0x1a1512));
    cap.position.y = h + 0.62; g.add(cap);
    const light = new THREE.PointLight(0xff9a4d, 14, 16, 1.8);
    light.position.y = h + 0.15;
    g.add(light);
    // additive glow sprite
    const glowMat = new THREE.SpriteMaterial({
      map: canvasTexture((gg, w, hh) => {
        const grad = gg.createRadialGradient(w/2, hh/2, 2, w/2, hh/2, w/2);
        grad.addColorStop(0, 'rgba(255,190,110,0.85)');
        grad.addColorStop(1, 'rgba(255,150,60,0)');
        gg.fillStyle = grad; gg.fillRect(0, 0, w, hh);
      }, 64, 64),
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const glow = new THREE.Sprite(glowMat);
    glow.scale.setScalar(2.6);
    glow.position.y = h + 0.15;
    g.add(glow);
    g.position.set(x, 0, z);
    this.scene.add(g);
    this.lanterns.push({ light, base: 14, phase: Math.random() * 10, glowMat });
    this.addCollider(x, z, 0.4, 0.4);
  }

  makeStall(x, z, ry, awningColor) {
    const g = new THREE.Group();
    const top = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.12, 2.2),
      new THREE.MeshStandardMaterial({ color: awningColor, roughness: 1, side: THREE.DoubleSide }));
    top.position.y = 2.3; top.rotation.z = 0.08; top.castShadow = true; g.add(top);
    for (const sx of [-1.4, 1.4]) for (const sz of [-0.9, 0.9]) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.3, 6), this.woodMat());
      pole.position.set(sx, 1.15, sz); g.add(pole);
    }
    const counter = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.9, 1.6), this.woodMat(0x5a4028));
    counter.position.y = 0.45; counter.castShadow = true; g.add(counter);
    // goods: little boxes / jars
    for (let i = 0; i < 5; i++) {
      const good = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.3, 0.35),
        new THREE.MeshStandardMaterial({ color: [0xa8894e, 0x7a5a3a, 0x8a2a1a][i % 3], roughness: 1 }));
      good.position.set(-1.1 + i * 0.55, 1.05, (Math.random() - 0.5) * 0.6);
      good.castShadow = true; g.add(good);
    }
    g.position.set(x, 0, z); g.rotation.y = ry;
    this.scene.add(g);
    this.addCollider(x, z, 3.4, 2.4);
  }

  makeWell(x, z) {
    const g = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.2, 0.9, 12, 1, true),
      new THREE.MeshStandardMaterial({ color: 0x6a625a, roughness: 1, side: THREE.DoubleSide }));
    ring.position.y = 0.45; ring.castShadow = true; g.add(ring);
    for (const s of [-1, 1]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.2, 6), this.woodMat());
      post.position.set(s * 0.9, 1.1, 0); g.add(post);
    }
    const roof = new THREE.Mesh(new THREE.ConeGeometry(1.5, 0.8, 4), this.woodMat(0x2f2a33));
    roof.position.y = 2.5; roof.rotation.y = Math.PI / 4; roof.castShadow = true; g.add(roof);
    g.position.set(x, 0, z);
    this.scene.add(g);
    this.addCollider(x, z, 2.6, 2.6);
  }

  makeBanner(x, z, ry, color, emblem) {
    // war banner on a pole with a clan emblem drawn on canvas
    const g = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 6.5, 8), this.woodMat(0x2a2018));
    pole.position.y = 3.25; pole.castShadow = true; g.add(pole);
    const cloth = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 3.4, 1, 6),
      new THREE.MeshStandardMaterial({
        map: canvasTexture((gg, w, h) => {
          gg.fillStyle = color; gg.fillRect(0, 0, w, h);
          gg.strokeStyle = '#e8dcc4'; gg.lineWidth = 10;
          gg.beginPath(); gg.arc(w/2, h*0.32, w*0.26, 0, Math.PI * 2); gg.stroke();
          if (emblem === 'diamond') {
            gg.beginPath();
            gg.moveTo(w/2, h*0.32 - w*0.16); gg.lineTo(w/2 + w*0.16, h*0.32);
            gg.lineTo(w/2, h*0.32 + w*0.16); gg.lineTo(w/2 - w*0.16, h*0.32);
            gg.closePath(); gg.fillStyle = '#e8dcc4'; gg.fill();
          } else { // triple-slash for the rebels
            gg.lineWidth = 14;
            for (let i = -1; i <= 1; i++) {
              gg.beginPath(); gg.moveTo(w/2 + i * 34 - 14, h*0.32 - 30);
              gg.lineTo(w/2 + i * 34 + 14, h*0.32 + 30); gg.stroke();
            }
          }
        }), side: THREE.DoubleSide, roughness: 1,
      }));
    cloth.position.set(0.8, 4.4, 0);
    cloth.castShadow = true;
    g.add(cloth);
    g.position.set(x, 0, z); g.rotation.y = ry;
    this.scene.add(g);
    this._banners = this._banners || [];
    this._banners.push(cloth);
    this.addCollider(x, z, 0.5, 0.5);
  }

  buildTown() {
    // torii at the south entrance (player spawn)
    this.makeTorii(0, 44);
    // clan manor — large, imposing, north
    this.makeHouse(0, -34, 16, 10, 5.5, { wall: 0x5a4a5e, roof: 0x1f1a22, noren: 0x7a1f1f, lit: true });
    this.makeBanner(-10, -28, 0.4, '#7a1f1f', 'diamond');
    this.makeBanner(10, -28, -0.4, '#7a1f1f', 'diamond');
    // rice storehouse — east, the rebels' target
    this.makeHouse(30, -6, 10, 7, 4, { wall: 0x9a7a4d, roof: 0x3a3226, noren: 0x4a5a3a });
    this.storehousePos = new THREE.Vector3(30, 0, -6);
    // market stalls — west plaza
    this.makeStall(-20, 6, 0.3, 0x3a5a7a);
    this.makeStall(-24, -2, -0.2, 0x7a3a2a);
    this.makeStall(-17, -9, 0.5, 0x4a6a3a);
    // houses ringing the town
    this.makeHouse(-34, 20, 8, 6, 3.6, { noren: 0x5a3a6a });
    this.makeHouse(-36, -18, 7, 6, 3.4, { noren: 0x3a5a7a });
    this.makeHouse(24, 22, 8, 6, 3.8, { noren: 0x6a5a2a });
    this.makeHouse(36, 14, 7, 6, 3.4, { noren: 0x2a5a4a });
    this.makeHouse(-14, 30, 7, 6, 3.5, { noren: 0x7a4a2a });
    this.makeHouse(16, 32, 8, 6, 3.6, { noren: 0x4a3a5a });
    this.makeHouse(-30, -32, 8, 6, 3.6, { noren: 0x5a2a3a });
    this.makeHouse(28, -26, 7, 6, 3.4, { noren: 0x3a4a5a });
    // rebel banner hidden in an alley (story flavor)
    this.makeBanner(-27, -13, 1.2, '#2f3a2f', 'slashes');
    // central well
    this.makeWell(2, 8);
    // lanterns along the paths
    const L = [
      [-6, 40], [6, 40], [-6, 28], [6, 28], [-6, 16], [6, 16],
      [-12, 4], [12, 2], [-12, -8], [12, -10], [-6, -20], [6, -20],
      [-16, 12], [16, 12], [-24, -14], [24, 2], [-8, -30], [8, -30],
    ];
    for (const [x, z] of L) this.makeLantern(x, z);
    // dead trees (silhouettes against the dusk)
    for (let i = 0; i < 10; i++) {
      const t = new THREE.Group();
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.3, 4.5, 7), this.woodMat(0x241a12));
      trunk.position.y = 2.25; trunk.castShadow = true; t.add(trunk);
      for (let b = 0; b < 4; b++) {
        const br = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.09, 2.2, 5), this.woodMat(0x241a12));
        br.position.set((Math.random() - 0.5) * 1.6, 3.6 + Math.random(), (Math.random() - 0.5) * 1.6);
        br.rotation.set(Math.random() * 1.2 - 0.6, 0, Math.random() * 1.4 - 0.7);
        br.castShadow = true; t.add(br);
      }
      const a = (i / 10) * Math.PI * 2;
      const tx = Math.cos(a) * 52, tz = Math.sin(a) * 52;
      if (tx > 44) continue; // v2: keep the river clear
      t.position.set(tx, 0, tz);
      this.scene.add(t);
    }
  }

  // v2: riverside docks district — east of the storehouse. Warehouses, a
  // wooden dock over dark animated water, moored boats, crates, lanterns.
  buildDocks() {
    // river — dark animated water along the east edge
    const riverTex = canvasTexture((g, w, h) => {
      g.fillStyle = '#16263e'; g.fillRect(0, 0, w, h);
      for (let i = 0; i < 260; i++) {
        g.strokeStyle = `rgba(140,170,220,${0.08 + Math.random() * 0.14})`;
        g.lineWidth = 1 + Math.random() * 2;
        const y = Math.random() * h, x = Math.random() * w, len = 12 + Math.random() * 30;
        g.beginPath(); g.moveTo(x, y); g.lineTo(x + len, y); g.stroke();
      }
    });
    riverTex.wrapS = riverTex.wrapT = THREE.RepeatWrapping;
    riverTex.repeat.set(2, 6);
    const river = new THREE.Mesh(
      new THREE.PlaneGeometry(26, 110),
      new THREE.MeshStandardMaterial({ map: riverTex, roughness: 0.35, metalness: 0.45 })
    );
    river.rotation.x = -Math.PI / 2;
    river.position.set(61, -0.12, -5);
    this.scene.add(river);
    this.riverTex = riverTex;
    // muddy bank where water meets land
    const bank = new THREE.Mesh(
      new THREE.BoxGeometry(3, 0.3, 110),
      new THREE.MeshStandardMaterial({ color: 0x3a2f22, roughness: 1 })
    );
    bank.position.set(47.5, -0.05, -5);
    bank.receiveShadow = true;
    this.scene.add(bank);
    // wooden dock deck reaching over the water
    const dockMat = this.woodMat(0x5a4630);
    const deck = new THREE.Mesh(new THREE.BoxGeometry(9, 0.14, 7), dockMat);
    deck.position.set(46, 0.1, -4);
    deck.castShadow = deck.receiveShadow = true;
    this.scene.add(deck);
    for (let i = 0; i < 4; i++) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.16, 1.6, 7), dockMat);
      post.position.set(42.6 + i * 2.3, -0.4, -0.9);
      this.scene.add(post);
      const post2 = post.clone();
      post2.position.z = -7.1;
      this.scene.add(post2);
    }
    // warehouses
    this.makeHouse(38, -20, 9, 7, 4.5, { wall: 0x6a5a3a, roof: 0x2a2620, noren: 0x4a5a3a });
    this.makeHouse(40, 10, 8, 6, 4, { wall: 0x5a4a3a, roof: 0x2a2620, noren: 0x3a4a5a });
    // cargo crates (shared geometry/material)
    const crateGeo = new THREE.BoxGeometry(1, 1, 1);
    const crateMat = new THREE.MeshStandardMaterial({ color: 0x7a5f36, roughness: 1 });
    for (const [x, z, s, ry] of [[44, -12, 1, 0.3], [45.2, -11, 0.8, 1.1], [44.5, -10.4, 0.65, 0.7], [37, 2, 1.1, 0.2], [38.2, 3.1, 0.9, 0.9]]) {
      const m = new THREE.Mesh(crateGeo, crateMat);
      m.position.set(x, s / 2, z);
      m.scale.setScalar(s);
      m.rotation.y = ry;
      m.castShadow = m.receiveShadow = true;
      this.scene.add(m);
      this.addCollider(x, z, s + 0.25, s + 0.25);
    }
    // moored boats (visual, beyond the water's edge)
    for (const [bx, bz, ry] of [[55, -12, 0.25], [56, 6, -0.2]]) {
      const boat = new THREE.Group();
      const hullMat = this.woodMat(0x4a3826);
      const hull = new THREE.Mesh(new THREE.BoxGeometry(4.6, 1, 1.9), hullMat);
      hull.position.y = 0.5; hull.castShadow = true; boat.add(hull);
      const rim = new THREE.Mesh(new THREE.BoxGeometry(4.9, 0.18, 2.2), this.woodMat(0x5a4630));
      rim.position.y = 1.05; boat.add(rim);
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 4.4, 7), hullMat);
      mast.position.y = 3; boat.add(mast);
      const sail = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 1.6, 2.6),
        new THREE.MeshStandardMaterial({ color: 0xcbb98f, roughness: 1 })
      );
      sail.position.y = 3.4; boat.add(sail);
      boat.position.set(bx, -0.1, bz);
      boat.rotation.y = ry;
      this.scene.add(boat);
    }
    // lanterns along the waterfront
    for (const [x, z] of [[42, -14], [48, -8], [40, -23]]) this.makeLantern(x, z);
    // keep the player out of the water (dock deck itself stays walkable)
    this.addCollider(60, -5, 19, 110);
  }

  // v2: begin the dusk->night transition (lerped over ~10s in update).
  // instant=true snaps straight to night (used when loading a night save).
  setNight(instant = false) {
    this.nightTarget = 1;
    if (instant) { this.nightK = 1; this.applyNight(1); }
  }

  // v2: lerp every dusk parameter toward its night value by k (0..1).
  applyNight(k) {
    const p = this.skyGeo.getAttribute('position');
    const colAttr = this.skyGeo.getAttribute('color');
    const dTop = new THREE.Color(0x1a1440), dMid = new THREE.Color(0x8a3a5a), dHor = new THREE.Color(0xff9a4d);
    const nTop = new THREE.Color(0x050514), nMid = new THREE.Color(0x16162e), nHor = new THREE.Color(0x3a2030);
    const d = new THREE.Color(), n = new THREE.Color(), c = new THREE.Color();
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i) / 400;
      if (y < 0.12) {
        const f = Math.max(0, (y + 0.08) / 0.2);
        d.copy(dHor).lerp(dMid, f); n.copy(nHor).lerp(nMid, f);
      } else {
        const f = Math.min(1, (y - 0.12) / 0.55);
        d.copy(dMid).lerp(dTop, f); n.copy(nMid).lerp(nTop, f);
      }
      c.copy(d).lerp(n, k);
      colAttr.setXYZ(i, c.r, c.g, c.b);
    }
    colAttr.needsUpdate = true;
    this.sunDiscMat.opacity = 0.95 * (1 - k);
    this.moonMat.opacity = 0.9 * k;
    this.sun.intensity = 1.6 + (0.25 - 1.6) * k;
    this.sun.color.setHex(0xff9a55).lerp(new THREE.Color(0x8aa0ff), k);
    this.hemi.intensity = 0.55 + (0.22 - 0.55) * k;
    this.fill.intensity = 0.25 + (0.1 - 0.25) * k;
    this.scene.fog.color.setHex(0x3a2438).lerp(new THREE.Color(0x0b0b18), k);
    this.lanternMul = 1 + k * 0.9;
  }

  buildParticles() {
    // drifting petals / dust motes catching the lantern light
    const N = 320;
    const geo = new THREE.BufferGeometry();
    const p = new Float32Array(N * 3);
    this._moteVel = [];
    for (let i = 0; i < N; i++) {
      p[i * 3] = (Math.random() - 0.5) * 120;
      p[i * 3 + 1] = Math.random() * 9;
      p[i * 3 + 2] = (Math.random() - 0.5) * 120;
      this._moteVel.push(0.3 + Math.random() * 0.9);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
    this.motes = new THREE.Points(geo, new THREE.PointsMaterial({
      color: 0xffc890, size: 0.14, transparent: true, opacity: 0.65,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    this.motes.frustumCulled = false;
    this.scene.add(this.motes);
  }

  update(t, dt) {
    // v2: dusk -> night transition (~10s)
    if (this.nightK !== this.nightTarget) {
      const dir = Math.sign(this.nightTarget - this.nightK);
      this.nightK = Math.min(1, Math.max(0, this.nightK + dir * dt / 10));
      this.applyNight(this.nightK);
    }
    // v2: drifting river current
    if (this.riverTex) this.riverTex.offset.x += dt * 0.03;
    // lantern flicker
    for (const l of this.lanterns) {
      const f = 0.86 + 0.14 * Math.sin(t * 9 + l.phase) * Math.sin(t * 3.7 + l.phase * 2);
      l.light.intensity = l.base * f * this.lanternMul;
      l.glowMat.opacity = 0.75 * f;
    }
    // banner cloth ripple (cheap vertex sway via rotation)
    if (this._banners) {
      this._banners.forEach((b, i) => { b.rotation.y = Math.sin(t * 1.8 + i * 2) * 0.12; });
    }
    // drifting motes
    const p = this.motes.geometry.getAttribute('position');
    for (let i = 0; i < p.count; i++) {
      let x = p.getX(i) + this._moteVel[i] * dt * 1.4;
      let y = p.getY(i) + Math.sin(t * 0.8 + i) * dt * 0.25;
      if (x > 62) { x = -62; y = Math.random() * 9; }
      p.setXYZ(i, x, y, p.getZ(i) + Math.sin(t * 0.5 + i * 2) * dt * 0.6);
    }
    p.needsUpdate = true;
  }
}
