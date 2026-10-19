// fx.js — game feel: hit-stop, screen shake, sword trails, hit sparks,
// floating damage numbers, drifting petal/dust motes. This is where the
// combat "crunch" lives.

import * as THREE from 'three';

export class FX {
  constructor(scene, camera) {
    this.scene = scene;
    this.camera = camera;
    this.hitStopUntil = 0;      // performance.now() ms — while active, dt = 0
    this.shake = 0;             // current shake magnitude
    this.trails = [];           // active sword trails
    this.bursts = [];           // particle bursts
    this.floaters = [];         // damage numbers (HTML)
    this.floatLayer = document.getElementById('viewport');
  }

  // Freeze the world briefly on impact. Light hits ~55ms, heavy ~110ms,
  // parries ~140ms (that extra beat sells the parry).
  hitStop(ms) {
    this.hitStopUntil = Math.max(this.hitStopUntil, performance.now() + ms);
  }
  inHitStop() { return performance.now() < this.hitStopUntil; }

  addShake(mag) { this.shake = Math.min(1.2, this.shake + mag); }

  // Apply shake as a camera offset; called after the camera is positioned.
  applyShake(dt) {
    if (this.shake > 0.001) {
      const s = this.shake;
      this.camera.position.x += (Math.random() - 0.5) * s * 0.9;
      this.camera.position.y += (Math.random() - 0.5) * s * 0.6;
      this.camera.rotation.z += (Math.random() - 0.5) * s * 0.02;
      this.shake *= Math.pow(0.001, dt); // fast decay
    }
  }

  // ---- sword trail: a fading ribbon following the blade tip ----
  startTrail(getTip, color = 0xffe9b0) {
    const MAX = 14;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(MAX * 2 * 3);
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const idx = [];
    for (let i = 0; i < MAX - 1; i++) {
      const a = i * 2;
      idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
    geo.setIndex(idx);
    const mat = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.85, side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    const trail = { mesh, geo, mat, getTip, points: [], life: 0.32, maxLife: 0.32, MAX };
    this.trails.push(trail);
    return trail;
  }
  endTrail(trail) { if (trail) trail.ending = true; }

  updateTrails(dt) {
    for (let i = this.trails.length - 1; i >= 0; i--) {
      const t = this.trails[i];
      t.life -= dt;
      if (!t.ending && t.life > 0) {
        const p = t.getTip();
        t.points.unshift(p.clone());
        if (t.points.length > t.MAX) t.points.pop();
      } else {
        // fade out: drop oldest points
        t.points.pop();
      }
      if (t.points.length < 2 || t.life <= -0.25) {
        this.scene.remove(t.mesh);
        t.geo.dispose(); t.mat.dispose();
        this.trails.splice(i, 1);
        continue;
      }
      // rebuild ribbon: each point becomes a vertical quad sliver
      const posAttr = t.geo.getAttribute('position');
      const n = t.points.length;
      for (let j = 0; j < t.MAX; j++) {
        const p = t.points[Math.min(j, n - 1)];
        const q = t.points[Math.min(j + 1, n - 1)];
        // width direction: perpendicular to segment, mostly horizontal
        const dir = new THREE.Vector3().subVectors(q, p);
        const side = new THREE.Vector3(-dir.z, 0, dir.x).normalize().multiplyScalar(0.05 + 0.02 * j);
        posAttr.setXYZ(j * 2, p.x - side.x, p.y - side.y, p.z - side.z);
        posAttr.setXYZ(j * 2 + 1, p.x + side.x, p.y + side.y, p.z + side.z);
      }
      posAttr.needsUpdate = true;
      t.mat.opacity = Math.max(0, 0.85 * (t.life / t.maxLife));
    }
  }

  // ---- particle bursts (sparks, dust, blood-petal) ----
  burst(pos, { count = 14, color = 0xffd27a, speed = 6, life = 0.5, size = 0.09, gravity = -9 } = {}) {
    const geo = new THREE.BufferGeometry();
    const p = new Float32Array(count * 3);
    const vels = [];
    for (let i = 0; i < count; i++) {
      p[i * 3] = pos.x; p[i * 3 + 1] = pos.y; p[i * 3 + 2] = pos.z;
      const a = Math.random() * Math.PI * 2;
      const e = (Math.random() - 0.35) * Math.PI;
      const s = speed * (0.4 + Math.random() * 0.8);
      vels.push(new THREE.Vector3(Math.cos(a) * Math.cos(e) * s, Math.sin(e) * s + speed * 0.35, Math.sin(a) * Math.cos(e) * s));
    }
    geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
    const mat = new THREE.PointsMaterial({
      color, size, transparent: true, opacity: 1,
      blending: THREE.AdditiveBlending, depthWrite: false,
    });
    const pts = new THREE.Points(geo, mat);
    pts.frustumCulled = false;
    this.scene.add(pts);
    this.bursts.push({ pts, geo, mat, vels, life, maxLife: life, gravity });
  }

  updateBursts(dt) {
    for (let i = this.bursts.length - 1; i >= 0; i--) {
      const b = this.bursts[i];
      b.life -= dt;
      if (b.life <= 0) {
        this.scene.remove(b.pts); b.geo.dispose(); b.mat.dispose();
        this.bursts.splice(i, 1); continue;
      }
      const p = b.geo.getAttribute('position');
      for (let j = 0; j < b.vels.length; j++) {
        const v = b.vels[j];
        v.y += b.gravity * dt;
        p.setXYZ(j, p.getX(j) + v.x * dt, Math.max(0.02, p.getY(j) + v.y * dt), p.getZ(j) + v.z * dt);
      }
      p.needsUpdate = true;
      b.mat.opacity = b.life / b.maxLife;
    }
  }

  // ---- floating damage numbers (HTML overlay projected from 3D) ----
  floatText(worldPos, text, cls = '') {
    const el = document.createElement('div');
    el.className = 'floater ' + cls;
    el.textContent = text;
    this.floatLayer.appendChild(el);
    const v = worldPos.clone().project(this.camera);
    el.style.left = ((v.x * 0.5 + 0.5) * window.innerWidth) + 'px';
    el.style.top = ((-v.y * 0.5 + 0.5) * window.innerHeight) + 'px';
    this.floaters.push({ el, life: 0.9 });
    setTimeout(() => el.remove(), 950);
  }

  updateFloaters(dt) {
    for (let i = this.floaters.length - 1; i >= 0; i--) {
      const f = this.floaters[i];
      f.life -= dt;
      if (f.life <= 0) { f.el.remove(); this.floaters.splice(i, 1); continue; }
      f.el.style.transform = `translate(-50%, ${-(1 - f.life / 0.9) * 60}px)`;
      f.el.style.opacity = Math.min(1, f.life / 0.4);
    }
  }

  update(dt) {
    this.updateTrails(dt);
    this.updateBursts(dt);
    this.updateFloaters(dt);
  }
}
