// characters.js — stylized low-poly samurai built from primitives, with an
// articulated rig driven procedurally (walk cycle, sword swings, block,
// stagger, death). Original art direction: ink-brush minimalism, no mocap.

import * as THREE from 'three';

function box(w, h, d, color, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, d),
    new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.05 })
  );
  m.position.set(x, y, z);
  m.castShadow = true;
  return m;
}

// Builds a rigged figure. Returns { group, parts }.
// parts: torso, head, armL, armR (shoulder pivots), swordPivot, swordTip,
//        legL, legR (hip pivots), sash
export function createSamurai(opts = {}) {
  const {
    robe = 0x2e3a4c, trim = 0xc0392b, skin = 0xd9a877,
    headband = null, hat = false, armor = false,
  } = opts;
  const group = new THREE.Group();
  const parts = {};

  // legs (hakama — wide, dark)
  const hakama = box(0.5, 0.55, 0.34, 0x1c1a20, 0, 0.72, 0);
  group.add(hakama);
  for (const s of [-1, 1]) {
    const hip = new THREE.Group();
    hip.position.set(s * 0.14, 0.5, 0);
    const leg = box(0.15, 0.5, 0.17, 0x242228, 0, -0.25, 0);
    hip.add(leg);
    group.add(hip);
    parts[s < 0 ? 'legL' : 'legR'] = hip;
  }

  // torso (kimono)
  const torso = new THREE.Group();
  torso.position.set(0, 1.0, 0);
  const chest = box(0.56, 0.62, 0.34, robe, 0, 0.3, 0);
  torso.add(chest);
  // trim: collar V
  const collar = box(0.1, 0.5, 0.02, trim, 0, 0.32, 0.18);
  collar.rotation.z = 0.5;
  torso.add(collar);
  // obi sash
  const sash = box(0.58, 0.12, 0.36, 0x14100c, 0, 0.02, 0);
  torso.add(sash);
  if (armor) {
    const plate = box(0.6, 0.4, 0.38, 0x4a4a52, 0, 0.32, 0);
    plate.material.metalness = 0.6; plate.material.roughness = 0.4;
    torso.add(plate);
  }
  group.add(torso);
  parts.torso = torso;

  // head
  const headG = new THREE.Group();
  headG.position.set(0, 1.78, 0);
  const head = box(0.27, 0.3, 0.27, skin, 0, 0.1, 0);
  headG.add(head);
  const knot = box(0.1, 0.1, 0.1, 0x1a1410, 0, 0.3, -0.04);
  headG.add(knot);
  if (headband) {
    const band = box(0.29, 0.06, 0.29, headband, 0, 0.2, 0);
    headG.add(band);
  }
  if (hat) {
    // kasa (straw hat): cone + brim
    const brim = new THREE.Mesh(
      new THREE.ConeGeometry(0.42, 0.16, 10),
      new THREE.MeshStandardMaterial({ color: 0xa8894e, roughness: 0.95 })
    );
    brim.position.y = 0.32; brim.castShadow = true;
    headG.add(brim);
  }
  group.add(headG);
  parts.head = headG;

  // arms — shoulder pivots so swings rotate naturally
  for (const s of [-1, 1]) {
    const sh = new THREE.Group();
    sh.position.set(s * 0.36, 1.52, 0);
    const arm = box(0.14, 0.58, 0.15, robe, 0, -0.28, 0);
    sh.add(arm);
    const hand = box(0.12, 0.1, 0.12, skin, 0, -0.6, 0);
    sh.add(hand);
    group.add(sh);
    parts[s < 0 ? 'armL' : 'armR'] = sh;
  }

  // katana in right hand: pivot at the hand so the whole sword swings
  const swordPivot = new THREE.Group();
  swordPivot.position.set(0, -0.62, 0.02);
  parts.armR.add(swordPivot);
  const blade = box(0.045, 1.05, 0.014, 0xd8dce2, 0, 0.55, 0);
  blade.material.metalness = 0.85; blade.material.roughness = 0.25;
  swordPivot.add(blade);
  const edge = box(0.012, 1.0, 0.016, 0xffffff, 0.022, 0.55, 0);
  swordPivot.add(edge);
  const guard = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.07, 0.025, 10),
    new THREE.MeshStandardMaterial({ color: 0x2a2a30, metalness: 0.7, roughness: 0.4 })
  );
  guard.rotation.z = Math.PI / 2; guard.position.y = 0.02;
  swordPivot.add(guard);
  const grip = box(0.035, 0.28, 0.035, 0x14100c, 0, -0.13, 0);
  swordPivot.add(grip);
  const tip = new THREE.Object3D();
  tip.position.set(0, 1.08, 0);
  swordPivot.add(tip);
  // rest pose: blade points forward-down
  swordPivot.rotation.x = 1.35;
  parts.swordPivot = swordPivot;
  parts.swordTip = tip;
  parts.blade = blade;

  return { group, parts, height: 1.95 };
}

// ---------- procedural poses ----------
// walkPhase advances with distance; swingT is 0..1 through an attack.

export function poseIdle(p, t) {
  p.torso.rotation.set(0, 0, Math.sin(t * 1.4) * 0.015);
  p.torso.position.y = 1.0 + Math.sin(t * 1.4) * 0.008;
  p.armL.rotation.set(Math.sin(t * 1.4) * 0.05, 0, 0.12);
  p.armR.rotation.set(Math.sin(t * 1.4 + 1) * 0.05, 0, -0.12);
  p.head.rotation.y = Math.sin(t * 0.4) * 0.15;
  p.legL.rotation.x = 0; p.legR.rotation.x = 0;
  if (!p._swinging) p.swordPivot.rotation.x += (1.35 - p.swordPivot.rotation.x) * 0.2;
}

export function poseWalk(p, phase, speedK = 1) {
  const s = Math.sin(phase), c = Math.cos(phase);
  p.legL.rotation.x = s * 0.55 * speedK;
  p.legR.rotation.x = -s * 0.55 * speedK;
  p.armL.rotation.x = -s * 0.4 * speedK;
  p.armR.rotation.x = s * 0.3 * speedK;
  p.armL.rotation.z = 0.12; p.armR.rotation.z = -0.12;
  p.torso.position.y = 1.0 + Math.abs(c) * 0.035 * speedK;
  p.torso.rotation.y = s * 0.06;
  if (!p._swinging) p.swordPivot.rotation.x += (1.35 - p.swordPivot.rotation.x) * 0.25;
}

// swing arcs: each attack has a distinct blade path.
// t: 0..1. Sets arm/sword/torso rotation. Call with p._swinging = true.
export function poseSwing(p, t, style = 'h1') {
  p._swinging = true;
  const e = t < 0.35 ? 2 * t * t / 0.35 : 1 - Math.pow((t - 0.35) / 0.65, 2) * 0.9; // ease
  if (style === 'h1') {           // horizontal right-to-left
    p.torso.rotation.y = -0.7 + e * 1.1;
    p.armR.rotation.set(-1.2 + e * 0.5, 0, -1.1 + e * 0.6);
    p.swordPivot.rotation.x = 1.35;
    p.swordPivot.rotation.z = -1.4 + e * 2.6;
  } else if (style === 'h2') {    // horizontal left-to-right (backhand)
    p.torso.rotation.y = 0.7 - e * 1.1;
    p.armR.rotation.set(-1.2 + e * 0.5, 0, 1.1 - e * 0.6);
    p.swordPivot.rotation.x = 1.35;
    p.swordPivot.rotation.z = 1.4 - e * 2.6;
  } else if (style === 'oh') {    // overhead chop
    p.torso.rotation.x = -0.25 + e * 0.45;
    p.armR.rotation.set(-2.4 + e * 2.2, 0, -0.15);
    p.swordPivot.rotation.set(0.4, 0, 0);
  } else if (style === 'thrust') {// straight thrust
    p.torso.rotation.y = -0.5 + e * 0.6;
    p.torso.rotation.x = e * 0.18;
    p.armR.rotation.set(-1.5, 0, -0.1);
    p.armR.position.z = e * 0.35;
    p.swordPivot.rotation.set(1.57, 0, 0); // blade forward
  }
  if (t >= 1) p._swinging = false;
}

export function poseBlock(p) {
  p.armR.rotation.set(-1.1, 0, -0.9);
  p.armL.rotation.set(-1.0, 0, 0.7);
  p.swordPivot.rotation.set(0.2, 0, 1.35); // blade vertical across body
  p.torso.rotation.set(0.08, -0.15, 0);
  p.legL.rotation.x = 0.25; p.legR.rotation.x = -0.2;
}

export function poseStagger(p, t) {
  // t: 0..1 through the stagger
  p.torso.rotation.x = -0.35 * Math.sin(t * Math.PI);
  p.torso.rotation.z = 0.2 * Math.sin(t * Math.PI);
  p.head.rotation.x = -0.4 * Math.sin(t * Math.PI);
  p.armL.rotation.z = 0.7 * Math.sin(t * Math.PI);
  p.armR.rotation.z = -0.7 * Math.sin(t * Math.PI);
}

export function poseDeath(p, t) {
  // collapse: tip over backward, sink slightly
  const e = Math.min(1, t * 1.6);
  p.torso.rotation.x = -1.35 * e;
  p.torso.position.y = 1.0 - 0.55 * e;
  p.head.rotation.x = -0.5 * e;
  p.armL.rotation.z = 1.2 * e; p.armR.rotation.z = -1.2 * e;
  p.legL.rotation.x = 0.1 * e; p.legR.rotation.x = -0.1 * e;
  p.swordPivot.rotation.x = 1.35 + 0.6 * e;
}

export function resetSwordArm(p) {
  p._swinging = false;
  p.armR.position.z = 0;
}
