// npcs.js — townsfolk, clan, and rebels. Simple state machines
// (idle / wander / talk), canvas nameplates, and per-NPC dialogue keys.
// All names, characters, and dialogue are original.

import * as THREE from 'three';
import { createSamurai, poseIdle, poseWalk } from './characters.js';

function nameplate(text, color = '#e8dcc4') {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.font = '28px Georgia, serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.shadowColor = 'rgba(0,0,0,0.9)'; g.shadowBlur = 8;
  g.fillStyle = color;
  g.fillText(text, 128, 32);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false }));
  s.scale.set(2.4, 0.6, 1);
  return s;
}

// id, name, faction, look, home position, wander radius, dialogue key
export const NPC_DEFS = [
  { id: 'jiro', name: 'Elder Jiro', faction: 'town', robe: 0x4a3a52, trim: 0x8a7a5a, hat: true,
    home: [5, 13], wander: 4, tree: 'jiro_intro', plate: '#e8c96a' },
  { id: 'mei', name: 'Mei the Merchant', faction: 'town', robe: 0x6a4a6a, trim: 0xd4a24e,
    home: [-20, 9.5], wander: 3, tree: 'mei', plate: '#e8c96a' },
  { id: 'kiku', name: 'Kiku', faction: 'town', robe: 0x7a5a3a, trim: 0xc0392b, scale: 0.68,
    home: [-12, 27], wander: 6, tree: 'kiku', plate: '#e8c96a' },
  { id: 'gendo', name: 'Magistrate Gendo', faction: 'clan', robe: 0x2a1a3a, trim: 0xc0392b, armor: true,
    home: [0, -27], wander: 0, tree: 'gendo', plate: '#e08080' },
  { id: 'isamu', name: 'Captain Isamu', faction: 'clan', robe: 0x4a1f1f, trim: 0xd8c69a, armor: true,
    home: [6, -18], wander: 8, tree: 'isamu', plate: '#e08080' },
  { id: 'ren', name: 'Ren of the Ashen Blades', faction: 'rebel', robe: 0x2f3a34, trim: 0x8a8a7a, headband: 0x3a3f38,
    home: [-27, -11], wander: 3, tree: 'ren', plate: '#9dc08a' },
  { id: 'tsubaki', name: 'Tsubaki', faction: 'rebel', robe: 0x3a3f4a, trim: 0x6a7a8a, headband: 0x555a4a,
    home: [14, 18], wander: 14, tree: 'tsubaki', plate: '#9dc08a' },
  // v2: riverside docks district
  { id: 'souta', name: 'Fisherman Souta', faction: 'town', robe: 0x3a4a5a, trim: 0x8a7a5a, hat: true,
    home: [45, -3], wander: 3, tree: 'souta', plate: '#e8c96a' },
  { id: 'hana', name: 'Hana the Dockworker', faction: 'town', robe: 0x5a4a3a, trim: 0xd4a24e,
    home: [40, 4], wander: 5, tree: 'hana', plate: '#e8c96a' },
  { id: 'kage', name: 'Kage', faction: 'rebel', robe: 0x2a2a30, trim: 0x6a6a7a, headband: 0x1a1a1a,
    home: [42.5, -9], wander: 3, tree: 'kage', plate: '#9dc08a' },
  { id: 'kano', name: 'Inspector Kano', faction: 'clan', robe: 0x4a1f1f, trim: 0xd8c69a, armor: true,
    home: [37, -16], wander: 6, tree: 'kano', plate: '#e08080' },
];

export class NPC {
  constructor(scene, world, def) {
    this.def = def;
    this.id = def.id;
    const rig = createSamurai({ robe: def.robe, trim: def.trim, hat: def.hat, armor: def.armor, headband: def.headband });
    this.parts = rig.parts;
    this.group = rig.group;
    if (def.scale) this.group.scale.setScalar(def.scale);
    this.home = new THREE.Vector3(def.home[0], 0, def.home[1]);
    this.group.position.copy(this.home);
    scene.add(this.group);
    this.plate = nameplate(def.name, def.plate);
    this.plate.position.y = 2.5 * (def.scale || 1);
    this.group.add(this.plate);

    this.yaw = Math.random() * Math.PI * 2;
    this.state = 'idle';
    this.stateT = Math.random() * 3;
    this.target = new THREE.Vector3();
    this.walkPhase = Math.random() * 6;
    this.talking = false;
    this.hidden = false;
    this.world = world;
  }

  get pos() { return this.group.position; }

  hide() { this.hidden = true; this.group.visible = false; }
  show() { this.hidden = false; this.group.visible = true; }

  faceToward(p) {
    const dx = p.x - this.pos.x, dz = p.z - this.pos.z;
    this.yaw = Math.atan2(dx, dz);
  }

  update(dt, nowSec, camera) {
    if (this.hidden) return;
    this.plate.lookAt(camera.position);
    if (this.talking) {
      poseIdle(this.parts, nowSec);
      this.group.rotation.y = this.yaw;
      return;
    }
    this.stateT += dt;
    if (this.state === 'idle') {
      poseIdle(this.parts, nowSec + this.walkPhase);
      if (this.def.wander > 0 && this.stateT > 2 + Math.random() * 2) {
        const a = Math.random() * Math.PI * 2, r = Math.random() * this.def.wander;
        this.target.set(this.home.x + Math.cos(a) * r, 0, this.home.z + Math.sin(a) * r);
        this.state = 'walk'; this.stateT = 0;
      }
    } else if (this.state === 'walk') {
      const d = new THREE.Vector3().subVectors(this.target, this.pos);
      d.y = 0;
      const dist = d.length();
      if (dist < 0.4 || this.stateT > 8) { this.state = 'idle'; this.stateT = 0; }
      else {
        d.normalize();
        this.yaw = Math.atan2(d.x, d.z);
        this.pos.addScaledVector(d, 1.6 * dt);
        this.walkPhase += dt * 8;
        poseWalk(this.parts, this.walkPhase, 0.7);
        this.world.collide(this.pos, 0.4);
      }
    }
    this.group.rotation.y = this.yaw;
  }
}
