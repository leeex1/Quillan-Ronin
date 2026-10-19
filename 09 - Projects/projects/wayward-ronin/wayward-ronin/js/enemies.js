// enemies.js — three hostile archetypes with distinct behavior:
//   ashigaru — clan soldier. Defensive: blocks frontal attacks, counters.
//   rusher   — rebel ronin. Aggressive: fast lunges, quick slash chains.
//   hatamoto — clan elite. Slow, hyper-armored, devastating overheads.
// Simple state machines: idle/chase/windup/strike/recover/stagger/block/dead.

import * as THREE from 'three';
import { createSamurai, poseIdle, poseWalk, poseSwing, poseBlock, poseStagger, poseDeath, resetSwordArm } from './characters.js';

const TYPES = {
  ashigaru: { hp: 62, dmg: 10, speed: 3.6, range: 2.3, windup: 0.5, recover: 0.55,
              robe: 0x5a1f1f, trim: 0xd8c69a, armor: true, name: 'Clan Ashigaru',
              xp: 1, blockChance: 0.45 },
  rusher:   { hp: 44, dmg: 8, speed: 5.4, range: 2.1, windup: 0.32, recover: 0.34,
              robe: 0x3a4038, trim: 0x8a8a7a, headband: 0x555a4a, name: 'Ashen Ronin',
              xp: 1, chain: true },
  hatamoto: { hp: 135, dmg: 22, speed: 2.6, range: 2.8, windup: 0.85, recover: 0.8,
              robe: 0x1a1a20, trim: 0xc0392b, armor: true, scale: 1.16, name: 'Hatamoto Elite',
              xp: 3, hyperarmor: true },
};

export class Enemy {
  constructor(scene, world, fx, audio, type, x, z, opts = {}) {
    this.scene = scene; this.world = world; this.fx = fx; this.audio = audio;
    this.def = TYPES[type];
    this.type = type;
    this.faction = opts.faction || 'clan'; // 'clan' | 'rebel'
    this.name = opts.name || this.def.name;
    this.drops = opts.drops || null; // v2: sword id dropped on defeat
    const rig = createSamurai({ robe: this.def.robe, trim: this.def.trim, armor: this.def.armor, headband: this.def.headband });
    this.rig = rig; this.parts = rig.parts;
    this.group = rig.group;
    if (this.def.scale) this.group.scale.setScalar(this.def.scale);
    this.group.position.set(x, 0, z);
    scene.add(this.group);

    this.maxHp = this.def.hp * (opts.hpMul || 1);
    this.hp = this.maxHp;
    this.state = 'idle';
    this.stateT = 0;
    this.yaw = Math.atan2(-x, -z);
    this.walkPhase = Math.random() * 6;
    this.didHit = false;
    this.staggerT = -1;
    this.deathT = -1;
    this.removeAt = -1;
    this.blockT = -1;
    this.aggroRange = opts.aggroRange || 26;
    this.home = new THREE.Vector3(x, 0, z);
    this.strafeDir = Math.random() < 0.5 ? 1 : -1;
    this.strafeT = 0;
    this.chainLeft = 0;
    this.dead = false;
    this.hitFlash = 0;
    this.buildHpBar();
  }

  get pos() { return this.group.position; }

  buildHpBar() {
    const g = new THREE.Group();
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(1.1, 0.13),
      new THREE.MeshBasicMaterial({ color: 0x0a0a0a, transparent: true, opacity: 0.7, depthWrite: false }));
    const fg = new THREE.Mesh(new THREE.PlaneGeometry(1.06, 0.09),
      new THREE.MeshBasicMaterial({ color: this.faction === 'clan' ? 0xc0392b : 0x7a9a5a, depthWrite: false }));
    fg.position.z = 0.001;
    g.add(bg, fg);
    g.position.y = 2.45 * (this.def.scale || 1);
    this.group.add(g);
    this.hpBar = g; this.hpFg = fg;
  }

  facePlayer(p) {
    const d = new THREE.Vector3().subVectors(p, this.pos);
    this.yaw = Math.atan2(d.x, d.z);
  }

  distTo(p) {
    const dx = p.x - this.pos.x, dz = p.z - this.pos.z;
    return Math.sqrt(dx * dx + dz * dz);
  }

  setState(s) { this.state = s; this.stateT = 0; }

  // Player's blade hit us. Returns damage actually dealt.
  takeDamage(dmg, fromPos, isHeavy, nowSec) {
    if (this.dead) return 0;
    const toAtk = new THREE.Vector3().subVectors(fromPos, this.pos).normalize();
    const facing = new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw));
    const frontal = toAtk.dot(facing) > 0.1;
    // ashigaru active block
    if (this.type === 'ashigaru' && frontal && (this.state === 'chase' || this.state === 'idle' || this.state === 'block')) {
      if (Math.random() < this.def.blockChance || this.state === 'block') {
        this.setState('block'); this.blockT = 0;
        this.audio.clash(false);
        const bp = this.pos.clone(); bp.y = 1.4;
        this.fx.burst(bp, { count: 8, color: 0x9fd8ff, speed: 4, life: 0.35, size: 0.07 });
        return dmg * 0.15; // chip
      }
    }
    this.hp -= dmg;
    this.hitFlash = 0.12;
    if (this.hp <= 0) { this.kill(); return dmg; }
    if (!this.def.hyperarmor || isHeavy) {
      this.staggerT = 0;
      this.setState('stagger');
      // knockback
      this.pos.addScaledVector(toAtk, isHeavy ? 1.6 : 0.7);
      this.world.collide(this.pos, 0.45);
    }
    return dmg;
  }

  kill() {
    this.dead = true;
    this.deathT = 0;
    this.setState('dead');
    this.hpBar.visible = false;
    const bp = this.pos.clone(); bp.y = 1.0;
    this.fx.burst(bp, { count: 18, color: 0xa03020, speed: 5, life: 0.7, size: 0.1 });
  }

  // Parry support: a perfectly-timed block staggers the attacker longer.
  applyParry() {
    this.staggerDur = 1.25;
    this.setState('stagger');
    resetSwordArm(this.parts);
  }

  update(dt, player, nowSec, camera) {
    const P = this.parts;
    if (this.hitFlash > 0) {
      this.hitFlash -= dt;
      P.blade.material.emissive = P.blade.material.emissive || new THREE.Color();
      this.parts.blade.material.emissive.setHex(this.hitFlash > 0 ? 0x661111 : 0x000000);
    }
    this.hpBar.lookAt(camera.position);
    this.hpFg.scale.x = Math.max(0.001, this.hp / this.maxHp);
    this.hpFg.position.x = -0.53 * (1 - this.hp / this.maxHp);

    if (this.dead) {
      this.deathT += dt;
      poseDeath(P, this.deathT);
      if (this.deathT > 2.6) {
        this.group.position.y -= dt * 0.5; // sink
        if (this.deathT > 3.6 && this.removeAt < 0) this.removeAt = nowSec;
      }
      return;
    }

    this.stateT += dt;
    const pp = player.pos;
    const dist = this.distTo(pp);
    const playerDead = player.dead;

    // --- stagger ---
    if (this.state === 'stagger') {
      const dur = this.staggerDur || 0.5;
      poseStagger(P, Math.min(1, this.stateT / dur));
      if (this.stateT > dur) { this.staggerDur = 0.5; this.setState('chase'); resetSwordArm(P); }
      this.group.rotation.y = this.yaw;
      return;
    }
    // --- block hold ---
    if (this.state === 'block') {
      poseBlock(P);
      this.facePlayer(pp);
      this.group.rotation.y = this.yaw;
      if (this.stateT > 0.7) {
        // counterattack
        this.setState(Math.random() < 0.6 && dist < this.def.range + 0.6 ? 'windup' : 'chase');
        resetSwordArm(P);
      }
      return;
    }

    if (playerDead) { // lose interest
      if (this.state !== 'idle') { this.setState('idle'); resetSwordArm(P); }
      poseIdle(P, nowSec);
      return;
    }

    switch (this.state) {
      case 'idle': {
        poseIdle(P, nowSec + this.walkPhase);
        if (dist < this.aggroRange) this.setState('chase');
        break;
      }
      case 'chase': {
        this.facePlayer(pp);
        if (dist > this.aggroRange * 1.6) { this.setState('idle'); break; }
        if (dist <= this.def.range) { this.setState('windup'); this.didHit = false; break; }
        // approach, with occasional strafing for the defensive type
        let mx = 0, mz = 0;
        const dx = (pp.x - this.pos.x) / (dist || 1), dz = (pp.z - this.pos.z) / (dist || 1);
        if (this.type === 'ashigaru' && dist < 6) {
          this.strafeT += dt;
          if (this.strafeT > 1.6) { this.strafeT = 0; this.strafeDir *= -1; }
          mx = -dz * this.strafeDir * 0.7 + dx * 0.7;
          mz = dx * this.strafeDir * 0.7 + dz * 0.7;
        } else {
          // rusher lunges in bursts
          const sp = this.type === 'rusher' ? this.def.speed * (this.stateT % 1.2 < 0.5 ? 1.7 : 0.7) : this.def.speed;
          mx = dx * sp / this.def.speed; mz = dz * sp / this.def.speed;
          this.pos.x += dx * sp * dt; this.pos.z += dz * sp * dt;
        }
        if (this.type === 'ashigaru' && dist < 6) {
          const sp = this.def.speed * 0.8;
          this.pos.x += mx * sp * dt; this.pos.z += mz * sp * dt;
        }
        this.walkPhase += dt * 10;
        poseWalk(P, this.walkPhase, 1);
        this.world.collide(this.pos, 0.45);
        break;
      }
      case 'windup': {
        this.facePlayer(pp);
        // telegraph: blade glows
        this.parts.blade.material.emissive.setHex(0x771111);
        poseSwing(P, Math.min(0.3, this.stateT / this.def.windup) * 0.5, 'oh');
        if (this.stateT >= this.def.windup) {
          this.parts.blade.material.emissive.setHex(0x000000);
          this.strikeStyle = this.type === 'hatamoto' ? 'oh' : this.type === 'rusher' ? (Math.random() < 0.5 ? 'h1' : 'h2') : 'h1';
          this.setState('strike'); this.didHit = false;
          this.audio.swing(this.type === 'hatamoto');
        }
        break;
      }
      case 'strike': {
        const dur = this.type === 'hatamoto' ? 0.5 : this.type === 'rusher' ? 0.3 : 0.38;
        const t = Math.min(1, this.stateT / dur);
        poseSwing(P, t, this.strikeStyle || 'h1');
        // the actual hit test happens in main.js via strikeLanded()
        if (t >= 1) {
          this.setState('recover');
          resetSwordArm(P);
          if (this.type === 'rusher' && this.chainLeft <= 0 && Math.random() < 0.5 && this.distTo(pp) < this.def.range + 0.5) {
            this.chainLeft = 1; // chain a second slash
          }
        }
        break;
      }
      case 'recover': {
        poseIdle(P, nowSec);
        if (this.stateT >= this.def.recover) {
          if (this.chainLeft > 0) { this.chainLeft--; this.setState('windup'); }
          else this.setState('chase');
        }
        break;
      }
    }
    this.group.rotation.y = this.yaw;
  }

  // Called by main during the enemy's strike window. Returns true once per strike
  // if the player is in range — main then resolves block/parry/damage.
  strikeLanded(player) {
    if (this.state !== 'strike' || this.didHit) return false;
    const inWindow = this.stateT > 0.08;
    if (!inWindow) return false;
    if (this.distTo(player.pos) > this.def.range + 0.5) return false;
    this.didHit = true;
    return true;
  }
}
