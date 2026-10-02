// player.js — the rōnin: WASD movement, sprint, 3-hit light combo, heavy
// attack, hold-to-block with timed parry, dodge roll with i-frames.
// All combat timing constants live here.

import * as THREE from 'three';
import { createSamurai, poseIdle, poseWalk, poseSwing, poseBlock, poseStagger, poseDeath, resetSwordArm } from './characters.js';

const LIGHTS = [
  { dur: 0.36, hitAt: [0.30, 0.62], dmg: 12, style: 'h1', lunge: 2.2, stop: 55, shake: 0.25 },
  { dur: 0.36, hitAt: [0.30, 0.62], dmg: 13, style: 'h2', lunge: 2.2, stop: 55, shake: 0.25 },
  { dur: 0.52, hitAt: [0.35, 0.65], dmg: 19, style: 'oh', lunge: 3.0, stop: 90, shake: 0.45 },
];
const HEAVY = { dur: 0.78, hitAt: [0.42, 0.68], dmg: 32, style: 'oh', lunge: 3.6, stop: 120, shake: 0.7, knockback: 9 };

export class Player {
  constructor(scene, world, fx, audio) {
    this.world = world; this.fx = fx; this.audio = audio;
    const rig = createSamurai({ robe: 0x2e3a4c, trim: 0xc0392b, headband: 0x8a1f1f });
    this.rig = rig; this.parts = rig.parts;
    this.group = rig.group;
    this.group.position.set(0, 0, 38); // south gate
    scene.add(this.group);

    this.maxHp = 100; this.hp = 100;
    this.maxStam = 100; this.stam = 100;
    this.yaw = Math.PI; // face north into town
    this.vel = new THREE.Vector3();
    this.walkPhase = 0;
    this.dead = false;

    // combat state
    this.atk = null;          // {kind:'light'|'heavy', stage, t, dur, def, didHit, queued}
    this.blocking = false;
    this.blockPressedAt = -10;
    this.parryWindow = 0.24;
    // v2: modified by the equipped sword (see swords.js)
    this.dmgMul = 1;
    this.speedMul = 1;
    this.trailColor = 0xffe9b0;
    this.dodgeT = -1; this.dodgeDir = new THREE.Vector3();
    this.iframesUntil = 0;
    this.staggerT = -1;
    this.trail = null;
    this.stepT = 0;
  }

  get pos() { return this.group.position; }
  isAttacking() { return !!this.atk; }
  isDodging() { return this.dodgeT >= 0; }
  invulnerable() { return performance.now() / 1000 < this.iframesUntil || this.dodgeT >= 0; }

  tryLight() {
    if (this.dead || this.staggerT >= 0 || this.dodgeT >= 0) return;
    if (!this.atk) this.startAttack('light', 0);
    else if (this.atk.kind === 'light' && this.atk.t / this.atk.dur > 0.45) this.atk.queued = true;
  }
  tryHeavy() {
    if (this.dead || this.staggerT >= 0 || this.dodgeT >= 0 || this.atk) return;
    this.startAttack('heavy', 0);
  }
  startAttack(kind, stage) {
    const def = kind === 'light' ? LIGHTS[stage] : HEAVY;
    this.atk = { kind, stage, t: 0, dur: def.dur, def, didHit: false, queued: false };
    resetSwordArm(this.parts);
    this.audio.swing(kind === 'heavy');
    // face the nearest threat or keep facing
    if (this.aimTarget) {
      const d = new THREE.Vector3().subVectors(this.aimTarget, this.pos);
      if (d.lengthSq() > 0.01) this.yaw = Math.atan2(d.x, d.z);
    }
    this.trail = this.fx.startTrail(() => {
      const v = new THREE.Vector3();
      this.parts.swordTip.getWorldPosition(v);
      return v;
    }, kind === 'heavy' ? 0xff9a5a : this.trailColor);
  }
  endAttack() {
    if (this.trail) { this.fx.endTrail(this.trail); this.trail = null; }
    this.atk = null;
    resetSwordArm(this.parts);
  }

  tryDodge(dir) {
    if (this.dead || this.dodgeT >= 0 || this.staggerT >= 0 || this.stam < 18) return;
    this.stam -= 18;
    if (this.atk) this.endAttack();
    this.dodgeT = 0;
    this.dodgeDir.copy(dir.lengthSq() > 0.01 ? dir : new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw)));
    this.iframesUntil = performance.now() / 1000 + 0.38;
    this.audio.dodge();
    this.fx.burst(this.pos.clone().add(new THREE.Vector3(0, 0.3, 0)), { count: 8, color: 0x8a7a5a, speed: 3, life: 0.4, size: 0.12, gravity: -2 });
  }

  setBlock(held, justPressed) {
    if (this.dead) { this.blocking = false; return; }
    if (justPressed) this.blockPressedAt = performance.now() / 1000;
    this.blocking = held && this.dodgeT < 0 && this.staggerT < 0 && !this.atk;
  }

  // Returns 'parried' | 'blocked' | 'broken' | 'hit'. nowSec for parry timing.
  takeDamage(dmg, fromPos, nowSec) {
    if (this.dead || this.invulnerable()) return 'dodged';
    const toAtk = new THREE.Vector3().subVectors(fromPos, this.pos).normalize();
    const facing = new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw));
    const frontal = toAtk.dot(facing) > -0.15;
    if (this.blocking && frontal) {
      const sincePress = nowSec - this.blockPressedAt;
      if (sincePress <= this.parryWindow && sincePress >= 0) {
        return 'parried'; // no damage, no stamina cost — pure timing reward
      }
      if (this.stam >= 16) {
        this.stam -= 16;
        this.hp -= dmg * 0.18;
        if (this.hp <= 0) { this.die(); return 'hit'; }
        return 'blocked';
      }
      // guard break
      this.staggerT = 0; this.blocking = false;
      this.hp -= dmg * 0.6;
      if (this.hp <= 0) { this.die(); return 'hit'; }
      return 'broken';
    }
    this.hp -= dmg;
    if (this.hp <= 0) { this.die(); return 'hit'; }
    // light stagger on heavy hits
    if (dmg >= 18 && this.staggerT < 0) this.staggerT = 0;
    return 'hit';
  }

  die() {
    this.hp = 0; this.dead = true;
    this.deathT = 0;
    if (this.trail) { this.fx.endTrail(this.trail); this.trail = null; }
    this.atk = null;
    this.audio.deathSting();
  }

  heal(n) { this.hp = Math.min(this.maxHp, this.hp + n); }

  update(dt, input, nowSec) {
    const P = this.parts;
    if (this.dead) {
      this.deathT += dt;
      poseDeath(P, this.deathT);
      this.group.position.y = 0;
      return;
    }
    // stamina regen (not while blocking)
    if (!this.blocking) this.stam = Math.min(this.maxStam, this.stam + 26 * dt);

    // --- stagger ---
    if (this.staggerT >= 0) {
      this.staggerT += dt;
      poseStagger(P, Math.min(1, this.staggerT / 0.55));
      if (this.staggerT > 0.55) { this.staggerT = -1; resetSwordArm(P); }
      this.group.rotation.y = this.yaw;
      return;
    }

    // --- dodge roll ---
    if (this.dodgeT >= 0) {
      this.dodgeT += dt;
      const k = Math.min(1, this.dodgeT / 0.42);
      this.pos.addScaledVector(this.dodgeDir, 7.5 * dt * (1 - k * 0.5));
      P.torso.rotation.x = 0.9 * Math.sin(k * Math.PI);
      P.torso.position.y = 1.0 - 0.35 * Math.sin(k * Math.PI);
      this.world.collide(this.pos, 0.45);
      if (this.dodgeT >= 0.42) { this.dodgeT = -1; P.torso.rotation.x = 0; P.torso.position.y = 1.0; }
      this.group.rotation.y = this.yaw;
      return;
    }

    // --- attack ---
    if (this.atk) {
      const a = this.atk;
      a.t += dt * (this.speedMul || 1);
      const t = Math.min(1, a.t / a.dur);
      // forward lunge during the strike
      if (t > 0.15 && t < 0.7) {
        const fwd = new THREE.Vector3(Math.sin(this.yaw), 0, Math.cos(this.yaw));
        this.pos.addScaledVector(fwd, a.def.lunge * dt);
      }
      poseSwing(P, t, a.def.style);
      // legs brace
      P.legL.rotation.x = 0.3; P.legR.rotation.x = -0.35;
      this.world.collide(this.pos, 0.45);
      if (a.t >= a.dur) {
        if (a.kind === 'light' && a.queued && a.stage < 2) {
          const ns = a.stage + 1;
          this.endAttack();
          this.startAttack('light', ns);
        } else this.endAttack();
      }
      this.group.rotation.y = this.yaw;
      return;
    }

    // --- block pose ---
    if (this.blocking) {
      poseBlock(P);
      this.group.rotation.y = this.yaw;
      return;
    }

    // --- locomotion ---
    const mv = new THREE.Vector3();
    if (input.f) mv.z -= 1; if (input.b) mv.z += 1;
    if (input.l) mv.x -= 1; if (input.r) mv.x += 1;
    const moving = mv.lengthSq() > 0.01;
    if (moving) {
      mv.normalize();
      // camera-relative: camera looks north (-z) by default; keep simple world-aligned
      // since the camera stays behind the player on the z axis.
      const sprint = input.sprint && this.stam > 1;
      const speed = sprint ? 8.2 : 5.0;
      if (sprint) this.stam = Math.max(0, this.stam - 14 * dt);
      const targetYaw = Math.atan2(mv.x, mv.z);
      let dy = targetYaw - this.yaw;
      while (dy > Math.PI) dy -= Math.PI * 2;
      while (dy < -Math.PI) dy += Math.PI * 2;
      this.yaw += dy * Math.min(1, 14 * dt);
      this.pos.addScaledVector(mv, speed * dt);
      this.walkPhase += dt * (sprint ? 13 : 9);
      poseWalk(P, this.walkPhase, sprint ? 1.25 : 1);
      this.stepT += dt * (sprint ? 2.2 : 1.6);
      if (this.stepT > 0.62) { this.stepT = 0; this.audio.footstep(); }
    } else {
      poseIdle(P, nowSec);
    }
    this.world.collide(this.pos, 0.45);
    this.group.rotation.y = this.yaw;
  }

  // world position the blade tip was at — used by main for hit tests
  bladeTip(out) {
    return this.parts.swordTip.getWorldPosition(out);
  }
}
