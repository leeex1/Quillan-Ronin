// main.js — Wayward Ronin bootstrap: renderer, input, camera, game states
// (title / play / dialog / pause / ending), combat resolution, HUD, and
// the story context object that story.js drives.

import * as THREE from 'three';
import { World } from './world.js';
import { Player } from './player.js';
import { Enemy } from './enemies.js';
import { NPC, NPC_DEFS } from './npcs.js';
import { Dialogue } from './dialogue.js';
import { Story, ENDINGS } from './story.js';
import { FX } from './fx.js';
import { AudioEngine } from './audio.js';
import { TouchControls } from './touch.js';
import { saveGame, loadGame, clearSave, hasSave } from './save.js';
import { SWORDS, SwordRack } from './swords.js';

// ---------- renderer ----------
// v2: touch devices get a smaller pixel-ratio cap + shadow map (safe perf win)
const isTouchDevice = ('ontouchstart' in window) || (typeof navigator !== 'undefined' && navigator.maxTouchPoints > 0);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(isTouchDevice ? 1.5 : 2, window.devicePixelRatio));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.getElementById('viewport').appendChild(renderer.domElement);

const world = new World({ shadowSize: isTouchDevice ? 1024 : 2048 });
const scene = world.scene;
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 600);

const audio = new AudioEngine();
const fx = new FX(scene, camera);
const player = new Player(scene, world, fx, audio);
const dialogue = new Dialogue(audio);
// v2: sword collection — the rack owns what the player has found
const swords = new SwordRack();
swords.equip('worn', player);

// ---------- input ----------
const input = {
  f: false, b: false, l: false, r: false, sprint: false,
  blockHeld: false, blockEdge: false,
  lightQueued: false, heavyQueued: false, dodgeQueued: false,
};
const keymap = { KeyW: 'f', KeyS: 'b', KeyA: 'l', KeyD: 'r', ShiftLeft: 'sprint', ShiftRight: 'sprint' };
let gameState = 'title'; // title | play | dialog | pause | ending
let prevState = 'play';

function typing() { return document.activeElement && document.activeElement.tagName === 'INPUT'; }

window.addEventListener('keydown', (e) => {
  if (typing()) return;
  if (e.repeat && (e.code === 'KeyE' || e.code === 'Space')) return; // no auto-advance spam
  if (keymap[e.code] !== undefined) { input[keymap[e.code]] = true; e.preventDefault(); }
  if (gameState !== 'play' && gameState !== 'dialog') {
    if (e.code === 'KeyM') toggleMute();
    return;
  }
  switch (e.code) {
    case 'KeyJ': input.lightQueued = true; break;
    case 'KeyK': input.heavyQueued = true; break;
    case 'KeyF':
      if (!input.blockHeld) input.blockEdge = true;
      input.blockHeld = true; break;
    case 'Space': input.dodgeQueued = true; e.preventDefault(); break;
    case 'KeyE':
      if (gameState === 'dialog') dialogue.advance();
      else tryInteract();
      break;
    case 'KeyP': case 'Escape': togglePause(); break;
    case 'KeyM': toggleMute(); break;
  }
});
window.addEventListener('keyup', (e) => {
  if (keymap[e.code] !== undefined) input[keymap[e.code]] = false;
  if (e.code === 'KeyF') input.blockHeld = false;
});
renderer.domElement.addEventListener('mousedown', (e) => {
  if (gameState !== 'play') return;
  if (e.button === 0) input.lightQueued = true;
  if (e.button === 2) input.heavyQueued = true;
});
window.addEventListener('contextmenu', (e) => e.preventDefault());
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- HUD helpers ----------
const $ = (id) => document.getElementById(id);
function setObjective(t) { $('objective-text').textContent = t; }
function banner(t) {
  const b = $('banner');
  $('banner-text').textContent = t;
  b.classList.remove('hidden');
  b.style.opacity = 1;
  clearTimeout(b._t);
  b._t = setTimeout(() => {
    b.style.transition = 'opacity 1.2s'; b.style.opacity = 0;
    setTimeout(() => b.classList.add('hidden'), 1300);
  }, 2200);
}
let toastTimer = null;
function toast(t) {
  const el = $('toast');
  el.textContent = t;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 2600);
}
function updateRepHud(rep) {
  for (const f of ['clan', 'rebel', 'town']) {
    const el = $('rep-' + f);
    const v = rep[f];
    if (v >= 0) { el.style.left = '50%'; el.style.width = (v / 2) + '%'; }
    else { el.style.left = (50 + v / 2) + '%'; el.style.width = (-v / 2) + '%'; }
  }
}
function damageFlash() {
  const el = $('damage-flash');
  el.style.transition = 'none'; el.style.opacity = 0.9;
  requestAnimationFrame(() => {
    el.style.transition = 'opacity 0.6s'; el.style.opacity = 0;
  });
}

// zeroed input used while dialogue is open (the body stays still)
const NO_INPUT = { f: false, b: false, l: false, r: false, sprint: false };

// ---------- entities ----------
const npcs = NPC_DEFS.map((d) => new NPC(scene, world, d));
const enemies = [];
const actors = []; // temporary scene actors (event 1 tax collectors)
const groups = {}; // tag -> Set of enemies

function spawnEnemies(list, tag) {
  const set = new Set();
  for (const d of list) {
    const e = new Enemy(scene, world, fx, audio, d.type, d.x, d.z,
      { faction: d.faction, name: d.name, hpMul: d.hpMul, aggroRange: d.aggroRange, drops: d.drops });
    enemies.push(e); set.add(e);
  }
  if (tag) groups[tag] = set;
  return set;
}
function spawnActors() {
  const mk = (id, name, x, z, tree) => {
    const n = new NPC(scene, world, {
      id, name, faction: 'clan', robe: 0x5a1f1f, trim: 0xd8c69a, armor: true,
      home: [x, z], wander: 0, tree, plate: '#e08080',
    });
    npcs.push(n); actors.push(n);
    return n;
  };
  mk('sergeant', 'Tax Sergeant', -19, 7, 'sergeant');
  mk('guardactor', 'Clan Guard', -21.5, 4, 'sergeant');
  return actors[0];
}
function hideNpc(id) {
  const n = npcs.find((x) => x.id === id);
  if (n) n.hide();
}
function clearActors() {
  for (const a of actors) a.hide();
  actors.length = 0;
}

// ---------- save system (v2) ----------
function collectSave() {
  const g = {};
  for (const tag of Object.keys(groups)) {
    const list = [];
    for (const e of groups[tag]) {
      if (e.dead) continue;
      list.push({
        type: e.type, x: +e.pos.x.toFixed(2), z: +e.pos.z.toFixed(2),
        faction: e.faction, name: e.name, hpMul: +(e.maxHp / e.def.hp).toFixed(2),
        drops: e.drops || null,
      });
    }
    if (list.length) g[tag] = list;
  }
  return {
    rep: { ...story.rep },
    flags: { ...story.flags },
    kills: { ...story.kills },
    player: {
      hp: Math.round(player.hp), maxHp: player.maxHp,
      x: +player.pos.x.toFixed(2), z: +player.pos.z.toFixed(2), yaw: +player.yaw.toFixed(3),
    },
    swords: { owned: [...swords.owned], equipped: swords.equipped },
    night: world.nightK >= 1,
    objective: document.getElementById('objective-text').textContent,
    groups: g,
  };
}
function doSave(silent) {
  const ok = saveGame(collectSave());
  if (!silent) {
    if (ok) { toast('Progress saved.'); audio.uiClick(); }
    else toast('Save failed — storage unavailable.');
  }
}
function applySave(s) {
  story.rep = { clan: s.rep.clan || 0, rebel: s.rep.rebel || 0, town: s.rep.town || 0 };
  story.flags = { ...(s.flags || {}) };
  story.kills = { clan: (s.kills && s.kills.clan) || 0, rebel: (s.kills && s.kills.rebel) || 0 };
  player.maxHp = s.player.maxHp || 100;
  player.hp = Math.min(player.maxHp, s.player.hp);
  player.stam = player.maxStam;
  player.dead = false;
  player.pos.set(s.player.x, 0, s.player.z);
  player.yaw = s.player.yaw == null ? Math.PI : s.player.yaw;
  player.group.rotation.y = player.yaw;
  swords.owned = [...(s.swords ? s.swords.owned : ['worn'])].filter((id) => SWORDS[id]);
  if (!swords.owned.includes('worn')) swords.owned.unshift('worn');
  swords.equip(s.swords && SWORDS[s.swords.equipped] ? s.swords.equipped : 'worn', player);
  if (s.night) world.setNight(true);
  // re-hide NPCs removed by story flags
  if (story.flags.event2done === 'report') hideNpc('tsubaki');
  if (story.flags.event3done === 'clan') hideNpc('ren');
  if (story.flags.warned) { hideNpc('ren'); hideNpc('tsubaki'); }
  // re-spawn fights that were live when the game was saved
  for (const tag of Object.keys(s.groups || {})) {
    if (s.groups[tag].length) spawnEnemies(s.groups[tag], tag);
  }
  setObjective(s.objective || 'Find your path.');
  updateRepHud(story.rep);
  toast('Journey restored.');
}

// ---------- story context ----------
const storyCtx = {
  audio, fx, dialogue, npcs,
  spawnEnemies, setObjective, banner, toast, heal: (n) => player.heal(n),
  hideNpc, clearActors, updateRepHud,
  save: () => doSave(true), // v2: auto-save hook for the story
  setNight: () => { world.setNight(); banner('NIGHT FALLS'); }, // v2
  endGame: (key, stats) => endGame(key, stats),
};
const story = new Story(storyCtx);

// ---------- interaction ----------
function nearestNpc() {
  let best = null, bd = 3.4;
  for (const n of [...npcs, ...actors]) {
    if (n.hidden || n.talking) continue;
    const dx = n.pos.x - player.pos.x, dz = n.pos.z - player.pos.z;
    const d = Math.sqrt(dx * dx + dz * dz);
    if (d < bd) { bd = d; best = n; }
  }
  return best;
}
let talkTarget = null;
function tryInteract() {
  const n = nearestNpc();
  if (!n) return;
  talkTarget = n;
  n.talking = true;
  n.faceToward(player.pos);
  gameState = 'dialog';
  audio.uiClick();
  dialogue.start(story.treeFor(n), storyCtx, () => {
    n.talking = false;
    talkTarget = null;
    if (gameState === 'dialog') gameState = 'play';
  });
}

// ---------- touch controls (v2) ----------
// Same `input` pipeline as keyboard/mouse. Auto-shows only on touch devices;
// dialogue choices are native HTML buttons, already tappable.
const touchUi = document.getElementById('touch-ui');
const touchCtl = new TouchControls(input, {
  interact: () => { if (gameState === 'dialog') dialogue.advance(); else tryInteract(); },
  dodge: () => {
    if (gameState !== 'play') return;
    _v1.set(0, 0, 0);
    if (input.f) _v1.z -= 1; if (input.b) _v1.z += 1;
    if (input.l) _v1.x -= 1; if (input.r) _v1.x += 1;
    if (_v1.lengthSq() > 0.01) _v1.normalize();
    player.tryDodge(_v1);
  },
  pause: () => togglePause(),
});

// ---------- UI wiring ----------
function toggleMute() {
  audio.setMuted(!audio.muted);
  $('btn-mute').textContent = 'Sound: ' + (audio.muted ? 'Off' : 'On');
}
function togglePause() {
  if (gameState === 'play') {
    prevState = 'play'; gameState = 'pause';
    // v2: always open on the main pause menu, never stranded on the sword panel
    $('sword-panel').classList.add('hidden');
    $('pause-menu').querySelector('.title-menu').classList.remove('hidden');
    $('pause-menu').classList.remove('hidden');
  } else if (gameState === 'pause') {
    gameState = prevState;
    $('pause-menu').classList.add('hidden');
  }
}
// v2: sword collection UI
function renderSwords() {
  const list = $('sword-list');
  list.innerHTML = '';
  for (const id of swords.owned) {
    const s = SWORDS[id];
    const row = document.createElement('div');
    row.className = 'sword-row' + (swords.equipped === id ? ' equipped' : '');
    const name = document.createElement('div');
    name.className = 'sword-name';
    name.textContent = s.name + (swords.equipped === id ? ' — EQUIPPED' : '');
    const stats = document.createElement('div');
    stats.className = 'sword-stats';
    stats.textContent = `Damage ×${s.dmgMul} · Speed ×${s.speedMul} · Parry +${Math.round(s.parryBonus * 1000)}ms`;
    const desc = document.createElement('div');
    desc.className = 'sword-desc';
    desc.textContent = s.desc;
    row.appendChild(name); row.appendChild(stats); row.appendChild(desc);
    if (swords.equipped !== id) {
      const b = document.createElement('button');
      b.className = 'menu-btn ghost small';
      b.textContent = 'Equip';
      b.addEventListener('click', () => {
        swords.equip(id, player);
        audio.uiClick();
        toast(`${s.name} equipped.`);
        renderSwords();
        doSave(true);
      });
      row.appendChild(b);
    }
    list.appendChild(row);
  }
}
$('btn-save').addEventListener('click', () => doSave(false));
$('btn-swords').addEventListener('click', () => {
  audio.uiClick();
  $('pause-menu').querySelector('.title-menu').classList.add('hidden');
  $('sword-panel').classList.remove('hidden');
  renderSwords();
});
$('btn-swords-close').addEventListener('click', () => {
  audio.uiClick();
  $('sword-panel').classList.add('hidden');
  $('pause-menu').querySelector('.title-menu').classList.remove('hidden');
});
if (hasSave()) $('btn-continue').classList.remove('hidden');
$('btn-continue').addEventListener('click', () => {
  const s = loadGame();
  audio.init(); audio.uiClick();
  $('title-screen').classList.add('hidden');
  $('hud').classList.remove('hidden');
  gameState = 'play';
  if (s) applySave(s);
  else setObjective('Speak with Elder Jiro by the well');
  updateRepHud(story.rep);
  banner(s && s.night ? 'KIRISATO — NIGHT' : 'KIRISATO — DUSK');
});
$('btn-start').addEventListener('click', () => {
  clearSave(); // new journey wipes any old save
  audio.init(); audio.uiClick();
  $('title-screen').classList.add('hidden');
  $('hud').classList.remove('hidden');
  gameState = 'play';
  setObjective('Speak with Elder Jiro by the well');
  updateRepHud(story.rep);
  banner('KIRISATO — DUSK');
});
$('btn-again').addEventListener('click', () => { clearSave(); location.reload(); });
$('btn-restart').addEventListener('click', () => { clearSave(); location.reload(); });
$('btn-resume').addEventListener('click', togglePause);
$('btn-mute').addEventListener('click', () => { audio.uiClick(); toggleMute(); });
let helpReturn = 'title';
$('btn-help-title').addEventListener('click', () => { helpReturn = 'title'; $('help-overlay').classList.remove('hidden'); });
$('btn-help-pause').addEventListener('click', () => { helpReturn = 'pause'; $('help-overlay').classList.remove('hidden'); });
$('btn-help-close').addEventListener('click', () => $('help-overlay').classList.add('hidden'));

function endGame(key, stats) {
  if (gameState === 'ending') return;
  gameState = 'ending';
  const e = ENDINGS[key];
  $('ending-title').textContent = e.title;
  $('ending-epilogue').textContent = e.epilogue;
  const r = stats.rep, k = stats.kills;
  $('ending-stats').innerHTML =
    `Clan ${r.clan} &nbsp;·&nbsp; Rebels ${r.rebel} &nbsp;·&nbsp; Town ${r.town} &nbsp;·&nbsp; Foes cut down: ${k.clan + k.rebel}`;
  $('ending-screen').classList.remove('hidden');
  $('hud').classList.add('hidden');
  audio.endingChord();
}

// ---------- combat resolution ----------
const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();

function resolvePlayerAttack() {
  const a = player.atk;
  if (!a || a.didHit) return;
  const t = a.t / a.dur;
  if (t < a.def.hitAt[0] || t > a.def.hitAt[1]) return;
  a.didHit = true;
  const facing = _v1.set(Math.sin(player.yaw), 0, Math.cos(player.yaw));
  let hitAny = false, combo = 0;
  for (const e of enemies) {
    if (e.dead) continue;
    _v2.subVectors(e.pos, player.pos); _v2.y = 0;
    const dist = _v2.length();
    if (dist > 2.9) continue;
    _v2.normalize();
    if (_v2.dot(facing) < 0.1 && dist > 1.2) continue; // frontal arc
    hitAny = true; combo++;
    const isHeavy = a.kind === 'heavy';
    const dealt = e.takeDamage(a.def.dmg * (player.dmgMul || 1), player.pos, isHeavy, performance.now() / 1000);
    const wasBlocked = dealt < a.def.dmg * 0.5; // ashigaru active block = chip damage
    _v3.copy(e.pos); _v3.y = 1.4;
    if (wasBlocked) {
      fx.floatText(_v3, 'BLOCKED', 'block');
      audio.clash(false);
      fx.hitStop(40);
    } else {
      fx.burst(_v3, { count: isHeavy ? 22 : 13, color: 0xffd27a, speed: isHeavy ? 8 : 6, life: 0.5, size: 0.09 });
      fx.burst(_v3, { count: 8, color: 0xa03020, speed: 4, life: 0.6, size: 0.08 });
      fx.floatText(_v3, Math.round(dealt), isHeavy ? 'crit' : '');
      audio.hitFlesh(isHeavy);
      fx.hitStop(a.def.stop);
      fx.addShake(a.def.shake);
      if (e.dead) {
        story.kills[e.faction === 'clan' ? 'clan' : 'rebel']++;
        if (e.drops && swords.add(e.drops)) {
          // v2: elite sword drop
          const sw = SWORDS[e.drops];
          toast(`Claimed ${sw.name}!`);
          banner('SWORD CLAIMED');
          audio.repUp();
          doSave(true);
        } else {
          toast(`${e.name} cut down.`);
          audio.repDown();
        }
      }
    }
  }
  if (hitAny) {
    const cc = $('combo-counter');
    const n = parseInt($('combo-n').textContent || '0', 10) + combo;
    $('combo-n').textContent = n;
    cc.classList.remove('hidden');
    clearTimeout(cc._t);
    cc._t = setTimeout(() => { cc.classList.add('hidden'); $('combo-n').textContent = '0'; }, 2200);
  }
}

function resolveEnemyAttacks(nowSec) {
  for (const e of enemies) {
    if (e.dead) continue;
    if (!e.strikeLanded(player)) continue;
    if (player.invulnerable()) {
      _v3.copy(player.pos); _v3.y = 1.5;
      fx.floatText(_v3, 'MISS', 'block');
      continue;
    }
    const res = player.takeDamage(e.def.dmg, e.pos, nowSec);
    _v3.copy(player.pos); _v3.y = 1.5;
    if (res === 'parried') {
      fx.floatText(_v3, 'PARRY!', 'parry');
      audio.clash(true);
      fx.burst(_v3, { count: 20, color: 0x9fd8ff, speed: 7, life: 0.5, size: 0.08 });
      fx.hitStop(150);
      fx.addShake(0.35);
      e.applyParry();
      toast('Parried! They\u2019re wide open — strike!');
    } else if (res === 'blocked') {
      fx.floatText(_v3, 'blocked', 'block');
      audio.clash(false);
      fx.hitStop(45);
      fx.addShake(0.2);
    } else if (res === 'broken') {
      fx.floatText(_v3, 'GUARD BREAK', 'crit');
      audio.clash(false);
      fx.hitStop(90);
      fx.addShake(0.5);
      damageFlash();
    } else if (res === 'hit') {
      audio.hitFlesh(false);
      damageFlash();
      fx.hitStop(70);
      fx.addShake(0.4);
      fx.burst(_v3, { count: 10, color: 0xa03020, speed: 4, life: 0.5, size: 0.09 });
    }
  }
}

// ---------- event triggers ----------
function checkTriggers() {
  const p = player.pos, f = story.flags;
  // event 1: market trouble
  if (f.introDone && !f.event1done && !f.event1scene) {
    const dx = p.x - (-20), dz = p.z - 5;
    if (dx * dx + dz * dz < 11 * 11) {
      f.event1scene = true;
      const sgt = spawnActors();
      banner('TROUBLE AT THE MARKET');
      toast('Tax collectors are shaking down Mei...');
      // auto-start the confrontation dialogue
      sgt.talking = true; sgt.faceToward(p);
      talkTarget = sgt;
      gameState = 'dialog';
      dialogue.start(story.trees.sergeant, storyCtx, () => {
        sgt.talking = false; talkTarget = null;
        if (gameState === 'dialog') gameState = 'play';
      });
    }
  }
  story.maybeTriggerStorehouseAmbush(p, storyCtx);
  story.maybeTriggerExile(p, storyCtx);
}

function checkGroups() {
  for (const tag of Object.keys(groups)) {
    const set = groups[tag];
    let allDead = true;
    for (const e of set) if (!e.dead) { allDead = false; break; }
    if (allDead) {
      delete groups[tag];
      story.onEnemiesCleared(tag, storyCtx);
    }
  }
}

// ---------- camera ----------
const camTarget = new THREE.Vector3(0, 0, 38);
function updateCamera(dt) {
  _v1.set(player.pos.x + 3.5, 4.6, player.pos.z + 9.5);
  camera.position.lerp(_v1, Math.min(1, 5 * dt));
  _v2.set(player.pos.x, 1.7, player.pos.z - 2);
  camTarget.lerp(_v2, Math.min(1, 7 * dt));
  camera.lookAt(camTarget);
  fx.applyShake(dt);
}

// ---------- main loop ----------
const clock = new THREE.Clock();
let elapsed = 0;
let deathTimer = -1;

function frame() {
  requestAnimationFrame(frame);
  let dt = Math.min(0.05, clock.getDelta());
  elapsed += dt;
  const nowSec = performance.now() / 1000;

  if (fx.inHitStop()) dt = 0; // impact freeze frames

  if (gameState === 'play' || gameState === 'dialog') {
    // aim assist: nearest living enemy within 9
    let best = null, bd = 9;
    for (const e of enemies) {
      if (e.dead) continue;
      const d = e.pos.distanceTo(player.pos);
      if (d < bd) { bd = d; best = e.pos; }
    }
    player.aimTarget = best;

    if (gameState === 'play') {
      // consume edge-triggered inputs
      if (input.lightQueued) { player.tryLight(); input.lightQueued = false; }
      if (input.heavyQueued) { player.tryHeavy(); input.heavyQueued = false; }
      if (input.dodgeQueued) {
        _v1.set(0, 0, 0);
        if (input.f) _v1.z -= 1; if (input.b) _v1.z += 1;
        if (input.l) _v1.x -= 1; if (input.r) _v1.x += 1;
        if (_v1.lengthSq() > 0.01) _v1.normalize();
        player.tryDodge(_v1);
        input.dodgeQueued = false;
      }
      player.setBlock(input.blockHeld, input.blockEdge);
      input.blockEdge = false;
      checkTriggers();
    } else {
      // in dialogue: freeze the body — no movement, attacks, or triggers
      input.lightQueued = input.heavyQueued = input.dodgeQueued = false;
      input.blockEdge = false;
      player.setBlock(false, false);
    }

    player.update(dt, gameState === 'play' ? input : NO_INPUT, nowSec);
    if (gameState === 'play') {
      resolvePlayerAttack();
      resolveEnemyAttacks(nowSec);
      checkGroups();
    }

    for (const e of enemies) e.update(dt, player, nowSec, camera);
    // remove fully-expired corpses
    for (let i = enemies.length - 1; i >= 0; i--) {
      const e = enemies[i];
      if (e.removeAt > 0 && nowSec > e.removeAt) {
        scene.remove(e.group);
        enemies.splice(i, 1);
        for (const tag of Object.keys(groups)) groups[tag].delete(e);
      }
    }
    for (const n of npcs) n.update(dt, nowSec, camera);
    world.update(elapsed, dt);

    // interact prompt
    if (gameState === 'play') {
      const n = nearestNpc();
      const ip = $('interact-prompt');
      if (n) {
        ip.classList.remove('hidden');
        $('interact-name').textContent = 'Talk to ' + n.def.name;
      } else ip.classList.add('hidden');
    } else $('interact-prompt').classList.add('hidden');

    // v2: touch UI visible only on touch devices, only during play/dialog
    if (touchCtl.active) touchUi.classList.toggle('hidden', gameState !== 'play' && gameState !== 'dialog');

    // HUD bars
    $('health-fill').style.width = (player.hp / player.maxHp * 100) + '%';
    $('stamina-fill').style.width = (player.stam / player.maxStam * 100) + '%';

    // death → death ending after a beat
    if (player.dead && deathTimer < 0) deathTimer = 0;
    if (deathTimer >= 0) {
      deathTimer += dt;
      if (deathTimer > 2.4) { deathTimer = -1; story.onDeath(storyCtx); }
    }
  } else {
    // menus: the town stays alive behind the title / pause / ending screens
    for (const n of npcs) n.update(dt, nowSec, camera);
    world.update(elapsed, dt);
  }

  fx.update(dt);
  updateCamera(dt === 0 ? 0.0001 : dt);
  renderer.render(scene, camera);
}

frame();
