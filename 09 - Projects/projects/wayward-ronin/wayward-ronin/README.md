# Wayward Ronin

An original browser-based 3D samurai action game — an open-ended dusk town,
real-time sword combat, faction choices, branching dialogue, and **6 endings**.
Every story beat, character name, line of dialogue, and model is original;
only the *genre formula* (wandering swordsman, town factions, choice-driven
endings) is shared with the classic samurai-action games that inspired its
design. No borrowed characters, story text, or assets.

---

## AGENT QUICKSTART — "bam game" in 5 steps

**Prerequisites:** Node.js (for syntax checks), Python 3 (for the static
server), internet access on first load (three.js CDN).

```bash
# 1. Unzip / enter the project
cd wayward-ronin

# 2. Syntax-check all 13 JS modules (must all pass, zero output = good)
node --check js/main.js && node --check js/world.js && node --check js/characters.js \
 && node --check js/player.js && node --check js/enemies.js && node --check js/npcs.js \
 && node --check js/dialogue.js && node --check js/story.js && node --check js/fx.js \
 && node --check js/audio.js && node --check js/touch.js && node --check js/save.js \
 && node --check js/swords.js

# 3. Serve over HTTP (ES modules refuse file:// — this step is mandatory)
python3 -m http.server 8080

# 4. Open in a desktop browser (keyboard + mouse required)
#    http://localhost:8080/

# 5. Smoke-test the golden path:
#    Title → "Begin the Road" → talk to Elder Jiro (E) → walk west to the
#    market → tax-collector confrontation fires → pick a choice →
#    combat works (LMB attack, F block, Space dodge) → HUD updates.
```

**Done = all of:** page loads with no console errors, title screen renders,
player moves with WASD, dialogue opens/closes with E, one full event
completes, an ending screen is reachable. If any of that fails, the build is
not done.

---

## Controls

| Key | Action |
|---|---|
| WASD | Move (camera-relative, fixed follow camera) |
| SHIFT | Sprint (drains stamina) |
| LMB / J | Light attack — 3-hit combo, queueable |
| RMB / K | Heavy attack — guard-breaking, slower |
| F (hold) | Block (frontal, drains stamina on hits) |
| F (tap, ≤0.24 s before a hit lands) | **Parry** — no damage, attacker staggered 1.25 s |
| SPACE | Dodge roll (invulnerability frames) |
| E | Talk / advance dialogue |
| 1–4 / click | Dialogue choices |
| P or ESC | Pause |
| M | Mute |

Combat notes: attacks auto-aim to the nearest enemy in a forward arc.
Ashigaru block frontal attacks — flank them or break guard with heavies
(which drain enemy stamina; emptied stamina = guard break). Red blade glow
telegraphs enemy strikes.

### Touch controls (mobile/tablet)

On touch devices a virtual joystick and buttons appear automatically —
desktop is unaffected.

| Control | Action |
|---|---|
| Left stick | Move (push past ~80% to the rim = sprint) |
| ATK / HVY | Light / heavy attack |
| BLK (hold) | Block; tap to parry |
| ROLL | Dodge roll |
| TALK | Talk / advance dialogue |
| ⏸ | Pause |

Dialogue choices are tappable buttons (also works with 1–4 on keyboard).

### Save system

Progress auto-saves to `localStorage` (`wayward-ronin-save-v1`, versioned):
reputation, flags, kills, HP/max HP, position, owned + equipped swords,
night state, objective, and any live enemy groups (re-spawned on load).
Auto-save fires after each story choice and fight, on nightfall, and on
sword pickups. The title screen offers **Continue Journey** when a save
exists; **Save** is also in the pause menu. Save is cleared on new game /
restart / after an ending. Not saved: nothing else — assume a fresh load is
a snapshot of the moment the last save fired.

### Swords

Five blades (`js/swords.js`), each with damage/speed/parry/blade-color/trail
stats. You start with the **Worn Blade**; the other four drop from elite
enemies — Tsubaki (Crane Feather), Ren (Ember of the Forge), Captain Isamu
(Pale Moon Crossing), and the dock hatamoto guarding Kage's shipment
(Willow in Rain). Equip them from the pause menu (**Swords** button); the
equipped blade visibly changes color on your character's back and in trails.

---

## Architecture

```
index.html          title screen, HUD, dialogue panel, pause, ending screen,
                    help overlay. All UI element IDs listed below.
css/style.css       ink-brush UI theme (vermilion/gold, brush-rule dividers)
js/main.js          bootstrap: renderer, input, camera, game states, combat
                    resolution, HUD helpers, event proximity triggers, storyCtx
js/world.js         dusk town: sky dome, 21 flickering lanterns, houses, torii,
                    banners, river + docks district, collision. API: new World({ shadowSize }) → {
                    scene, update(elapsed, dt), collide(pos, radius),
                    setNight(instant), nightK }
                    (night lerps sky/sun/moon/fog/lanterns over ~10 s;
                    touch devices get a smaller shadow map)
js/characters.js    procedural samurai rig + pose library
                    (idle/walk/swings/block/death). API: createSamurai(opts),
                    poseIdle/poseWalk/…(parts, t)
js/player.js        movement, 3-hit queueable combo, heavy, hold-block /
                    tap-parry, dodge i-frames, stamina, HP.
                    API: player.tryLight/tryHeavy/tryDodge(dir),
                    player.setBlock(held, edge), player.takeDamage(dmg, fromPos, now),
                    player.heal(n), player.update(dt, input, now)
js/enemies.js       3 AI archetypes: ashigaru (defensive blocker), rusher
                    (aggressive), hatamoto (elite, more HP/patterns).
                    API: new Enemy(scene, world, fx, audio, type, x, z,
                    { faction, name, hpMul, aggroRange, drops }) — `drops` is a
                    sword id (see swords.js) granted on kill
js/npcs.js          11 NPCs, wander/idle state machines, canvas nameplates.
                    API: NPC_DEFS array, new NPC(scene, world, def),
                    npc.hide()/show()/faceToward(p)
js/dialogue.js      typewriter dialogue engine, conditional choices.
                    API: dialogue.start(tree, ctx, onEnd), dialogue.advance(),
                    dialogue.choose(i) (programmatic choice — used by touch)
js/story.js         ALL story data: trees, events, reputation, endings.
                    This is the file you edit to change the story.
js/fx.js            hit-stop, screen shake, sword trails, particles,
                    damage numbers. API: fx.hitStop(ms), fx.addShake(n),
                    fx.burst(pos, opts), fx.floatText(pos, str, cls)
js/audio.js         fully synthesized WebAudio: clangs, wind, koto plucks.
                    API: audio.init(), audio.setMuted(b), audio.hitFlesh(),
                    audio.clash(parried), audio.repUp/repDown(),
                    audio.dialogueBlip(), audio.uiClick(), audio.endingChord()
js/touch.js         virtual joystick + buttons, active only on touch devices
                    (desktop untouched). API: new TouchControls(input,
                    { interact, dodge, pause }) → { active, destroy }
js/save.js          localStorage persistence. API: saveGame(data),
                    loadGame() (null on corrupt/version mismatch),
                    hasSave(), clearSave(), SAVE_VERSION
js/swords.js        5 blades with stat/visual profiles + the player's
                    collection. API: SWORDS, BASE_PARRY_WINDOW,
                    new SwordRack() → { owned, equipped, add(id), equip(id, player) }
                    — equip() applies dmgMul/speedMul/parryWindow/trailColor
                    and re-tints the blade mesh
```

### The story context (`storyCtx`)

`main.js` builds one context object and hands it to both `Story` and every
dialogue tree. Anything story code needs from the engine goes through this —
**never reach into globals from story.js:**

```js
{
  audio, fx, dialogue, npcs,          // subsystems
  spawnEnemies,                        // (list, tag) → Set — see enemies.js API
  setObjective, banner, toast,         // HUD text helpers
  heal: (n) => player.heal(n),
  hideNpc,                             // (id) — remove an NPC from the world
  clearActors,                          // remove temporary event NPCs
  updateRepHud,                        // refresh the 3 reputation bars
  save: () => doSave(true),            // auto-save hook (fired by story events)
  setNight: () => { world.setNight(); banner('NIGHT FALLS'); },
  endGame: (key, stats) => endGame(key, stats),
}
```

`spawnEnemies(list, tag)` groups enemies under `tag`; when every enemy in the
group is dead, `main.js` calls `story.onEnemiesCleared(tag, ctx)` — that is
how fights advance the plot.

### Game states

`gameState` in `main.js`: `title | play | dialog | pause | ending`.
Dialogue freezes the player body (zeroed input, no triggers). Restart is
`location.reload()`. The town keeps animating behind title/pause/ending.

---

## Story system (edit `js/story.js` — nothing else needed for content)

### Reputation

`story.rep = { clan, rebel, town }`, each clamped to −100…+100.
`story.adjust(faction, amt)` changes a meter, shows a toast, plays a sound,
updates the HUD. Thresholds that matter:

- Clan ending: `rep.clan ≥ 50`
- Rebel ending: `rep.rebel ≥ 50`
- Town ending: `rep.town ≥ 35`
- Butcher ending: `kills.clan + kills.rebel ≥ 7` **and** both clan/rebel `< 50`
- Death ending overrides everything (fired by `story.onDeath`)

### Ending priority (in `evaluateEnding`)

Butcher → Clan → Rebel → Town → Road (walk away). Death fires separately.

### Flags used (all in `story.flags`)

`introDone` · `boughtFood` · `reported` · `event1done` · `event1scene` ·
`event2done` (`'accept'`/`'refuse'`/`'report'`) · `storehouseAmbush` ·
`event3done` (`'clan'`/`'refuse'`/`'warn'`) · `warned` ·
`night` (set once, after event 2) ·
side stories: `sideHana` (`'search'`/`'done'`) · `foundCrate` ·
`sideSouta` (`'paid'`/`'fight'`) · `sideKage` (`'blind'`/`'report'`) ·
`sideKano` (`'order'`/`'defy'`).
Kills tracked in `story.kills = { clan, rebel }` (incremented in
`main.js` `resolvePlayerAttack` by the victim's faction).

### Dialogue tree format

```js
myTree: {
  speaker: 'Display Name', plate: '#e8c96a', start: 's',
  nodes: {
    s: {
      text: 'Literal string, or (c) => string for dynamic text',
      onEnter: (c) => { /* runs once when node opens */ },   // optional
      to: 'nextNode',        // linear chain (no choices) — optional
      choices: [
        { text: 'Player line',
          to: 'nodeId',      // null = end dialogue
          if: (c) => true,   // optional visibility condition
          do: (c) => { /* side effects: S.adjust('town', 5), flags, spawns */ } },
      ],
    },
  },
}
```

`treeFor(npc)` in `story.js` routes which tree an NPC uses — Jiro switches
between intro/idle by `introDone`; Ren switches to the warning tree when
`event3done === 'warn' && !warned`; the four docks NPCs (Souta, Hana, Kage,
Kano) each have a main and a `_done` tree.

---

## How to extend (recipes)

### 1. Add an NPC
1. Append a def to `NPC_DEFS` in `js/npcs.js`:
   `{ id:'takeshi', name:'Takeshi', faction:'town', robe:0x3a4a5a, trim:0xd4a24e, home:[x, z], wander:5, tree:'takeshi', plate:'#e8c96a' }`
   (optional: `hat:true`, `armor:true`, `headband:0x…`, `scale:0.8`).
2. Add a `takeshi` tree in `story.js` `buildTrees()`.

### 2. Add a dialogue choice that moves reputation
```js
{ text: `I'll help you.`, to: 'thanks',
  do: (c) => { S.adjust('town', 10); S.flags.helpedTakeshi = true; } },
```

### 3. Add a story event (choice → fight → consequence)
1. Add a dialogue choice whose `do` sets a flag and calls
   `c.spawnEnemies([{ type:'ashigaru', x, z, faction:'clan' }], 'myevent')`.
2. Handle the aftermath in `story.onEnemiesCleared(tag, c)`:
   `else if (tag === 'myevent') { c.setObjective('…'); S.adjust('rebel', 10); }`
3. Or trigger by proximity: add a check in `main.js` `checkTriggers()` like
   the market/storehouse/exile triggers, then call a `story` method.

### 4. Add an ending
1. Add an entry to `ENDINGS` in `story.js` (`title` + `epilogue`).
2. Add a branch in `evaluateEnding()` **and** document its priority.
3. Fire it via `c.endGame('mykey', { rep:{...}, kills:{...} })`.

### 5. Add an enemy type
Add a new `type` case in `js/enemies.js` (stats + AI behavior), then spawn
with `{ type:'mytype', x, z, faction:'clan' }`. Existing archetypes:
`ashigaru` (blocks frontal, flank or guard-break), `rusher` (fast,
aggressive), `hatamoto` (elite, high HP via `hpMul`).

### 6. Temporary event NPCs (like the Tax Sergeant)
See `spawnActors()` in `main.js`: construct `new NPC(scene, world, def)`
directly, push to `npcs` + `actors`, remove with `clearActors()`.

---

## HUD element ID reference (for UI work)

`viewport` · `damage-flash` · `vignette` · `title-screen` (`btn-start`,
`btn-continue`, `btn-help-title`) · `hud` (`objective-text`, `health-fill`,
`stamina-fill`, `rep-clan`, `rep-rebel`, `rep-town`, `combo-counter`,
`combo-n`, `interact-prompt`, `interact-name`, `banner`, `banner-text`,
`toast`) · `dialogue` (`dlg-name`, `dlg-text`, `dlg-choices`, `dlg-continue`) ·
`pause-menu` (`btn-resume`, `btn-save`, `btn-swords`, `btn-restart`,
`btn-mute`, `btn-help-pause`, `sword-panel`, `sword-list`,
`btn-swords-close`) · `ending-screen` (`ending-title`, `ending-epilogue`,
`ending-stats`, `btn-again`) · `help-overlay` (`btn-help-close`) ·
`touch-ui` (`joy-zone`, `joy-knob`, `tb-atk`, `tb-hvy`, `tb-blk`,
`tb-roll`, `tb-talk`, `tb-pause`; static markup in `index.html`, wired by
`js/touch.js`, shown only on touch devices).

## Town map coordinates (x, z)

- Well / Jiro: (5, 13) · Mei's stall / market: (−20, 9) · Manor / Gendo: (0, −27)
- West alley / Ren: (−27, −11) · Storehouse: (30, −6) · North road exit: z < −44
- Docks (east): deck along x = 36–50, z = −18…6 · Souta: (45, −3) · Hana: (40, 4) ·
  Kage: (42.5, −9) · Kano: (37, −16) · lost crate turn-in: east pilings (48, 0)
- Event triggers: market r=11 at (−20, 5) · storehouse r=16 at (30, −6) ·
  exile gate z<−44, |x|<14. Custom collision is circle-vs-AABB in `world.js`.

---

## Story bible

You are a masterless swordsman who walks into the river-town of **Kirisato**
at dusk. The **Kurogane Clan** (Magistrate Gendo, Captain Isamu) holds the
town with tax collectors and patrols. The **Ashen Blades** (Ren, Tsubaki)
plot in the shadows. The **townsfolk** (Elder Jiro, Mei the merchant, the
child Kiku) just want to survive the night.

Three branching events drive the plot:

1. **The tax collector** — a clan sergeant shakes down Mei's stall.
   Intervene, side with the guards, or walk away.
2. **The storehouse** — Ren asks you to burn the clan's rice storehouse.
   Accept, refuse, or report the plot to Captain Isamu (each triggers a
   different ambush). After the dust settles, **night falls** over Kirisato.
3. **The magistrate's order** — Gendo demands a rebel hunt. Accept, refuse,
   or warn Ren first.

Then walk the north road — or die trying — and the town reckons your name.

### The docks

East of the storehouse, the river docks hold four NPCs and three side
stories (optional — they move reputation but never the main plot):

- **Fisherman Souta** — demands a clan harbor tax he can't pay. Pay it for
  him, walk away, or cut down the dock enforcer who comes collecting.
- **Hana the dockworker** — her supply crate went overboard; Souta knows
  where it drifted. Return it for a reward.
- **Kage the smuggler** — offers hush money for a blind eye, or gets
  reported (his bodyguard carries the **Willow in Rain** blade).
- **Inspector Kano** — a clan tax inspector; back his order or defy him to
  his face.

The river is animated; its waters are blocked by collision, so the player
stays on the deck.

| Ending | How it's earned |
|---|---|
| **The Magistrate's Blade** | Clan reputation ≥ 50 |
| **Ashen Dawn** | Rebel reputation ≥ 50 |
| **The Lantern Keeper** | Town reputation ≥ 35 |
| **The Butcher of Kirisato** | 7+ kills while belonging to neither side |
| **The Road Goes On** | Refuse the hunt and walk the north road |
| **Cut Down** | Fall in battle |

---

## Verification status (2026-10-01, v2 iteration)

- `node --check` passes on all 13 JS modules.
- Headless runtime test (Node + three.js 0.160.0, stubbed DOM),
  `1493 passed, 0 failed`:
  - all dialogue trees walked across 15 flag presets (549 nodes / 720
    choices — every target resolves, every handler runs clean),
  - 5 representative playthroughs land on the expected endings
    (clan / rebel / town / wolf / butcher) + death ending,
  - sword drops present on all 4 elites (Tsubaki, Ren, Isamu, docks
    bodyguard); SwordRack stats/blade-color/trail changes apply to the
    player; unknown ids and unowned equips rejected,
  - save/load round-trip (rep, flags, kills, player, swords, night,
    objective, live enemy groups); corrupt/wrong-version saves rejected,
  - touch joystick (direction + rim-sprint + release) and all 6 buttons
    drive the shared input object; desktop stays inactive,
  - night transition lerps to full night (~15 simulated s: sun dims, moon
    fades in, lanterns brighten); fires exactly once after event 2;
    instant-night path for save loads,
  - all 11 NPCs construct and wander; docks tree routing correct
    (incl. the Souta-crate-lead exception during Hana's search),
  - `dialogue.choose()` resolves choices programmatically.
- **Not yet verified in a real GPU browser.** Visual quality, frame rate,
  and audio mixing need a live check before this is called shippable.

## Design notes

- **No third-party assets.** Every model is built from three.js primitives at
  runtime; every sound is synthesized in WebAudio; banner emblems and
  nameplates are drawn on canvas. (A `Soldier.glb` placeholder from an earlier
  engine experiment was deliberately **not** used — a modern soldier model
  would break the art direction; the stylized procedural samurai fit better
  and keep everything original.)
- **Combat feel first:** hit-stop freeze frames, screen shake, sword-trail
  ribbons, spark bursts, floating damage numbers, and a full 3-hit combo with
  input queueing.
- **Fixed-angle follow camera** keeps the action readable without mouse-look
  disorientation.
- No physics library — lightweight circle-vs-AABB collision tuned for the
  town layout.

## Iteration roadmap

- [ ] Live GPU browser verification (visuals, framerate, audio mix)
- [x] Save system (localStorage: rep, flags, checkpoint, swords, night)
- [x] Larger town / second district, more NPCs and side stories (docks)
- [x] Sword collection & upgrading (the genre's signature loot loop)
- [x] Day/night cycle (falls after event 2; NPC schedules still future)
- [x] Touch controls for mobile
- [x] Performance pass (pixel-ratio + shadow-map caps on touch devices)
