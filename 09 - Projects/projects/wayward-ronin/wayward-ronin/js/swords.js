// swords.js — collectible blades. Elite / named enemies drop swords on defeat;
// the pause menu lets the player equip one. Each sword modifies damage,
// attack speed, and parry window, and visibly changes the blade color + trail.
// All names and flavor text are original.

export const SWORDS = {
  worn: {
    name: 'Worn Traveler\u2019s Katana',
    dmgMul: 1.0, speedMul: 1.0, parryBonus: 0.0,
    blade: 0xd8dce2, glow: 0xffe9b0,
    desc: 'A plain, honest blade. It has never failed its master.',
  },
  crane: {
    name: 'Crane Feather',
    dmgMul: 1.12, speedMul: 1.12, parryBonus: 0.03,
    blade: 0xbfe3ff, glow: 0x9fd8ff,
    desc: 'Light as rumor, quick as a lie. Claimed from Tsubaki.',
  },
  ember: {
    name: 'Ember of the Forge',
    dmgMul: 1.35, speedMul: 0.92, parryBonus: 0.0,
    blade: 0xffb37a, glow: 0xff9a5a,
    desc: 'A heavy rebel blade, still warm with defiance. Claimed from Ren.',
  },
  willow: {
    name: 'Willow in Rain',
    dmgMul: 0.95, speedMul: 1.28, parryBonus: 0.06,
    blade: 0xcfe8c0, glow: 0xd6ffb0,
    desc: 'Bends like a willow; its guard is generous. Claimed at the docks.',
  },
  moon: {
    name: 'Pale Moon Crossing',
    dmgMul: 1.5, speedMul: 1.0, parryBonus: 0.04,
    blade: 0xe8e8ff, glow: 0xcfd0ff,
    desc: 'A magistrate\u2019s heirloom, cold and perfect. Claimed from Captain Isamu.',
  },
};

export const BASE_PARRY_WINDOW = 0.24;

export class SwordRack {
  constructor() {
    this.owned = ['worn'];
    this.equipped = 'worn';
  }

  has(id) { return this.owned.includes(id); }

  // Returns true if this was a new acquisition.
  add(id) {
    if (!SWORDS[id] || this.owned.includes(id)) return false;
    this.owned.push(id);
    return true;
  }

  // Applies the sword's stats + visuals to the player. Returns false if the
  // player doesn't own the sword.
  equip(id, player) {
    if (!SWORDS[id] || !this.owned.includes(id)) return false;
    this.equipped = id;
    const s = SWORDS[id];
    player.dmgMul = s.dmgMul;
    player.speedMul = s.speedMul;
    player.parryWindow = BASE_PARRY_WINDOW + s.parryBonus;
    player.trailColor = s.glow;
    if (player.parts && player.parts.blade && player.parts.blade.material) {
      player.parts.blade.material.color.setHex(s.blade);
    }
    return true;
  }
}
