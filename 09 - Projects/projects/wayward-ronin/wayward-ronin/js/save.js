// save.js — localStorage persistence for Wayward Ronin.
// Saves are plain versioned JSON. Anything that can't be restored safely
// (live enemies) is serialized as spawn defs and re-spawned on load.

const KEY = 'wayward-ronin-save-v1';
export const SAVE_VERSION = 1;

export function saveGame(data) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...data, v: SAVE_VERSION }));
    return true;
  } catch (err) {
    return false;
  }
}

export function loadGame() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    return d && d.v === SAVE_VERSION ? d : null;
  } catch (err) {
    return null;
  }
}

export function clearSave() {
  try { localStorage.removeItem(KEY); } catch (err) { /* storage unavailable */ }
}

export function hasSave() {
  try { return !!localStorage.getItem(KEY); } catch (err) { return false; }
}
