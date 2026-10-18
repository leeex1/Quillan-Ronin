# Save & Persistence for Browser Games (localStorage vs IndexedDB, versioned saves, serialization, autosave)

Consolidated from MDN storage docs and the `idb` library (see Sources). Focus: what an AI builder needs so saves **survive updates, don't corrupt, and don't block the frame** — the usual bugs are lost progress on a schema change, quota errors, and jank from synchronous writes.

---

## 1. Pick the right storage

**`localStorage`** — simple key/value, **synchronous, string-only, ~5MB** per origin.
- Good for: small settings, high scores, current-level, a single small save blob, key bindings, volume.
- **Synchronous = it blocks the main thread.** Fine for occasional small writes; **do NOT write large/frequent blobs to it every frame** (causes hitches). Values must be strings → `JSON.stringify`/`parse`.
- `sessionStorage` = same API but cleared when the tab closes (use for transient state only).

**IndexedDB** — **asynchronous**, transactional, large (hundreds of MB to GB, quota-based), stores **structured data** (objects, arrays, `Blob`, `ArrayBuffer`, typed arrays via the structured clone algorithm — no manual JSON needed).
- Good for: big/complex saves, multiple save slots, replays, generated assets/level caches, offline data.
- The raw API is verbose/callbacky — **use the `idb` wrapper** (Jake Archibald) for a clean promise-based API. Don't hand-roll raw IndexedDB.

**Rule of thumb:** settings/scores/small single save → `localStorage`; anything large, multi-slot, binary, or frequently written → **IndexedDB (via `idb`)**. Cookies are the wrong tool for game saves (tiny, sent on every request).

**Cache API / service worker** is for caching the *game itself* (assets, offline PWA), not for player save data.

---

## 2. ALWAYS version your saves (the #1 lost-progress bug)

Your save schema *will* change between game versions. If you `JSON.parse` an old save into new code with no version handling, you get crashes or silent corruption and angry players who lost progress.

**Rule: every save carries a `version` number, and you write migration steps between versions.**
```js
const SAVE_VERSION = 3;
function migrate(save) {
  let s = { ...save };
  if (s.version === 1) { s.coins = s.gold ?? 0; delete s.gold; s.version = 2; }
  if (s.version === 2) { s.settings = { ...defaults.settings, ...s.settings }; s.version = 3; }
  return s; // now at SAVE_VERSION
}
function loadSave(raw) {
  let save = raw ?? structuredClone(defaultSave);
  if (save.version !== SAVE_VERSION) save = migrate(save);
  return save;
}
```
- Migrations run **in sequence** (v1→v2→v3) so any old save upgrades.
- IndexedDB has its own **schema versioning** via the `open(name, version)` + `onupgradeneeded`/`idb`'s `upgrade` callback — use it to create/alter object stores. That's separate from your *data* version above; you often want both.
- **New fields:** merge over defaults (`{ ...defaults, ...loaded }`) so older saves gain new keys with sane values instead of `undefined`.
- Keep a **backup of the previous save** before overwriting, so a failed migration/corruption is recoverable.

---

## 3. Serialization: what to save

- **Save state, not behavior.** Serialize plain data (positions, inventory, flags, RNG seed + counters). Never try to serialize class instances/functions/DOM/engine objects directly.
- **Save the seed + progress**, not the whole generated world, for procgen games — regenerate deterministically on load (see procgen skill). Massively smaller and robust.
- Use `structuredClone` / IndexedDB for objects with `Map`/`Set`/`Date`/typed arrays; **plain `JSON` drops those** (Map→`{}`, Date→string, `undefined`/functions omitted, `NaN`/`Infinity`→`null`). If using JSON, convert them explicitly.
- **Wrap every load in try/catch.** Corrupt/partial JSON must fall back to defaults, not white-screen the game.
- Consider a **checksum** or at least shape-validation of loaded data before trusting it. For competitive/leaderboard games remember client saves are trivially editable — validate server-side, don't trust local high scores.

---

## 4. Autosave (do it safely)

- **Autosave on meaningful events** (level complete, checkpoint, item gained) and on a throttled timer — not every frame.
- **Debounce/throttle writes** (e.g. at most once every few seconds) to avoid thrashing storage and causing jank; coalesce rapid changes.
- **Save on `visibilitychange` (hidden) and `pagehide`/`beforeunload`** — mobile browsers kill backgrounded tabs without firing `beforeunload` reliably, so `visibilitychange → hidden` is the most reliable "player is leaving" signal. Do a final synchronous-ish flush there (localStorage) or a queued IndexedDB write.
- **Write atomically:** for IndexedDB use a transaction; for localStorage, build the full string then set it once (a half-written multi-key save = corruption). Keep the previous save until the new one is confirmed.
- Give visible feedback ("Saved") and expose manual save/load + multiple slots where appropriate.

---

## 5. Quota, availability & privacy gotchas
- **Storage can throw or be disabled:** private/incognito mode, quota exceeded (`QuotaExceededError`), or blocked third-party storage. **Feature-detect and wrap in try/catch**; degrade gracefully (in-memory only) rather than crashing.
- **Eviction:** browsers can clear storage under pressure ("best-effort" default). For important saves, request **`navigator.storage.persist()`** to reduce eviction risk, and check `navigator.storage.estimate()` for usage/quota.
- Storage is **per-origin**; a domain change loses saves. iOS Safari has historically been aggressive about clearing storage from rarely-visited sites — don't treat browser storage as permanent; offer export/import (download/upload a save file) for anything precious, and cloud save for accounts.
- **`localStorage` blocks the main thread** — keep values small; move big/async work to IndexedDB.

---

## 6. Bug-prevention checklist
- **No version field** → old saves crash new builds / silent data loss; version + sequential migrations.
- **Parsing without try/catch** → corrupt save white-screens the game; fall back to defaults.
- **Merging loaded save without defaults** → new fields are `undefined`; spread over defaults.
- **JSON round-trip of Map/Set/Date/typed arrays** → data silently lost; use structuredClone/IndexedDB or convert explicitly.
- **Frequent/large `localStorage.setItem`** → main-thread hitches; throttle + use IndexedDB for big data.
- **No save on tab hide** → mobile players lose progress; save on `visibilitychange`/`pagehide`.
- **Non-atomic multi-key writes** → partial/corrupt saves; write one blob / use a transaction, keep a backup.
- **Assuming storage always works** → crashes in private mode / over quota; feature-detect + try/catch + graceful degrade.
- **Trusting local high scores** → trivially cheatable; validate server-side for leaderboards.
- **Saving the whole procgen world** → bloated saves; save seed + progress and regenerate.

---

## Defaults to apply
- **Ship a tiny save module by default:** `save(state)` / `load()` with a **`version` field + sequential `migrate()`**, **try/catch → defaults**, and **merge-over-defaults**. This single pattern prevents the most damaging bug (progress lost on update).
- **Storage routing rule baked in:** settings/scores/small save → `localStorage`; large/multi-slot/binary/frequent → **IndexedDB via `idb`**.
- **Autosave defaults:** save on checkpoints + throttled timer + **`visibilitychange`/`pagehide`**, written atomically with a retained backup.
- **Persist the seed, not the world** for procgen; offer **export/import** for precious saves and request `storage.persist()`. Never trust local scores for leaderboards.

---

## Sources
- MDN — Web Storage API (`localStorage`/`sessionStorage`): https://developer.mozilla.org/en-US/docs/Web/API/Web_Storage_API
- MDN — IndexedDB API + Using IndexedDB (versioning/`onupgradeneeded`): https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API , https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API/Using_IndexedDB
- MDN — Structured clone algorithm (what serializes): https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm ; `structuredClone()`: https://developer.mozilla.org/en-US/docs/Web/API/structuredClone
- MDN — Storage quotas & eviction / `navigator.storage`: https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria , https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist
- MDN — `Document.visibilitychange` / Page Visibility API: https://developer.mozilla.org/en-US/docs/Web/API/Page_Visibility_API
- `idb` (promise-based IndexedDB wrapper, Jake Archibald): https://github.com/jakearchibald/idb
- web.dev — "Storage for the web" (choosing storage): https://web.dev/articles/storage-for-the-web
- localForage (localStorage-like API over IndexedDB): https://github.com/localForage/localForage
