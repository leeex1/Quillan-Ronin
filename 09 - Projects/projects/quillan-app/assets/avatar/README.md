# Avatar Sprite Sheets

Drop your custom sheets here:

- `quillan_sheet.png`  — Quillan (your ronin)
- `jdxx_sheet.png`     — JDXX (you)
- `clippy_sheet.png`   — Clippy demo / generic

## Format
- Horizontal strip or grid. App auto-detects:
  - Strip: 1 row, N columns (frameW = sheet.width / frames)
  - Grid: configured via window.setAvatarFrames({ frameW, frameH, states })
- Recommended frame size: 200×200 px (matches window)
- States the engine expects (procedural fallback covers all if sheet missing):
  `idle` (4f @6fps), `blink` (3f), `talk` (4f @12fps), `think`, `dance`, `walk`, `chill`, `desk`, `stream`

## Wiring custom sheets at runtime
```js
// From DevTools or your loader:
avatar.setSpriteSheet("assets/avatar/my_sheet.png")
avatar.setSpriteConfig({ frameW:256, frameH:256, states:{ idle:{fps:6, frames:8}, dance:{fps:14, frames:8} } })
```

## Without sheets
App draws a clean procedural vector avatar (no sheet required) — so it works immediately demo.

3D GLB fallback (`quillan_textured.glb`, `jdxx_textured.glb`) still loads if available and if config.type="glb".
