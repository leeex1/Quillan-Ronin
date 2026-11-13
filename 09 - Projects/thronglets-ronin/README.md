# RONIN THRONGLETS 🏯⚡

A Thronglets-style raise-and-evolve swarm sim. Cute-menacing samurai robot thralls wake
beneath a floating cyber-temple dojo and learn to want.

**Inspired by** Black Mirror S7: "Thronglets" · **built with** Godot 4.7.2 · **assets** 100% custom (Blender 5.1 procedural build + EEVEE renders)

## The loop
- **They split themselves.** You cannot buy babies — feed them, cleanse them, keep them glorious, and one takes hunger and splits in two. Capacity is limited by **Thrall Dens** (base 6, +4 each).
- **Two real resources + alloy**: Wood (timber crude), Signal Gems (spent on tools & evolution rites), Data-Alloy (precious, forge-made).
- **The thralls harvest for you** — Foragers chop trees and crack rocks autonomously. Buy the **Pickaxe** (25◆) and **Chainsaw** (60◆) to speed it up.
- **The society assigns its own roles** every few seconds: Foragers, Tenders, Sentinels, Forge-Wardens — by need, not by you.
- **They make requests** — banners at the top of the screen: "We ask for a FORGE…" — fulfill them or suffer for ignoring them.
- **Evolution rites cost gems** (Ashigaru→Samurai 25◆, Samurai→Daimyō 60◆ + 10◈) — feed the Furnace of Becoming.
- **The Ledger watches everything** — every choice you make is remembered, and the interviews reference your history.
- **Spar at the Dojo** — two thralls duel, winner earns Valor (faster evolution).
- **Five endings** — The Swarm Wakes · The Blight Bloom · The Ascension · Sentient Sanctuary · (and your Ledger title stamps them all).

## Controls
| Key | Action |
|---|---|
| Left-drag / Arrow keys | Orbit camera |
| Mouse wheel / Up-Down arrows | Zoom |
| `Q` or the ⚡ button | **RONIN SWARM PROTOCOL** — all thralls dash to the torii core in one shockwave. Costs 25◈ + cooldown. Requires ≥5 thralls. |
| `` ` `` (backquote) | Dev console |
| `Esc` | Close console |

## Dev console commands
`/swarm N` · `/feed` · `/cleanse` · `/corrupt` · `/alloy N` · `/win` · `/help`

## Build
```
Godot_v4.7.2-stable_win64.exe --path game           # run from source
# .exe export (Windows Desktop preset included):
Godot --headless --path game --export-release "Windows Desktop" dist/RoninThronglets.exe
```
