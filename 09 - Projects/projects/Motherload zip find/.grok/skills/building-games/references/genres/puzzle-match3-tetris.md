# Genre Playbook — Grid Puzzle (Match-3 & Tetris)

Grid-based puzzlers (Bejeweled/Candy Crush match-3, Tetris) live or die on **a clean grid data model** and **exactly-correct board logic**. Both are 2D — Canvas/Phaser/DOM all work; the board is data, the rendering is a thin view. Read `../threejs-foundational.md` first (delta time, pooling). This file gives the precise algorithms, including the full **SRS wall-kick tables** (verbatim from the Tetris wiki).

---

## 0. The golden rule: separate model from view

**The board is a 2D array of ints/enums; rendering reads from it.** Never store game state in sprite positions or the DOM. All logic (matches, gravity, rotation, collision) runs on the array; then you sync visuals (tween pieces to their cell). This prevents the entire class of "the animation and the logic disagree" bugs.

---

# Part A — Match-3 (Bejeweled / Candy Crush)

## A1. Core mechanics & minimal scope
1. A **grid** (e.g. 8×8) of colored gems (`grid[row][col] = colorId`).
2. **Swap two adjacent gems** (click-click or drag); the swap is only legal if it creates a match.
3. **Match detection** (3+ in a row/column) → clear.
4. **Gravity** (gems above fall into gaps) + **refill** (new gems drop from the top).
5. **Cascades** (chain reactions) scored with a multiplier, and a score/moves HUD.

That's a complete match-3. Add special gems, objectives, and levels later.

## A2. Match detection
- **Scan rows and columns for runs of ≥3 identical colors.** The simplest robust method: for each cell, count consecutive same-color horizontally and vertically; mark any run of length ≥3. Collect all marked cells into a set (so an L/T shape counts once).
- **Flood fill** is the right tool when matches aren't strictly lines (e.g. same-color blobs, or Puyo Puyo–style groups): from each unvisited cell, BFS/DFS to all 4-connected same-color neighbors; if the connected group size ≥ threshold, it's a match. Use flood fill for group-clear variants and for detecting connected clusters after cascades.
- **Swap validation:** perform the swap in the array, run match detection; if no match results, **swap back** (and animate the reversal). Only commit swaps that match.
- **Initial board must have no pre-existing matches** and should have at least one valid move — generate, then re-roll any accidental matches; optionally verify a move exists (shuffle if deadlocked).

## A3. Gravity, refill & cascades
- **Clear** matched cells (set to empty), award score.
- **Gravity:** for each column, compact non-empty cells downward (iterate from the bottom, pull the next non-empty cell down). Do this on the array first.
- **Refill:** fill the now-empty top cells with new random gems.
- **Cascade loop:** after gravity+refill, **run match detection again**; if new matches formed, clear/score them (with an increasing chain multiplier) and repeat until stable. This loop is the satisfying core — don't stop after one pass.
- **Animate after resolving:** compute the whole settled state, then tween gems to their new cells (falling animation). Lock input while the board is resolving.

---

# Part B — Tetris

## B1. Core mechanics & minimal scope
1. A **playfield** (standard 10 wide × 20 visible, plus hidden rows above for spawning).
2. **7 tetrominoes** (I, O, T, S, Z, J, L) each with 4 rotation states.
3. **Gravity** (piece falls on a timer), **soft/hard drop**, **move left/right**, **rotate CW/CCW**.
4. **Line clears** (full rows removed, rows above shift down) + scoring.
5. **Spawn next piece; game over** when a new piece can't spawn.

Guideline-quality Tetris additionally needs: **7-bag randomizer**, **SRS rotation + wall kicks**, **lock delay**, ghost piece, hold, and a next-queue. Do these — they're what players expect.

## B2. Data model & collision
- Represent each piece as a set of 4 cell offsets for its current rotation state, plus a board position `(x, y)`. The board is `grid[row][col]`.
- **Collision test:** a move/rotation is valid iff every one of the piece's 4 cells is in-bounds and lands on an empty board cell. Check *before* committing any move.
- **Move/rotate = test the target; if valid, apply; else reject** (rotation additionally tries wall kicks, below).
- **Line clear:** find full rows, remove them, shift everything above down, and clear the freed top rows. Award by lines cleared at once (1=Single … 4=Tetris) — Tetris (4) scores far more, incentivizing well-building.

## B3. Randomizer, gravity, lock delay
- **7-bag randomizer (mandatory for modern feel):** shuffle a bag of all 7 pieces, deal them out, refill/reshuffle when empty. Guarantees no long droughts and no floods of one piece. Pure random feels bad.
- **Gravity:** drop one row every `fallInterval` (decreasing with level). **Soft drop** = faster fall; **hard drop** = instantly place at the lowest valid position (+ small score per cell).
- **Lock delay (~0.5s):** when a piece lands (can't fall further), don't lock it immediately — give a short window during which moving/rotating keeps it alive. Standard is a **move/rotation reset limited to ~15 resets** (so you can't stall forever). Without lock delay the game feels harsh; with an unlimited reset you can stall infinitely — cap the resets.
- **DAS/ARR:** delayed auto-shift (a pause before a held direction auto-repeats) + auto-repeat rate. Tune these — they define movement feel for skilled play.

## B4. SRS rotation & wall kicks (verbatim tables)

**Rotation states:** `0` = spawn, `R` = one CW from spawn, `L` = one CCW from spawn, `2` = 180°.
When a rotation is attempted, test **5 offsets in order**; use the first that fits; if none fit, the rotation fails entirely.

**⚠️ Coordinate convention:** these tables use **x = right positive, y = UP positive**. If your grid's row index increases *downward* (the usual case), **negate the y value** when applying kicks (a `+2` up becomes `row − 2`). Getting this sign wrong is the #1 SRS bug.

**J, L, S, T, Z kick data** (all share one table):
```
0->R:  (0,0) (-1,0) (-1,+1) (0,-2) (-1,-2)
R->0:  (0,0) (+1,0) (+1,-1) (0,+2) (+1,+2)
R->2:  (0,0) (+1,0) (+1,-1) (0,+2) (+1,+2)
2->R:  (0,0) (-1,0) (-1,+1) (0,-2) (-1,-2)
2->L:  (0,0) (+1,0) (+1,+1) (0,-2) (+1,-2)
L->2:  (0,0) (-1,0) (-1,-1) (0,+2) (-1,+2)
L->0:  (0,0) (-1,0) (-1,-1) (0,+2) (-1,+2)
0->L:  (0,0) (+1,0) (+1,+1) (0,-2) (+1,-2)
```

**I piece kick data** (its own table):
```
0->R:  (0,0) (-2,0) (+1,0) (-2,-1) (+1,+2)
R->0:  (0,0) (+2,0) (-1,0) (+2,+1) (-1,-2)
R->2:  (0,0) (-1,0) (+2,0) (-1,+2) (+2,-1)
2->R:  (0,0) (+1,0) (-2,0) (+1,-2) (-2,+1)
2->L:  (0,0) (+2,0) (-1,0) (+2,+1) (-1,-2)
L->2:  (0,0) (-2,0) (+1,0) (-2,-1) (+1,+2)
L->0:  (0,0) (+1,0) (-2,0) (+1,-2) (-2,+1)
0->L:  (0,0) (-1,0) (+2,0) (-1,+2) (+2,-1)
```

**O piece does not kick** (it never needs to — its rotation is trivial; only its rotation center appears to move).

**Algorithm:** to rotate, compute the piece cells in the target state (pure rotation about the piece center), then for each of the 5 `(dx, dy)` offsets for that specific `from->to` transition, test the piece translated by `(dx, -dy)` (if y is down); apply the first that fits. If all 5 fail, cancel. This is what enables T-spins and squeezing the I-piece into tight wells.

## B5. Match-3 & Tetris common bugs (checklist)
- **State stored in sprites/DOM instead of an array** → logic/animation desync. Model is the array.
- **(Match-3) Not swapping back on a non-matching swap** → illegal moves stick.
- **(Match-3) Cascade loop stops after one pass** → miss chain reactions; loop until stable.
- **(Match-3) Initial board has matches or no valid moves** → re-roll on generate; detect deadlock and shuffle.
- **(Tetris) Pure random piece order** → droughts feel awful; use a 7-bag.
- **(Tetris) SRS y-sign flipped** → wall kicks push pieces the wrong way; negate y for a downward-row grid.
- **(Tetris) Wrong/omitted kick table** → no T-spins, I-piece won't fit tight spots; use the exact 5-offset tables per transition (I has its own; O doesn't kick).
- **(Tetris) No lock delay, or unlimited reset** → either too harsh, or you can stall forever; cap resets (~15).
- **(Tetris) Line clear shifts rows incorrectly** → clear from the model, shift above down, clear freed top rows; test multi-line clears.
- **(Tetris) Collision checked after committing** → test the target position first, commit only if valid.
- **Timing tied to frame rate** → gravity/fall interval must use delta/real time.

---

## Defaults to apply

1. **Model = 2D array; view is a thin renderer that tweens to cells.** Run all logic on the array, then animate. Lock input while resolving.
2. **Match-3: swap → detect → (if no match) swap back; clear → gravity → refill → re-detect in a cascade loop until stable.** Use line-scan for line matches and **flood fill** for connected-group/blob matches. Generate boards with no initial matches and at least one move.
3. **Tetris: implement the full guideline — 7-bag randomizer, SRS with the exact wall-kick tables above, lock delay with capped resets, hard/soft drop, ghost piece, hold, next queue.** Players expect all of it.
4. **SRS: mind the y-sign** (tables are y-up; negate for row-down grids). I-piece uses its own kick table; O doesn't kick.
5. **Always test a move/rotation against the model before committing; reject if any cell is out of bounds or occupied.**
6. **Both genres are timer-driven** — use delta/real time for fall speed and animations, never per-frame.

---

## Sources
- Hard Drop Tetris Wiki — SRS (spawn, rotation, and the verbatim wall-kick tables above): https://harddrop.com/wiki/SRS
- Hard Drop Tetris Wiki — Wall kick: https://harddrop.com/wiki/Wall_kick
- Hard Drop Tetris Wiki — Lock delay: https://harddrop.com/wiki/Lock_delay
- Hard Drop Tetris Wiki — Random Generator (7-bag): https://harddrop.com/wiki/Random_Generator
- Tetris Guideline (official standard): https://tetris.wiki/Tetris_Guideline
- "Tetris" implementation guide (javidx9 / general): https://tetris.wiki/Tetris_(NES,_Nintendo)
- Wikipedia — Flood fill (BFS/DFS): https://en.wikipedia.org/wiki/Flood_fill
- Bejeweled/Candy Crush match-3 algorithm write-ups (match detection + cascade): https://www.emanueleferonato.com/2018/07/13/build-a-html5-match-3-game-using-phaser/
