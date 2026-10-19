# Genre Playbook — Board & Card Games (Chess, Checkers, Tic-Tac-Toe, Card games)

Turn-based logic games where correctness *is* the game: an illegal move or a flaky turn transition breaks trust instantly. The genre is about a clean **turn state machine**, **rigorous legal-move validation**, and (for solo play) a **minimax / alpha-beta AI**. Almost always 2D (Canvas/DOM/Phaser); rendering is trivial, logic is everything. Read `../threejs-foundational.md` for general structure; this file is about game-logic correctness.

---

## 1. Core mechanics (minimal-but-good scope for a demo)

1. **A board/state data model** separate from the view (array of pieces / card lists).
2. **A turn state machine** that governs whose turn it is and what's allowed.
3. **Legal-move generation + validation** (only legal moves accepted and highlighted).
4. **Win/draw/loss detection** (checkmate/stalemate, three-in-a-row, deck empty, etc.).
5. **A simple AI opponent** (minimax/alpha-beta for perfect-info games; heuristics for card games).

For chess specifically: **do not hand-roll the rules for a demo unless asked — use `chess.js`** for move generation/validation/checkmate and `chessground`/a board lib for the UI. Hand-rolling chess rules (en passant, castling through check, pins, promotion, threefold repetition) is a huge bug surface. Build custom rules only for simpler games (tic-tac-toe, checkers, connect-4) or when explicitly required.

---

## 2. Turn state machine

Model the game as an explicit finite state machine — never as ad-hoc booleans scattered across handlers.

- **States (typical):** `PlayerTurn (idle → pieceSelected → moveConfirmed)` → `Resolving/Animating` → `CheckWinCondition` → `OpponentTurn` → … → `GameOver`.
- **One source of truth for `currentPlayer`.** Only accept input during that player's `idle`/`selecting` state; **ignore all input while animating or during the AI's turn** (a top bug: clicking during resolution corrupts state or lets a player move twice).
- **Strict transitions:** select piece → show legal moves → choose destination → validate → apply move → switch player → check terminal conditions. Each step gated; no shortcuts.
- **Undo/history:** keep a move stack (and/or full state snapshots) — enables undo, threefold-repetition detection, and replay. Store enough to reverse a move (captured piece, castling/en-passant flags).
- **Determinism:** the model must be pure and reproducible; the view only reflects it. This makes AI, undo, and (later) networking possible.

---

## 3. Legal-move validation (get this exactly right)

- **Generate, then filter.** Produce candidate moves per piece, then remove illegal ones. **Only offer/accept legal moves** — highlight them; reject clicks on illegal squares.
- **Chess "leaves your king in check" rule (the classic omission):** a move is illegal if, *after* making it, your own king is attacked. Implement by making the move on a copy, checking if your king is attacked, and rejecting if so. This automatically handles pins — don't try to special-case pins.
- **Special chess rules to not forget:** castling (king & rook unmoved, squares empty, king not in/through/into check), en passant (only immediately after the enemy pawn's two-square move), pawn promotion, and the 50-move / threefold-repetition / stalemate draws. These are exactly why `chess.js` exists.
- **Terminal detection:** checkmate = in check AND no legal moves; stalemate = not in check AND no legal moves (a draw, not a loss — a very common bug is scoring stalemate as a win). For tic-tac-toe/connect-4, check all lines after each move and also detect a full-board draw.
- **Card games:** validate legality against the rules (can you play this card now?), keep hidden information hidden (trivial in single-player; true hidden-info enforcement needs a server, which is out of scope on this deploy target), and shuffle with an unbiased algorithm (**Fisher–Yates**, not `sort(() => Math.random()-0.5)` which is biased).

---

## 4. AI: minimax & alpha-beta pruning

For **perfect-information, deterministic** games (chess, checkers, tic-tac-toe, connect-4, Othello):

- **Minimax:** recursively explore the game tree; the mover **maximizes** their evaluation, the opponent **minimizes** it. At the depth limit (or terminal node), return a heuristic evaluation. Choose the move leading to the best guaranteed outcome.
- **Alpha-beta pruning:** carry `alpha` (best already guaranteed to the maximizer) and `beta` (best guaranteed to the minimizer); **prune** a branch when `alpha >= beta` (the opponent would never allow it). Same result as minimax, far fewer nodes → much deeper search. Move ordering (try likely-good moves first, e.g. captures) dramatically improves pruning.
- **Depth = difficulty.** Tic-tac-toe: search to the end (perfect play). Connect-4/checkers: a few plies. Chess: alpha-beta + a material+position evaluation gets a decent club-level bot; go deeper/iterative-deepening for stronger.
- **Evaluation function (heuristic):** for chess, material values (P=1, N/B=3, R=5, Q=9) + piece-square tables (positional bonuses) + mobility/king safety. Terminal nodes return ±∞ for win/loss, 0 for draw. A good eval matters more than raw depth for feel.
- **Avoid freezing the UI:** run deeper searches so they don't block the main thread — use a **Web Worker**, iterative deepening with a time budget, or `requestAnimationFrame`-chunked search. A 2-second UI hang while the AI "thinks" feels broken.
- **Imperfect-info / non-deterministic games** (most card games) need different AI: rules-based heuristics, Monte Carlo (MCTS) with determinization, or expectiminimax. Don't shoehorn plain minimax into a hidden-hand card game.

---

## 5. Common bugs to avoid (checklist)

- **Accepting input during animation or the AI's turn** → double moves / corrupted state. Gate all input on the turn state.
- **Offering illegal moves** → validate + highlight only legal moves; reject the rest.
- **(Chess) Allowing a move that leaves your own king in check** → test post-move king safety; this also handles pins.
- **(Chess) Missing en passant / castling-through-check / promotion / draw rules** → use `chess.js` instead of hand-rolling.
- **Scoring stalemate as a win** → stalemate is a draw; checkmate = in check + no legal moves.
- **Biased shuffle** (`sort(Math.random)`) → use Fisher–Yates.
- **State in the DOM/sprites, not a model** → makes AI, undo, and win-checks unreliable; keep a pure model.
- **AI blocks the main thread** → search in a Web Worker or time-boxed/iterative; keep the UI responsive.
- **Minimax min/max sign errors** → a classic; test against known positions (mate-in-1, forced draws).
- **No move history** → can't undo or detect repetition; keep a reversible move stack.

---

## Defaults to apply

1. **For chess, default to `chess.js` (rules/validation/mate) + a board UI lib** rather than hand-rolling. Hand-roll only simple games (tic-tac-toe, connect-4, checkers) or when asked.
2. **Model the game as an explicit turn state machine with one `currentPlayer` source of truth; ignore input during animation/AI turns.** Keep a pure model separate from the view + a reversible move history.
3. **Only ever present and accept LEGAL moves** (generate → filter → highlight). For chess, reject any move that leaves your own king in check.
4. **Detect terminal states correctly** — checkmate vs stalemate (draw!), draws, and full-board ties.
5. **AI = minimax + alpha-beta pruning with a heuristic eval; depth = difficulty; run in a Web Worker / time-boxed so the UI never freezes.** Use MCTS/heuristics (not plain minimax) for hidden-info card games; shuffle with Fisher–Yates.
6. **Minimal scope:** correct rules + turn FSM + legal-move UI + win/draw detection + one AI opponent. Correctness beats features here.

---

## Sources
- chess.js — move generation, validation, check/checkmate/draw detection: https://github.com/jhlywa/chess.js
- chessground — Lichess's board UI: https://github.com/lichess-org/chessground
- Chess Programming Wiki — Minimax: https://www.chessprogramming.org/Minimax
- Chess Programming Wiki — Alpha-Beta: https://www.chessprogramming.org/Alpha-Beta
- Chess Programming Wiki — Evaluation & Piece-Square Tables: https://www.chessprogramming.org/Evaluation and https://www.chessprogramming.org/Piece-Square_Tables
- Red Blob Games — game trees / minimax intuition and grid tools: https://www.redblobgames.com/
- Wikipedia — Fisher–Yates shuffle (unbiased): https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
- Wikipedia — Monte Carlo tree search (imperfect-info/large trees): https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
- MDN — Web Workers (off-thread AI search): https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
