# Skill: debug
Use when the game crashes, won't run, or shows errors.

Loop (max 5):
1. Reproduce: run_game, capture last 20 log lines.
2. Isolate: which scene/script changed last?
3. Fix ONE cause at a time.
4. Re-run. If green, report file + line.

Never rewrite a whole script to fix one bug.
