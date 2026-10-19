# Hibernate, revive, and follow-up turns

`AGENTS.md` § "`/workspace/startup.sh`" holds the rules for the file itself.
This is the surrounding lifecycle behaviour.

## Session shapes you may land in

- **Fresh `/workspace`** — template + `node_modules` only, no app routes, no
  `startup.sh`. Scaffold, then write `startup.sh`.
- **Hibernate / revive (snapshot restore)** — the platform re-runs
  `/workspace/startup.sh` if it exists. Everything else in the workspace comes
  back as it was, including the app source. Your job on every turn is to leave
  that file able to bring the preview back on its own.
- **Reboot / recreate** — may wipe app files back to the template. Re-scaffold
  and **restore `startup.sh`** before verifying the preview.

A revive with no `startup.sh` leaves nothing listening on `:8080`, so the user
sees an empty preview pane.

## A `startup.sh` that satisfies the rules

```sh
#!/bin/sh
set -eu
cd /workspace
# :8081 is QA-only — a revive must never inherit a stale built-output preview.
# Called directly, not via npm: no node_modules needed, so nothing to wait for.
node scripts/preview.mjs stop || true
if curl -sf -o /dev/null --max-time 2 http://127.0.0.1:8080/; then
  exit 0
fi
npm run dev >>/tmp/app-startup.log 2>&1 &
```

## Follow-up turns (multi-turn continuity)

- Edit in place. Do not re-scaffold unless files were wiped or the change is too
  big to patch cleanly.
- Vite HMR pushes source edits to the preview instantly. Restart the dev server
  **only** for `vite.config` / dependency changes — and update `startup.sh` in
  the same turn if the restart command changed.
- Killing the dev server blanks the user's preview mid-session.
- After edits, re-run the smoke pass; re-read the screenshots only when the edit
  touched visible UI.
