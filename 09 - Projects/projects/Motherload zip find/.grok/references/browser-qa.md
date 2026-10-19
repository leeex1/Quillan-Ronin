# Browser QA (agent-driven only; the user is not your QA)

`AGENTS.md` § "Execution loop" states the mandatory pass. This is the menu of
capabilities and the depth judgment around it.

Everything here runs **in the sandbox** against `http://127.0.0.1:8080` — it is
**not** the user's Grok chat tab. Use whatever browser capability you have
**yourself**, so quality beats curl-only.

1. **Grok browser / computer-use / MCP browser tools**, if listed — open
   `http://127.0.0.1:8080`, glance at the UI, screenshot if supported.
2. **`web_fetch`** on that URL for an HTML-only check.
3. **Playwright helper (preinstalled)** — one run loads desktop **and** mobile,
   screenshots both, and prints a JSON verdict.

```bash
mkdir -p /workspace/screenshots
node scripts/browser-smoke.mjs http://127.0.0.1:8080/ /workspace/screenshots/app-builder-preview.png
# Writes app-builder-preview.png (desktop), -mobile.png, and .json (verdict).
# Then Read BOTH PNGs in one batched read if you have an image tool, and iterate if either looks wrong.
```

One run audits desktop (1280×800) **and** mobile (390×844): two PNGs (mobile
gets a `-mobile` suffix) plus a JSON verdict (per-viewport console/page errors,
body text, horizontal overflow) on stdout and next to the PNG (`.json`). It
writes under `/workspace/screenshots/` by default; pass an explicit path only
for a different name **under that directory**.

4. **`agent-browser` (preinstalled)** — the interactive pass: click, type, hold
   keys, read state back out. § "Interactive QA" below; it is also what the
   `controls` §5c self-test runs.

## Built output

Serve the build with `npm run preview:restart` (loopback `127.0.0.1:8081`) and
reuse the dev verdict as a baseline — the JSON reports `divergesFromBaseline`, so
you only re-read the built screenshots when it flags.

```bash
npm run preview:restart   # built output on 127.0.0.1:8081 — QA only, never the live preview
node scripts/browser-smoke.mjs http://127.0.0.1:8081/ /workspace/screenshots/app-builder-built.png --baseline /workspace/screenshots/app-builder-preview.json
npm run preview:stop      # frees :8081 when the built-output QA is done
```

`preview:restart` (`scripts/preview.mjs`) kills whatever holds `:8081` — the
port's owner, whoever started it — then serves the current build in the
background and returns once it answers. Use it instead of bare
`npm run preview`: `vite preview` is strictPort, so a preview left running from
an earlier build both fails a plain start and keeps serving stale output.

## Interactive QA — `agent-browser`

```bash
agent-browser open http://127.0.0.1:8080/
agent-browser snapshot -i                        # a11y tree with @e1 refs
agent-browser click @e1                          # or: find text "Start" click
agent-browser fill "#email" you@example.com      # type <text> for keystrokes
agent-browser press Enter                        # a tap; holding: see "Keys"
agent-browser wait --fn "window.__ready === true"   # or: wait 500 (ms)
agent-browser eval "document.querySelectorAll('.card').length"
agent-browser console                            # or: errors (page errors)
agent-browser screenshot /workspace/screenshots/step.png
agent-browser close
```

**Any multi-verb flow goes in one `batch`.** Each invocation is a tool call of
its own, so a 10-verb flow costs ten round trips; `batch` runs the flow in one.
Reach for it by default and drop to single commands only for a one-off probe.
It takes the commands as a JSON array of string arrays on stdin, which also
sidesteps the shell quoting a flow with quotes in its `eval` would need:

```bash
agent-browser batch --bail <<'JSON'
[["open","http://127.0.0.1:8080/"],
 ["find","text","Add note","click"],
 ["fill","#title","Grocery list"],
 ["press","Enter"],
 ["wait","300"],
 ["eval","if (![...document.querySelectorAll('li')].some(n => n.textContent.includes('Grocery list'))) throw Error('note did not appear in the list')"],
 ["screenshot","/workspace/screenshots/note-added.png"]]
JSON
```

That is the shape for any app: drive the thing a user would do, then assert the
result with a throwing `eval`. It exits 0 when the note lands and 1 when it does
not, so one tool call is the whole verdict. No `close` — the page is still there
for the next check.

`--bail` stops at the first failing command; without it the later ones still
run. Either way a failed step exits non-zero, so a batch is a verdict. That is
how to keep an interactive pass cheap — one tool call instead of one per verb.
`controls` §5c is the worked example.

**`open` once.** The page outlives the command: a later invocation, batched or
not, keeps the DOM and any `window` state you set. A second `open` re-navigates
and **discards that state**, so starting every batch with one both costs a page
load and throws away the state you just built. Open at the start of a pass and
again only after a rebuild, or to reset deliberately.

**Match the depth of the pass to the app.** A page that only renders is done
once the smoke screenshots look right — the interactive verbs are for apps with
something to drive. Verify each behaviour once; re-run a check after you change
the code it covers, not otherwise.

**A `✓` from `open` is not "the page loaded".** An HTTP error page reports
`✓` too (a 404 as `✓ Error response`), so assert on page content — a `find`, a
throwing `eval` — rather than trusting the tick.

The image build asserts the core verbs stay in the CLI's subcommand list.
**Their arguments and flags are not checked** — they are upstream's, and
upstream ships ~17 releases a month, so a manual here would be stale on arrival:
if a command rejects what you typed, `agent-browser <cmd> --help` decides, not
this file.

**`@ref`s go stale.** A `@e1` is valid only for the `snapshot` that produced
it, so re-snapshot after any DOM change — navigation, a click that re-renders,
a route change — rather than reusing an old ref. It is the most common misuse.

**`find text` is case-sensitive.** It matches a substring of an element's
trimmed text — `find text "Play" click` hits `Play now` — but `start` will not
find `Start`. Copy the label from `snapshot -i`; guessing it costs a failed
command and a re-snapshot.

**Keys.** `press <key>` sends a real `event.code` (`press d` → `KeyD`) and is
the only single-key command `--help` documents; raw keystrokes at the current
focus, with no selector, are `keyboard type <text>`. `keydown`/`keyup` exist but
are undocumented and send **`event.code === ""`**, so a game that tracks
`event.code` never sees them — do not build a check on either. Hold keys from
inside the page instead: the game's own probe (`controls`
§5b), or dispatch the event yourself, on `document.body` with `bubbles: true`
so listeners on `document` or `window` see it — dispatching on `window` itself
does not reach a `document` listener:

```bash
agent-browser eval "document.body.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyA', key: 'a', bubbles: true }))"
```

**Falsy is not failure — throw.** `eval "false"` exits **0**; `eval "if (!cond)
throw Error('why')"` exits **1**. Throwing is what turns a check into a verdict
instead of a line of output you have to read yourself; `controls` §5c is the
worked example. Anything past one expression goes in an IIFE —
`eval "(() => { … })()"`, the form the image build asserts and the only one you
can re-run (a bare `const x = …` fails the second time with "already
declared") — or through `eval --stdin` with a heredoc, or `eval -b <base64>`,
when the quoting gets awkward. `eval` awaits a promise, so an `async` IIFE
works and its rejection still exits 1: that is how you time something in page
time rather than across command round-trips.

**Output modes.** Plain output is a line per command (`eval "1+1"` prints `2`)
and is the default for a reason: `--json` wraps each one in a repeated
`lifecycle` block, measured at ~1 KB for three commands whichever three, so it
is worth it only when you are parsing. Same for
snapshots: prefer `snapshot -i | grep -i "<thing>"` or `snapshot -s "#sel"`,
and keep a bare `snapshot` for a page you cannot otherwise navigate.

**Operating notes** (conventions, not enforcement): loopback targets and
screenshots under `/workspace/screenshots/`, the same rules as everywhere else
here; a `--session <name>` per flow if you ever run two at once; if a command
hangs, `agent-browser close` and retry.

### Fallback

`agent-browser` is the tool for interactive QA. If it fails **twice on the same
step** — daemon wedged, a command it does not support — fall back to a short
Playwright script (still installed) and say so in one line of your summary.

## How deep to go

Depth **beyond the mandatory smoke pass + screenshot read** is your judgment: a
landing page usually needs nothing more than that pass. For a game with WASD /
vehicles / flight, still verify control signs (A left / D right from a chase
cam) per `.grok/skills/controls/SKILL.md` — you don't have to play end-to-end,
but inverted A/D must not ship.
