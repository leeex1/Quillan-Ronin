# Quillan MazeBench Harness

This directory contains the isolated local harness for preparing Quillan for
[MazeBench](https://mazebench.com/) runs.

## Current scope

- Pins the MazeBench package version used for local smoke testing.
- Defines bounded run limits and a safe artifact directory.
- Normalizes JSON observations into a small model-facing contract.
- Validates model actions before they are sent to a MazeBench runner.
- Records redacted, bounded trajectory metadata for later comparison.

The harness does not modify Quillan model code and does not claim leaderboard
compatibility by itself. Official agent runs may require Prime Sandbox access.

## Prerequisites

- Python 3.9 or newer
- Node.js, required by MazeBench
- A Quillan inference entrypoint that can consume an observation and return one
  action from the MazeBench action set

## Local setup

From the repository root:

```text
py -3 -m venv .venv-mazebench
.venv-mazebench\\Scripts\\python.exe -m pip install --upgrade pip
.venv-mazebench\\Scripts\\python.exe -m pip install -r benchmarks\\mazebench\\requirements.txt
```

Keep MazeBench's materialized runtime inside the repository instead of the
user profile:

```text
set MAZEBENCH_HOME=C:\\02_QUILLAN\\benchmarks\\mazebench\\runtime
```

Launch the local MazeBench UI:

```text
set MAZEBENCH_HOME=C:\\02_QUILLAN\\benchmarks\\mazebench\\runtime
.venv-mazebench\\Scripts\\mazebench.exe launch
```

Useful deterministic inspection modes:

```text
.venv-mazebench\\Scripts\\mazebench.exe ascii --level CxD
.venv-mazebench\\Scripts\\mazebench.exe json --level CxD --once
```

Run the adapter tests without installing MazeBench:

```text
.venv-mazebench\\Scripts\\python.exe benchmarks\\mazebench\\smoke_test.py
```

## Integration contract

The model callback receives a normalized observation dictionary and returns a
dictionary containing exactly one `action` string. MazeBench JSON uses an
`objects` mapping grouped by object type. The adapter accepts the
MazeBench movement, camera, undo, reset, and teleport action names and rejects
unknown actions before execution.

The actual browser/MCP transport remains outside this package boundary. This
keeps the local tests deterministic and allows the same policy callback to be
connected to a local runner or Prime Sandbox later.
