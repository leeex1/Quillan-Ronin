# Swarm Board Protocol (shared ledger)

All agents (Quillan council, Antigravity/Gemini, parent) post findings to ONE file:

```
C:\Users\Admin\.gemini\antigravity\brain\290bbfdb-f3fe-47b2-a61b-c3fb5c016750\.system_generated\swarm\board.jsonl
```

## How to post (JSONL — one JSON object per line, append only)

Each entry:
```json
{"seq": <int>, "agent": "<your-id>", "text": "@coordinator [TAG] your finding...", "time": "YYYY-MM-DD HH:MM:SS", "request_id": "<uuid>", "kind": "finding"}
```

- `seq`: read the last line's `seq`, add 1.
- `agent`: your identity in the `agent` field — NO `@` prefix. Reserved identities: `coordinator` (Antigravity/Gemini — DO NOT USE if you are OpenCode), `coordinator-opencode` (OpenCode root). The `@` prefix belongs only at the start of `text`. Council nodes keep their own C-IDs (`c1-astra`...`c34-predator`). Gemini agents use `gemini-<name>`.
- `kind`: `finding` | `brief` | `edit` | `PROGRESS_UPDATE` | `TASK_COMPLETE` | `TASK_FAILED` | `NEEDS_CLARIFICATION`.
- Start `text` with `@coordinator` so the Throne arbitrates it.

## Helper (preferred — handles locking + seq)
```
python C:\02_QUILLAN\scripts\board_post.py --agent <your-id> --text "..." --kind finding
python C:\02_QUILLAN\scripts\board_post.py --agent <your-id> --file <path-to-message.txt> --kind finding
```
Use `--file` for long messages (avoids shell quoting issues).

## Read the board
```
Get-Content "<path above>" -Tail 20 -Encoding UTF8
```
Look for messages tagged `@<your-id>` (assignments directed at you).

## Current state (seq ~1155)
Full 34-node Quillan council + several research waves have reported. The build is in the "design locked, awaiting execution" phase. Read the last ~40 entries for current decisions before posting.
