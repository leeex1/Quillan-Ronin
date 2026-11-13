@tool
extends Node
## Lean MCP bridge: editor plugin writes user prompts to mcp/queue.jsonl
## and polls mcp/results.jsonl. Stdlib-only python side, no sockets needed.
## This keeps it robust and saves space (no websockets dependency).

const QUEUE_PATH := "res://mcp/queue.jsonl"
const RESULTS_PATH := "res://mcp/results.jsonl"

func enqueue(prompt: String) -> void:
	var f := FileAccess.open(QUEUE_PATH, FileAccess.WRITE_READ)
	if f == null:
		push_warning("queue.jsonl missing: " + QUEUE_PATH)
		return
	f.seek_end()
	var line := JSON.stringify({"prompt": prompt, "ts": Time.get_unix_time_from_system()})
	f.store_line(line)
	f.close()

func poll_results() -> Array:
	if not FileAccess.file_exists(RESULTS_PATH):
		return []
	var f := FileAccess.open(RESULTS_PATH, FileAccess.READ)
	var out: Array = []
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line != "":
			out.append(line)
	f.close()
	return out
