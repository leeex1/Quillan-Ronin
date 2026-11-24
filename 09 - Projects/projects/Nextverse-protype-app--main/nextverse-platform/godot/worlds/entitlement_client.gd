extends Node
## Client wrapper for the server-side entitlement authority.
## The client asks; it never grants items itself.

@export var orchestrator_url: String = "http://127.0.0.1:3006"

var last_export: Dictionary = {}
var last_error: String = ""

func parse_export(body: String) -> bool:
	var parsed: Variant = JSON.parse_string(body)
	if parsed == null or not parsed is Dictionary:
		last_error = "invalid entitlement response"
		return false
	var payload: Dictionary = parsed
	if String(payload.get("status", "")) != "success":
		last_error = String(payload.get("message", "entitlement request failed"))
		return false
	var result: Dictionary = payload.get("result", {})
	if not result is Dictionary:
		last_error = "missing entitlement result"
		return false
	if not bool(result.get("provenance_valid", false)):
		last_error = "provenance check failed"
		return false
	last_export = result
	return true

func owns_item(item_id: String) -> bool:
	var account: Dictionary = last_export.get("account", {})
	var items: Array = account.get("items", [])
	return items.has(item_id)

func tier() -> String:
	var account: Dictionary = last_export.get("account", {})
	return String(account.get("tier", "FREE"))
