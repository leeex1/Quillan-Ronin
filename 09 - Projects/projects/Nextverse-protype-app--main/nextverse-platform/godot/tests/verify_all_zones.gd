extends SceneTree
## Verifies all nine design-doc zones instantiate as production scenes.

const ZONE_IDS := [
	"starting_zone_town_square", "starting_zone_market", "starting_zone_forest_edge",
	"dungeon_level_1", "unknown_zone", "dungeon_level_2", "nexus_atrium",
	"guild_temple_1", "live_event_arena",
]

func _initialize() -> void:
	var packed := load("res://worlds/ProductionZone.tscn") as PackedScene
	if packed == null:
		push_error("[FAIL] ProductionZone.tscn missing")
		quit(1)
		return
	for zone_id in ZONE_IDS:
		var zone: Node = packed.instantiate()
		zone.set("zone_id", zone_id)
		root.add_child(zone)
		zone._ready()
		var asset: Node = zone.get_node_or_null("ZoneAsset")
		var environment: Node = zone.get_node_or_null("WorldEnvironment")
		var ground: Node = zone.get_node_or_null("GroundCollider")
		var avatar: Node = zone.get_node_or_null("LocalAvatar")
		var hub: Node = zone.get_node_or_null("PresenceRequest")
		if asset == null or environment == null or ground == null or avatar == null or hub == null:
			push_error("[FAIL] incomplete production zone: " + zone_id)
			quit(1)
			return
		print("[PASS] zone ", zone_id, " asset=", zone.get("loaded_asset_name"))
		zone.queue_free()
	print("[PASS] all 9 production zones verified in Godot 4.7.2")
	quit(0)
