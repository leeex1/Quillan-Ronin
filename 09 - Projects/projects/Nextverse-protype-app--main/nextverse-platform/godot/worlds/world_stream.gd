extends Node3D
## Connected world layout: nine production zones placed on a ring with
## portal markers. Each zone keeps its own PBR scene instance.

const ZONE_IDS := [
	"starting_zone_town_square", "starting_zone_market", "starting_zone_forest_edge",
	"dungeon_level_1", "unknown_zone", "dungeon_level_2", "nexus_atrium",
	"guild_temple_1", "live_event_arena",
]

@export var ring_radius: float = 40.0

var zone_nodes: Array[Node3D] = []
var portals: Array[Area3D] = []

func _ready() -> void:
	for index in range(ZONE_IDS.size()):
		var packed := load("res://worlds/ProductionZone.tscn") as PackedScene
		if packed == null:
			continue
		var zone: Node3D = packed.instantiate() as Node3D
		zone.set("zone_id", ZONE_IDS[index])
		zone.name = "Zone_%s" % ZONE_IDS[index]
		var angle: float = TAU * float(index) / float(ZONE_IDS.size())
		zone.position = Vector3(cos(angle) * ring_radius, 0.0, sin(angle) * ring_radius)
		add_child(zone)
		zone_nodes.append(zone)

		var portal := Area3D.new()
		portal.name = "Portal_%s" % ZONE_IDS[index]
		var shape := CollisionShape3D.new()
		var sphere := SphereShape3D.new()
		sphere.radius = 2.0
		shape.shape = sphere
		portal.add_child(shape)
		portal.position = zone.position + Vector3(0.0, 1.0, 0.0)
		add_child(portal)
		portals.append(portal)

func zone_count() -> int:
	return zone_nodes.size()

func portal_count() -> int:
	return portals.size()
