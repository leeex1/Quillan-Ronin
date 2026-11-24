extends Node
## Quest state machine grounded in MMORPG_UNIVERSE_DESIGN_2026.md:
## three quests, five forward-only PlayerQuestStatus values.

const STATUS_NAMES := ["NOT_ELIGIBLE", "NOT_ACCEPTED", "ACCEPTED", "READY_TO_COMPLETE", "COMPLETED"]

const QUESTS := {
	1: {"name": "Collect Iron Ore", "objective": "COLLECT_ITEM", "start_npc": 1001},
	2: {"name": "The Lost Sword", "objective": "TALK_TO_NPC", "start_npc": 1002},
	3: {"name": "Urgent Delivery", "objective": "TALK_TO_NPC", "start_npc": 1001},
}

var statuses: Dictionary = {}

func status_for(quest_id: int) -> int:
	return int(statuses.get(quest_id, 0))

func transition(quest_id: int, target: int) -> bool:
	if not QUESTS.has(quest_id) or target < 0 or target >= STATUS_NAMES.size():
		return false
	var current: int = status_for(quest_id)
	# Forward-only transitions; no skipping.
	if target != current and target != current + 1:
		return false
	statuses[quest_id] = target
	return true

func accept(quest_id: int) -> bool:
	return transition(quest_id, 2)

func complete(quest_id: int) -> bool:
	if status_for(quest_id) == 3:
		return transition(quest_id, 4)
	return false

func completed_count() -> int:
	var count: int = 0
	for quest_id in statuses:
		if status_for(quest_id) == 4:
			count += 1
	return count

func to_dict() -> Dictionary:
	return {"statuses": statuses.duplicate()}

func from_dict(data: Dictionary) -> void:
	statuses = data.get("statuses", {}).duplicate()
