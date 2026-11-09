class_name CombatTuning
extends RefCounted
## Ported from Wayward Ronin player.js / enemies.js - tune here, not in scenes.

const LIGHT_DAMAGE := 12.0
const HEAVY_DAMAGE := 28.0
const COMBO_WINDOW := 0.45      # sec to queue next hit
const PARRY_WINDOW := 0.24      # tap F within this = parry
const DODGE_IFRAMES := 0.35
const BLOCK_STAMINA_DRAIN := 15.0
const MAX_STAMINA := 100.0
const ENEMY_TYPES := ["ashigaru", "rusher", "hatamoto"]
