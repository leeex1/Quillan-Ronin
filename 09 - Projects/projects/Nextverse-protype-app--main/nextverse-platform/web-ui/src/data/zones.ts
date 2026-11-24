// Mirrors nextverse-platform/godot/worlds/ZoneRegistry.gd and HubRegistry.gd,
// which are themselves grounded in docs/MMORPG_UNIVERSE_DESIGN_2026.md and
// docs/SOCIAL_HUBS_DESIGN_2026.md. Single source of truth for the web runtime.

export interface Zone {
  id: string;
  name: string;
  tier: string;
  existsInLegacy: boolean;
  color: number;
}

export const ZONES: Zone[] = [
  { id: 'starting_zone_town_square', name: 'Town Square', tier: 'starter', existsInLegacy: true, color: 0x00d5ff },
  { id: 'starting_zone_market', name: 'Market District', tier: 'starter', existsInLegacy: true, color: 0xffb347 },
  { id: 'starting_zone_forest_edge', name: 'Forest Edge', tier: 'starter', existsInLegacy: true, color: 0x4caf50 },
  { id: 'dungeon_level_1', name: 'Abandoned Mines - Level 1', tier: 'dungeon', existsInLegacy: true, color: 0x9b7bff },
  { id: 'unknown_zone', name: 'The Void', tier: 'special', existsInLegacy: true, color: 0x607d8b },
  { id: 'dungeon_level_2', name: 'Abandoned Mines - Level 2', tier: 'dungeon', existsInLegacy: false, color: 0x7e57c2 },
  { id: 'nexus_atrium', name: 'Nexus Atrium', tier: 'civic', existsInLegacy: false, color: 0x40c4ff },
  { id: 'guild_temple_1', name: 'Guild Temple 1', tier: 'guild', existsInLegacy: false, color: 0xff7043 },
  { id: 'live_event_arena', name: 'Live Event Arena', tier: 'event', existsInLegacy: false, color: 0xff4081 },
];

export const HUB_TYPES: string[] = [
  'Lobby',
  'Creator stages',
  'Live events',
  'Temple and guild halls',
  'DAW jam rooms',
  'AI-assisted gathering spaces',
];

export const DEFAULT_ZONE = 'starting_zone_town_square';

export function findZone(id: string): Zone {
  return ZONES.find((zone) => zone.id === id) || ZONES[0];
}
