## 📄 **File 1: Game Design Document (GDD) - "Chronicles of the Elemental Bond"**

```markdown
# CHRONICLES OF THE ELEMENTAL BOND
## Game Design Document (GDD)
**Version:** 1.0 | **Genre:** Hybrid Tactical-Creature RPG | **Target Platform:** PC

---

## 1. CORE VISION

### 1.1 Elevator Pitch
A creature-collecting RPG where players don't just command monsters—they fight *alongside* them in dynamic ATB combat. Weapon mastery defines your class, elemental synergies reshape the battlefield, and every system from skills to progression is modular and player-driven.

### 1.2 Key Pillars
- **Symbiotic Combat**: You and your creatures share the battlefield, turns, and destiny.
- **Weapon-Driven Identity**: Your equipped weapon *is* your class; skills are unlocked through mastery, not menus.
- **True Customization**: Mix-and-match summons, creature abilities, and weapon tech trees like Suikoden/Warframe loadouts.
- **Living Mastery**: Skills level through use (Golden Sun meets Skyrim).

---

## 2. COMBAT SYSTEM

### 2.1 ATB Foundation (FFVII-Inspired)
- **Active Time Battle**: Time gauge fills based on Speed stat; actions pause for target selection.
- **Dual Timeline**: Player character and creature companion have *separate* ATB gauges.
- **Synchro Gauge**: When both gauges are full, execute "Bond Actions" (powerful combo attacks).

### 2.2 Creature System (Pokémon DNA)
- **Minion Roles**: Tank, DPS, Support, Controller (each with 4 ability slots).
- **Loyalty System**: Affects crit rate and synchro speed; built through battle performance, not friendship minigames.
- **Elemental Essence**: Every creature has Primary + Secondary element (e.g., Fire/Earth = Magma typing).

### 2.3 Summons (FFVII Grand Scale)
- **Not creatures**—summons are *world-altering spells*:
  - **Ifrit-CLASS**: Scorch the battlefield, adding Burn DOT to all non-Fire units.
  - **Shiva-CLASS**: Freeze water tiles into walkable ice bridges.
  - **Ramuh-CLASS**: Lightning chains between metal weapons, damaging wielders.
- **Summon Meter**: Charges when dealing weakness damage; no MP cost.

---

## 3. PROGRESSION & CUSTOMIZATION

### 3.1 Weapon Mastery System
| Weapon Type | Primary Role | Skill Unlocks Via Use | Max Mastery Level |
|-------------|--------------|------------------------|-------------------|
| **Greatsword** | Vanguard | Heavy Slash → Armor Break → Earth Shaker | Lv. 100 |
| **Dual Daggers** | Assassin | Swift Cut → Poison Edge → Wind Walk | Lv. 100 |
| **Staff** | Elementalist | Mana Bolt → Elemental Surge → Prism Beam | Lv. 100 |
| **Bow** | Marksman | Piercing Shot → Elemental Arrow → Sky Driller | Lv. 100 |
| **Gauntlets** | Monk | Focus Strike → Chi Wave → Comet Fist | Lv. 100 |

- **Class Fluidity**: Switch weapons mid-battle (1-turn penalty); skills from previous weapon are *grayed out* but retain XP.
- **Cross Mastery**: Using Fire spells while wielding a Sword unlocks "Flame Blade" passive.

### 3.2 Skill Leveling (Usage-Based)
- **No Generic XP**: Each skill has independent progression.
- **Example**: "Cleave" (Greatsword)
  - Use 10x → Rank 2 (+10% damage)
  - Use 50x → Rank 3 (+15% damage, -1 AP cost)
  - Use 200x → Mastery (adds Earth element, can trigger Quake)

### 3.3 Customization Grid (Warframe-Style)
- **Modular Slots**: Weapon (6 slots) | Creature (4 slots) | Summon (3 slots)
- **Examples**:
  - **Ice Gem** on Sword: Attacks gain Ice element, chance to Freeze.
  - **Phoenix Feather** on Creature: Revive once per battle at 50% HP.
  - **Quickening Rune** on Summon: -25% summon charge time.

---

## 4. WORLD & EXPLORATION

### 4.1 Elemental Field System (Golden Sun Psynergy)
- **Environmental Puzzles**: Use creature abilities outside combat.
  - **Geo-Shift**: Earth creatures raise pillars for platforming.
  - **Cryo-Lock**: Ice creatures freeze water to create paths.
  - **Gale Rush**: Wind creatures clear fog/reveal secrets.
- **No Separate "Field Moves"**: If creature knows "Magma Spire" in battle, it works in the field.

### 4.2 Dungeon Design Philosophy
- **Metroidvania Lite**: New elemental abilities backtrack to new areas.
- **Dynamic Events**: Day/night cycles shift enemy spawns and puzzle solutions.

---

## 5. STORY FRAMEWORK

### 5.1 Premise
You are a "Anchor"—a human who can bind with primordial creatures called "Eidokin." The world of **Aetheria** is fracturing as the **Elemental Pillars** collapse. You must forge bonds, not just to save the world, but to prevent your own essence from being consumed by the Void.

### 5.2 Key Factions
- **The Concordat**: Humans who enslave Eidokin (antagonists).
- **The Unbound**: Rogue Anchors who reject the system.
- **The Primordials**: Sentient elementals who see humans as parasites.

---

## 6. MULTIPLAYER & ENDGAME

### 6.1 Co-op Raids
- **2-Player Sync**: Share a 6-unit party (2 humans + 4 creatures).
- **Elemental Synergy Bonus**: If both players wield Fire weapons, unlock "Inferno Protocol" raid mechanic.

### 6.2 Endgame Loop
- **Procedural Torment Dungeons**: Roguelike mode; weapon mastery caps removed.
- **Transmutation Forge**: Sacrifice max-level weapons to create hybrid types (e.g., Bowblade).
- **Eidokin Breeding**: Combine two creatures; offspring inherits movesets from both.

---

## 7. TARGET SPECS

- **Engine:** Custom (C++17/OpenGL or Vulkan)
- **Resolution:** 1920x1080 (target), 4K support
- **Framerate:** 60 FPS locked in combat
- **Save System:** 3 auto-save slots + manual saves
```

---

## 📄 **File 2: Technical Design Document (TDD) - Engine Architecture**

```markdown
# CHRONICLES OF THE ELEMENTAL BOND
## Technical Design Document (TDD)
**Engine Codename:** AetherForge | **Language:** C++17 | **Renderer:** Vulkan (primary) / OpenGL (fallback)

---

## 1. ENGINE CORE ARCHITECTURE

### 1.1 Modular System Design (Warframe-Style Customization)
```cpp
// Core: Entity-Component-System (ECS) using EnTT library
struct WeaponComponent {
    std::string archetype; // "Greatsword", "Bow"
    uint32_t masteryLevel;
    std::vector<SkillID> unlockedSkills;
    std::array<ModSlot, 6> modSlots; // Warframe-style mods
};

struct CreatureComponent {
    Element primaryElement;
    Element secondaryElement;
    std::array<AbilityID, 4> abilityBar;
    float loyalty; // 0.0 to 1.0
};

struct SummonComponent {
    std::string summonClass; // "Ifrit", "Shiva"
    float chargeMeter;
    std::array<ModSlot, 3> modSlots;
};
```

### 1.2 Scripting & Data-Driven Design
- **Lua (sol3 bindings)**: All abilities, summons, and creature AI.
- **JSON Schema**: Weapon definitions, mastery curves, elemental reactions.
- **Hot Reload**: .lua and .json changes apply without recompile in debug mode.

---

## 2. COMBAT SYSTEM IMPLEMENTATION

### 2.1 ATB Timeline Manager
```cpp
class ATBManager {
    std::priority_queue< BattleUnit, std::vector<BattleUnit>, TurnComparator > turnQueue;
    
    void Update(float deltaTime) {
        for (auto& unit : activeUnits) {
            unit.atbGauge += (unit.stats.speed * deltaTime);
            if (unit.atbGauge >= MAX_GAUGE) {
                turnQueue.push(unit);
                unit.atbGauge = 0.0f;
            }
        }
    }
};
```
- **Tick Rate:** 60 ticks/second for gauge precision.
- **Pause Flags:** Player input, summon cinematics, bond actions.

### 2.2 Elemental Reaction System (Dynamic)
```cpp
enum ElementalReaction {
    MELT = (FIRE | ICE),
    VAPORIZE = (FIRE | WATER),
    CRYSTALLIZE = (EARTH | (ICE | WATER)),
    OVERLOAD = (LIGHTNING | (FIRE | EARTH))
};

// Applied on hit:
if (target.elements & attack.element) {
    DamageMultiplier *= 0.5; // Resist
} else if (IsWeakness(target.elements, attack.element)) {
    DamageMultiplier *= 2.0;
    summonMeter += 10.0f; // Charge summon on weakness
}
```

---

## 3. PROGRESSION & SAVE SYSTEM

### 3.1 Usage-Based Skill Mastery
```cpp
struct SkillInstance {
    SkillID id;
    uint32_t useCount;
    uint32_t rank; // 1-5 (Mastery)
    
    void OnUse() {
        useCount++;
        if (useCount >= rankThresholds[rank]) {
            rank++;
            ApplyRankBonus();
        }
    }
};
```
- **Storage:** Binary save (fast) + JSON backup (debug).
- **Mastery Thresholds:** Exponential curve (10, 50, 200, 500, 1000 uses).

### 3.2 Save File Structure
```
saves/
├── profile.bin          // Player name, options
├── world_state.bin      // Story flags, pillar status
├── inventory.bin        // Weapons, mods, items
├── creatures/           // One file per creature
│   ├── eidokin_001.bin
│   └── eidokin_002.bin
└── mastery_db.sqlite    // Skill usage counts (SQL for queries)
```

---

## 4. MODULAR CUSTOMIZATION ENGINE

### 4.1 Mod Slot System (Warframe-Style)
```cpp
struct ModSlot {
    bool isPolarized; // Reduces mod cost if matched
    Mod* installedMod;
    
    bool CanInstall(Mod* mod) {
        return (mod->capacity <= availableCapacity) && 
               (mod->validTargets & Target::WEAPON);
    }
};
```
- **Mod Capacity:** Determined by weapon mastery level.
- **Riven Mods** (Endgame): Procedurally generated mods with random stats.

### 4.2 Creature Fusion (Pokémon Breeding + Warframe Helminth)
```cpp
// Combine two creatures:
Creature Breed(Creature& parentA, Creature& parentB) {
    Creature child;
    child.primaryElement = parentA.primaryElement;
    child.secondaryElement = parentB.primaryElement;
    child.abilityBar = {
        parentA.abilityBar[0],
        parentB.abilityBar[1],
        InheritRandom(parentA.abilityBar),
        InheritRandom(parentB.abilityBar)
    };
    return child;
}
```

---

## 5. RENDERING & VISUALS

### 5.1 Vulkan Pipeline (High Performance)
- **Pipelines:**
  - `SKY_PIPELINE`: Weather, day/night.
  - `BATTLE_PIPELINE`: Particle-heavy, separate render pass for UI.
  - `FIELD_PIPELINE`: Terrain tessellation for elemental puzzles.
- **Instancing:** Up to 100 creatures on-screen (field), 8 units in battle.

### 5.2 Particle System (Elemental Effects)
```cpp
// Summon effects: GPU-driven particles
class ElementalParticleSystem {
    vk::Buffer particleBuffer;
    ComputeShader* reactionShader; // Updates 1M particles/frame
    
    void TriggerSummon(SummonID id) {
        LoadPreset(id);
        reactionShader->Dispatch(1024, 1, 1); // GPU parallel
    }
};
```

---

## 6. NETWORKING (CO-OP RAIDS)

### 6.1 Lockstep Simulation
- **Protocol:** UDP with custom reliability layer.
- **Tick Rate:** 30 Hz simulation, 60 Hz interpolation.
- **Input Delay:** 3-frame buffer to prevent desync.

### 6.2 State Synchronization
```cpp
// Only sync key events:
struct NetworkPacket {
    uint32_t frame;
    PacketType type; // ACTION, SUMMON, CREATURE_DEATH
    uint32_t unitID;
    uint32_t targetID;
    float param; // Damage, heal value
};
```
- **Anti-Cheat:** Server-authoritative skill usage validation (replay system).

---

## 7. TOOLCHAIN & WORKFLOW

### 7.1 In-House Tools (Qt6-based)
- **AetherEditor:** JSON/Lua editor with live preview.
- **CreatureForge:** Visual creature stat/ability designer.
- **MasteryTuner:** Graph editor for weapon XP curves.

### 7.2 Build Pipeline
```bash
# CMake 3.20+ targets:
cmake --build . --target AetherForge-Editor  # Dev tools
cmake --build . --target CoEB-Client         # Game
cmake --build . --target Headless-Server     # Dedicated raid server
```

---

## 8. PERFORMANCE BUDGETS

| System | CPU Budget | Memory Budget | GPU Budget |
|--------|------------|---------------|------------|
| **Combat** | 5ms/frame | 512 MB | 8ms (16.6ms total) |
| **Field** | 8ms/frame | 256 MB | 12ms |
| **UI** | 1ms/frame | 64 MB | 2ms |
| **Networking** | 2ms/frame | 128 MB | N/A |

---

## 9. THIRD-PARTY LIBRARIES

| Library | Purpose | License |
|---------|---------|---------|
| **EnTT** | ECS framework | MIT |
| **sol3** | Lua scripting | MIT |
| **GLFW** | Window/input | zlib |
| **VulkanSDK** | Rendering | Apache 2.0 |
| **SQLite3** | Mastery DB | Public Domain |
| **Dear ImGui** | Dev UI | MIT |

---

## 10. MILESTONE ROADMAP

- **MVP (6 months):** ATB combat + 2 weapons + 10 creatures.
- **Alpha (12 months):** 5 weapons, 30 creatures, 1 raid boss.
- **Beta (18 months):** Full story, 8 weapons, 100 creatures, mod system.
- **Release (24 months):** Polish, netcode, endgame loop.

---

## 11. ASSET PIPELINE SPECS

- **2D Sprites:** 2048x2048 atlas, 16 frames/animation.
- **3D Models (if used):** glTF 2.0, max 50k tris/unit.
- **Audio:** OGG Vorbis, 48kHz stereo.
- **Localization:** UTF-8 JSON, Google Sheets → JSON exporter.
```

---

### 🚀 **Next Steps:**

1. **Validate Core Loop**: Prototype ATB + 1 creature + 1 weapon in Python/Pygame first.
2. **ECS Foundation**: Set up EnTT + basic rendering window.
3. **Design One Weapon**: Flesh out Greatsword mastery tree completely as a vertical slice.
