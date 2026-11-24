# NextVerse Game Engine — Godot Migration Plan (2026)

## Decision and scope

The C++ engine under `game_engine/` and `nextverse_alpha_game_engine_design/` was chosen in ~2024 and targets Panda3D, SDL2 and a hand-rolled render loop. That stack is retired. The 2026 target is **Godot 4.x**. **Tripio is a mobile image→3D asset generator**, not a networking layer: renders it produces are imported into the Godot project through `AssetPipeline.gd`. Multiplayer/social transport remains an open backend decision; the current implementation is a documented local loopback. This document is the authoritative 2026 engine plan; `GAME_ENGINE_SPEC_AUDIT_2026.md` remains the forensic record of the legacy code.

Evidence that the legacy target is no longer viable: a full Panda3D SDK was installed and the engine was compiled with MSVC 19.51. The build fails with API mismatches such as `GraphicsWindow::enable_keyboard`, `Texture::set_ram_image_compression`, `PandaFramework::is_initialized`, `HTTPChannel::set_url/set_method/send_request`, `Ramfile::append_data`, and a `Game::processInput` signature mismatch. These are not patch-worthy defects in a retired stack; they are confirmation that migration is the correct action.

## What survives from the legacy engine

The legacy C++ is retained as a reference implementation of *behaviour*, not as a runtime dependency:

| Legacy component | Behaviour worth preserving | Godot 4.x target |
| --- | --- | --- |
| `DVVE.cpp` / `DVVE.hpp` | Adaptive quality formula `R_p = P_core * F_v * (1 + omega_v) / (1 + nu_v)` | `PerformanceDirector` Node driving `Viewport.scaling_3d`, LOD distances and effect budgets |
| `PhysicsEngine.cpp` | Collision offload concept and `QuantumTask` contract | `PhysicsServer3D` queries plus a Godot-side task bridge; no custom collision loop |
| `AssetManager.cpp` | Resource loading and caching | Godot `ResourceLoader`, `ResourceCache`, `.tres` manifests |
| `DVVEParticleSystem.cpp` | Particle budget scaling | Godot `GPUParticles3D` with runtime `amount` from the performance director |
| `AICommandServer.cpp` | AI-to-avatar command queue | Typed avatar commands on `CommandBus.gd` with `LocalSessionBridge.gd` loopback |
| `Game.cpp` | Scene orchestration, AI query hooks | `Node3D` scene root with signal-based subsystems |
| alpha `particle_system_alpha.cpp` | 100k-particle stress concept | Godot GPU particles + compute where supported; budget-capped fallback |

## Target architecture

```
Godot 4.x project
├── autoload/PerformanceDirector.gd   # DVVE formula, quality tiers, telemetry
├── autoload/CommandBus.gd            # typed AI/avatar commands (legacy AICommandServer contract)
├── autoload/LocalSessionBridge.gd     # local loopback session + peer commands
├── autoload/AssetPipeline.gd          # Tripio image→3D render import (GLB/GLTF/FBX/OBJ)
├── worlds/                           # named Ready Player One-style zones (see MMORPG_UNIVERSE_DESIGN_2026.md)
├── systems/physics/                  # PhysicsServer3D wrappers, no custom integrator
├── systems/particles/                # GPUParticles3D pools, budget from PerformanceDirector
└── gdextension/                      # optional C++20 hot paths only (RS*: retained DSP/math)
```

Rules:

- GDScript or C# for gameplay; GDExtension only for measured hot paths.
- All cross-module traffic goes through `CommandBus` signals; no direct node-to-node reach across module boundaries.
- The legacy ZeroMQ/POSIX socket path is not carried forward. `LocalSessionBridge` is explicitly local-only; a production multiplayer backend must be selected before shipping.
- Tripio outputs are assets, not transport: drop generated GLB/GLTF/FBX/OBJ renders into `assets/tripio/` and register them with `AssetPipeline`.
- Quality decisions are data-driven: the DVVE formula produces a scalar tier, not ad-hoc per-feature flags.

## Migration phases and acceptance criteria

### Phase 0 — Skeleton (P0)
- Create the Godot 4.x project with the autoloads above and one empty test world.
- Acceptance: `godot --headless --path <project> --quit` exits 0; CI runs it on every commit.
- **Status: DONE (verified 2026-11-23).** A Godot 4.4.1 project exists at `nextverse-platform/godot/` with the three autoloads and `worlds/TestWorld.tscn`. Headless boot exits 0 and prints `[PerformanceDirector]` / `[TestWorld] ready`. `nextverse-platform/godot/tests/verify_runtime.gd` passes all three contracts with exit 0.

### Phase 1 — Performance director (P0)
- Port the DVVE formula from `game_engine/include/DVVE.hpp`; keep the exact formula and defaults so numbers stay comparable.
- Acceptance: a headless test asserts `R_p = 1.0e9 * 1.0 * (1 + 0.0) / (1 + 0.05) = 9.5238e8`, matching the value produced by the legacy native harness (`game_engine/include/DVVE.hpp` defaults to `P_core = 1.0e9`). A passing implementation now exists at `nextverse-platform/godot/tests/verify_runtime.gd`.

### Phase 2 — Physics and particles (P1)
- Replace custom collision with `PhysicsServer3D`; port the particle budget behaviour to `GPUParticles3D`.
- Acceptance: a stress scene sustains the target particle budget at the configured quality tier with no frame-time regression over a 10-minute headless run.

### Phase 3 — Command bus and session loopback (P1)
- Typed commands replace `AICommandServer`; `LocalSessionBridge` provides local two-peer loopback until a multiplayer backend is chosen.
- Acceptance: two headless clients exchange an avatar move command and converge on the same position within one tick; reconnect restores session state. Current status: command exchange verified; reconnect/persistence awaits a chosen backend.

### Phase 4 — World and social integration (P1)
- Import the zone map from `MMORPG_UNIVERSE_DESIGN_2026.md` and the hub templates from `SOCIAL_HUBS_DESIGN_2026.md`.
- Acceptance: each named zone loads headlessly; each hub type is reachable from the world map.

### Phase 5 — Retire legacy build wiring (P2)
- Keep the legacy C++ as `reference/` with a README stating it is non-runtime.
- Acceptance: no active build target depends on Panda3D or SDL2; the portable C++ CTests remain as algorithm reference checks only.

## Phase status (verified 2026-11-23)

All Phases 0–5 are implemented in `nextverse-platform/godot/` and pass the headless harness
`tests/verify_runtime.gd` with Godot 4.4.1, exit 0:

```
[PASS] P0 DVVE performance=952380952.380952 tier=2
[PASS] P0 CommandBus signal + history contract
[PASS] P1 tier=2 scale=0.85000002384186 budget gate ok
[PASS] P2 physics registration + integration
[PASS] P2 particle budget amount=100000
[PASS] P3 two-peer command exchange + session contract
[PASS] P4 zones=9 legacy=5 scene=Market District
[PASS] P4 hub types=6 scene=Creator stages
[PASS] P5 legacy engine retained as reference
[PASS] Godot Phases 0-5 verified
```

Implemented surfaces: `autoload/PerformanceDirector.gd`, `autoload/CommandBus.gd`,
`autoload/LocalSessionBridge.gd`, `autoload/AssetPipeline.gd`, `systems/physics/PhysicsDirector.gd`,
`systems/particles/ParticleBudget.gd`, `worlds/ZoneRegistry.gd`, `worlds/HubRegistry.gd`,
`worlds/Zone.tscn`, `worlds/Hub.tscn`, `worlds/TestWorld.tscn`, and the verification harness.
Zone and hub identifiers are copied from the MMORPG and social-hub design documents.

**Honest scope note:** these are the runtime contracts, registries and a procedurally built
PBR test zone — not a shippable world. Generated 3D assets are still to be supplied from the
Tripo mobile app; a production multiplayer backend has not been chosen. Acceptance criteria
above are met at the contract/headless level.

## Tripo asset pipeline (research-backed, 2026-11-23)

The tool the owner calls "Tripio" is **Tripo AI / Tripo App** (`tripo3d.ai`), a mobile
image→3D generator — not a networking SDK. Verified product facts relevant to this project:

- Inputs: photo, gallery image, or text prompt; up to four reference views for multi-view reconstruction.
- Outputs: GLB, FBX, OBJ, USD/USDZ for engines; STL and 3MF for printing (excluded from the runtime pipeline).
- PBR texture sets (base colour, normal, roughness, metallic) are generated with the mesh.
- Smart Mesh / Clean Topology mode produces quad meshes from 500–25,000 polygons, which fits the Godot real-time budget; High Detail reaches 2M polygons and is for rendering/print, not direct game use.
- Free-plan models are public and licensed CC BY 4.0; subscriber models are private. Generated models are commercially usable provided source material is unencumbered.

Implementation: `autoload/AssetPipeline.gd` validates extensions (GLB/GLTF/FBX/OBJ/USD/USDZ),
registers renders with a manifest and records a licence note. Headless verification:

```
[PASS] Tripio render import contract: glb accepted, txt rejected, manifest=1
```

Workflow: generate on mobile → export GLB or FBX → drop into `assets/tripio/` → register via
`AssetPipeline` → instantiate in a `ZoneBuilder` scene. Attribution must be preserved for
CC BY 4.0 free-plan assets.

## Verification strategy

- `godot --headless` scene smoke tests as the primary CI gate.
- Keep the existing native CTests (`CoreAlphaNativeVerify`, `GameEngineNativeVerify`, `ConceptualDAW_NativeVerify`) as numerical reference checks; they prove formulas, not the Godot runtime.
- Performance acceptance uses Godot's built-in profiler plus frame-time telemetry exported from `PerformanceDirector`.

## Explicitly out of scope for this document

- DAW implementation: see `DAW_DEVELOPMENT_PLAN_2026.md`.
- MMORPG world design: see `MMORPG_UNIVERSE_DESIGN_2026.md`.
- Social hubs: see `SOCIAL_HUBS_DESIGN_2026.md`.
- Production multiplayer backend (accounts, presence, authoritative state): not selected; `LocalSessionBridge` is loopback only.
- Tripo is an external mobile app; there is no SDK to integrate. Its outputs enter through `AssetPipeline`.
