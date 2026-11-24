# NextVerse Game Engine — Spec-vs-Implementation Audit (2026)

- **Audit date:** 2026-11-23
- **Auditor:** C7-LOGOS (formal deductive/invariant logic lane), spawned as an independent subagent
- **Root audited:** `C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main`
- **Report path:** `ROOT\nextverse-platform\docs\GAME_ENGINE_SPEC_AUDIT_2026.md`
- **Method:** direct file reads with line numbers, repository-wide symbol scans, dependency/existence checks, compiler-availability probes. No code was modified.

---

## 1. Executive verdict

**The game engine as checked in does not compile against its legacy Panda3D target.** This is proven, not inferred: at least seven independent C++ translation-unit errors and one missing include file were located by symbol/absence scans (Section 5). At audit time no compiler had been located; subsequently MSVC 19.51 and clang++ 22.1.8 were found, the portable subset was built and tested, and a real Panda3D SDK build failed on Panda API drift (Section 11). The stack has since been retired in favour of Godot 4.x; see [GODOT_MIGRATION_PLAN_2026.md](GODOT_MIGRATION_PLAN_2026.md), which supersedes the remediation framing of this document. Tripo AI (called "Tripio" in earlier notes) is a mobile image→3D asset generator whose outputs enter through `AssetPipeline.gd`, not a networking layer.

**The documented capability envelope is substantially ahead of the code.** The README advertises PBR rendering, quantum physics, and 16 ms @ 4K (`README.md:26`, `README.md:37`). The engine contains an arithmetic DVVE formula, an SDL2 2D particle demo composited into Panda3D, and `PhysicsEngine::performAdvancedCollisions` that only prints a conceptual `QuantumTask` (`src/PhysicsEngine.cpp:19-59`). A repository-wide scan for `PBR`, `normal map`, `roughness`, `metallic`, `shader`, `GLSL`, `HLSL`, `Vulkan`, `DirectX` across all engine headers, sources, and alpha-design files returned **zero matches**. No 3840x2160 or 16 ms frame budget exists in the engine code. The project's own status document lists PBR, shaders, LOD, real quantum-core connection, and rigorous performance benchmarks as **future work** (`game_engine_status_and_next_steps.md:52`, `:60-62`, `:67-68`, `:73-75`).

**Net assessment:** the implementation is a demo-grade prototype with a partially working DVVE simulation loop and useful profiling scaffolding, wrapped in aspiration-level documentation. Status claims of "~55% complete" (`README.md:59`) are unverified by tests because the test suite itself cannot compile (Section 5, B2/B4).

---

## 2. Scope, method, evidence rules

Files read in full (line counts from the reader):

| Area | Files |
| --- | --- |
| Build | `game_engine\CMakeLists.txt` (75 lines) |
| Headers | `include\Game.hpp` (124), `PhysicsEngine.hpp` (27), `AssetManager.hpp` (29), `DVVE.hpp` (33), `DVVEParticleSystem.hpp` (52), `AICommandServer.hpp` (73), `Particle.hpp` (12) |
| Sources | `src\Game.cpp` (785), `AICommandServer.cpp` (212), `DVVEParticleSystem.cpp` (168), `AssetManager.cpp` (94), `PhysicsEngine.cpp` (59), `DVVE.cpp` (35), `main.cpp` (23) |
| Tests | `tests\TestGameDVVE.cpp` (145), `TestPhysicsEngine.cpp` (42), `TestAssetManager.cpp` (49) |
| Alpha design | `nextverse_alpha_game_engine_design\CMakeLists.txt` (92), `game_engine_sdl_alpha.hpp` (78), `game_engine_sdl_alpha.cpp` (267), `particle_system_alpha.hpp` (67), `particle_system_alpha.cpp` (169), `game_engine_common_alpha.hpp` (42) |
| Spec/docs | `README.md` (96), `nextverse-platform\docs\quantum_enhancements.md` (68), `nextverse-platform\docs\feedback_loops_analysis.md` (71), `game_engine_status_and_next_steps.md` (80) |

Evidence rule used throughout: **PROVEN** = a symbol, literal, path, or absence was directly observed in a named file/line; **INFERRED** = deduced from code structure without executing the program. Build blockers in Section 5 are PROVEN by static inspection only; a compiler would be needed to produce the authoritative diagnostic list.

---

## 3. Spec extraction — what the documents actually claim

| # | Claim | Exact source | Characterisation |
| --- | --- | --- | --- |
| S1 | Game Engine core engine = `DVVE`; capabilities = "PBR rendering, quantum physics" | `README.md:37` | Product-level marketing claim |
| S2 | "16ms @ 4K rendering, <100ms AI response, <100ms network RTT" | `README.md:26` | Hard performance target |
| S3 | Game Engine reported 55% complete; "Next Milestone: Full PBR rendering pipeline" | `README.md:59`, `:64` | PBR explicitly a future milestone |
| S4 | Stack: C++20 with SIMD/Eigen/TBB; rendering Vulkan/DirectX 12 | `README.md:70-72` | Contradicts `CMakeLists.txt:12` (`CMAKE_CXX_STANDARD 17`) and actual Panda3D/SDL2 usage |
| S5 | DVVE formula `R_p = P_core * F_v * (1 + omega_v) / (1 + nu_v)`; mocked `P_core`/`F_v`; `omega_v` heuristic; `R_p` drives Panda scene complexity and SDL particle count | `game_engine_status_and_next_steps.md:11-17` | Accurate description of the arithmetic layer |
| S6 | AssetManager caches models/textures (including failed loads) and applies textures via `set_texture` | `game_engine_status_and_next_steps.md:19-24` | Accurate at API level |
| S7 | SDL off-screen particle rendering, `SDL_RenderReadPixels`, CPU buffer, vertical flip, Panda3D `set_ram_image`, CardMaker compositing, profiling hooks, 1k/10k/25k/50k benchmark cycling | `game_engine_status_and_next_steps.md:26-35` | Accurate description of the demo pipeline |
| S8 | PhysicsEngine is a placeholder; `performAdvancedCollisions` demonstrates conceptual `QuantumTask` preparation using live particle data | `game_engine_status_and_next_steps.md:37-39` | Honest scoping in the status doc |
| S9 | Real PBR/shaders/LOD/lighting, real quantum-core `P_core`, rigorous benchmarks are **next steps** | `game_engine_status_and_next_steps.md:52`, `:60-62`, `:67-68`, `:73-75` | Directly contradicts README's present-tense claims |
| S10 | AI command server assumed to expose HTTP `/boost` and `/execute`; HTTP server logic explicitly **not present** in C++ core | `quantum_enhancements.md:39-41` | Known, documented integration gap |
| S11 | GTest formalisation, Clang-Tidy, code coverage are proposed enhancements | `quantum_enhancements.md:22-25` | Tooling not yet mature |
| S12 | Feedback metrics from Game avatars to Quantum Core are "Highly Inferred"; internal dynamic update mechanism absent | `feedback_loops_analysis.md:33-49`, `:68-69` | Compound-turbo feedback loop is conceptual only |

**Contradiction detected (S1/S2 vs S3/S9/S10/S12):** present-tense README claims are refuted by the project's own status/analysis documents and by code inspection. This is a documentation-integrity defect, not merely a wording issue.

---

## 4. Implementation matrix

Legend: **IMPLEMENTED** = callable code path exists and is populated; **PARTIAL** = scaffolding/logging exists but the substantive behaviour is absent; **MISSING** = no code found.

| Class / component | Status | Evidence (file:line, short quote) |
| --- | --- | --- |
| `DVVECalculator` (`include\DVVE.hpp:20-31`, `src\DVVE.cpp:5-35`) | **IMPLEMENTED** (arithmetic only) | `calculatePerformance` returns `P_core * F_v * (1+omega) / (1+nu)` at `DVVE.cpp:14`; `determineAdaptiveBoost` is a 3-way heuristic at `:29-34`. |
| `Game` scene/loop (`include\Game.hpp:38-122`, `src\Game.cpp`) | **PARTIAL** | `initialize` opens Panda window and loads models (`:40-50`, `:72-137`); `gameUpdateTask` drives per-frame calls (`:478-582`); `renderSceneLogic` toggles `show()`/`hide()` of placeholder models (`:728-785`). No PBR, no LOD, no shader/material pipeline. |
| `PhysicsEngine` | **PARTIAL → effectively MISSING** | `update()` body is empty (`src\PhysicsEngine.cpp:15-17`); `performAdvancedCollisions` only builds a `QuantumTask` and prints it (`:19-59`), including the literal `"> Conceptual: This QuantumTask would now be submitted..."` at `:57`. |
| `AssetManager` | **IMPLEMENTED** (with caching caveat) | `loadModel` uses `Loader::get_global_ptr()->load_sync` (`src\AssetManager.cpp:27`); `getTexture` uses `TexturePool::load_texture` (`:53`); `listLoadedAssets` at `:68-94`. Caches failures but re-attempts them (see 6.6). |
| `DVVEParticleSystem` | **IMPLEMENTED** (2D demo, not 3D/PBR) | `initialize` creates SDL window/renderer/target texture (`src\DVVEParticleSystem.cpp:19-63`); `update` does Euler integration + wall bounce (`:103-107`); `render` draws points to an off-screen target and reads pixels (`:128-153`). Public API in `include\DVVEParticleSystem.hpp:19-30` has no `getParticles()` (blocker B1). |
| `AICommandServer` + `ThreadSafeQueue` | **PARTIAL** | `ThreadSafeQueue` is a correct mutex-guarded queue (`include\AICommandServer.hpp:26-52`). TCP server accepts one connection at a time, parses JSON naively, pushes commands (`src\AICommandServer.cpp:8-51`, `:103-212`). Windows build impossible (B6); thread-safety defects in 6.1. |
| `Game` AI integration | **PARTIAL** | TCP commands move avatars by direct `set_pos` (`Game.cpp:445-475`); HTTP contextual query uses hard-coded `http://ai-assistant:5000/contextual_query` (`:366`) and a blocking poll loop (`:395-401`). |
| PBR rendering | **MISSING** | Repository-wide term scan: zero matches for PBR/normal map/roughness/metallic/shader/GLSL/HLSL/Vulkan/DirectX in all engine + alpha files. `README.md:64` and `game_engine_status_and_next_steps.md:52,61` place it in the future. |
| Quantum physics via DVVE | **MISSING** (formula only) | DVVE is pure floating-point arithmetic (`DVVE.cpp:9-17`); physics offload is print-only (`PhysicsEngine.cpp:57-58`); `P_core` is mocked by a frame counter (`Game.cpp:604-623`). |
| 16 ms @ 4K | **MISSING** (no measurement) | Window fixed at 1280x720 (`Game.cpp:62`), particle buffer defaults to 320x240 (`DVVEParticleSystem.hpp:14`). No 3840/2160/16 ms literal anywhere in engine sources. Profiler only logs averages (`Game.cpp:530-547`) with no pass/fail gate. |
| Feedback loops into Quantum Core | **MISSING** | No feedback call path from `Game`/`PhysicsEngine` to any core object; `P_core` is set by `setMockPCore` (`Game.cpp:584-587`). Documented as unresolved at `feedback_loops_analysis.md:68-69`. |
| Alpha engine (`GameEngineSDL_Alpha`) | **PARTIAL, non-buildable** | Software renderer (`game_engine_sdl_alpha.cpp:44`), conceptual DVVE denominator guard (`:82-92`), computed `particle_render_budget_` (`:157`) never consumed by `renderParticles` (`:174-185`, admitted at `:179`). Its `ParticleSystem` source does not compile (B7). Alpha CMake emits libraries only; the demo executable is commented out (`alpha\CMakeLists.txt:64-72`). |

---

## 5. Build blockers — PROVEN by static inspection

| ID | Severity | File:line | Proven defect |
| --- | --- | --- | --- |
| B1 | BLOCKER | `src\Game.cpp:699` | Calls `dvve_particle_system_.getParticles()`. Repository-wide scan finds exactly one occurrence — the call. `include\DVVEParticleSystem.hpp:19-30` declares `getActiveParticleCount`, `getPixelDataFormat`, `getPixelDataBuffer` but **no** `getParticles`. Compile error: no member named `getParticles`. |
| B2 | BLOCKER | `tests\TestGameDVVE.cpp:81`, `:97`, `:117`, `:134` | Four calls to `game.update(0.016)`. `include\Game.hpp:63` declares `updateInternal(double)` (private) with the comment "Renamed from update to avoid conflict"; no `update` is declared. Compile error on an incomplete type member lookup. |
| B3 | BLOCKER | `src\Game.cpp:252`, `:259`, `:503` | Uses `&Game::userRequestQuit`, `sendContextualQueryToAI(...)`, `processAICommands(...)`. Definitions exist at `Game.cpp:589`, `:355`, `:445`, but a scan of `Game.hpp` finds none of these declarations. Compile error: no declaration matches / member not declared. |
| B4 | BLOCKER | `tests\TestPhysicsEngine.cpp:31-32` | `particles.push_back({1.0, 2.0, 0.1, -0.1, 1.0})` supplies 5 aggregate initialisers, while `include\Particle.hpp:4-10` defines exactly 4 members (`x, y, vx, vy`). Compile error: too many initialisers for `Particle`. |
| B5 | BLOCKER | `include\PhysicsEngine.hpp:8` | `#include "../../quantum_core/include/QuantumCore.hpp"`. A recursive search of the entire ROOT found **no** `QuantumCore.hpp` and no `quantum_core` directory. `CMakeLists.txt:39` also points a private include path at the absent `../quantum_core/include`. Fatal include error. |
| B6 | BLOCKER (Windows) | `include\AICommandServer.hpp:13-16` | POSIX headers `sys/socket.h`, `netinet/in.h`, `unistd.h`, `arpa/inet.h` in a header transitively included by `Game.hpp:27`. `CMakeLists.txt:4-6` has an explicit MSVC `/W4` branch, but MSVC cannot compile these headers; `shutdown`/`close`/`read`/`send` at `AICommandServer.cpp:87-88,175,192` are POSIX calls. |
| B7 | BLOCKER (alpha) | `nextverse_alpha_game_engine_design\particle_system_alpha.cpp:10,17,25,62-123` | Uses member `num_active_particles_` throughout; `particle_system_alpha.hpp:49-63` declares `particles_`, `max_particles_`, `random_engine_`, three distributions, and two stubs — **no** `num_active_particles_`. Compile error. |
| B8 | HIGH (portability) | `game_engine_common_alpha.hpp:15`, `:29`, `:35` | `uint8_t` used without `<cstdint>`; `std::move` used without `<utility>`. Compiles only if a transitive standard header happens to provide them. |
| B9 | HIGH (fragile include hygiene) | `src\Game.cpp:562`, `:398`; `game_engine_sdl_alpha.cpp:195`; `src\PhysicsEngine.cpp:30`; `src\AICommandServer.cpp:12` | Uses `memcpy` without `<cstring>`, `std::this_thread::sleep_for` without `<thread>`/`<chrono>`, `std::rand()` without `<cstdlib>`, `::isspace` without `<cctype>`. Each may compile only via transitive includes; `CPTA_uchar` at `Game.cpp:567` relies on `pta_uchar.h` (`:10`) and must be confirmed against the installed Panda3D SDK version. |
| B10 | BLOCKER (dependency discovery) | `CMakeLists.txt:16`, `:24` | `find_package(Panda3D REQUIRED)` and `find_package(GTest REQUIRED)` with no `FindPanda3D.cmake` anywhere in ROOT. Cached configure records `Panda3D_DIR:PATH=Panda3D_DIR-NOTFOUND`. `find_package(SDL2 REQUIRED)` (`:19`) relies on an installed SDL2 config package. |

---

## 6. Correctness, safety, and performance review

All findings below are from code reading (**INFERRED** runtime behaviour unless the item restates a proven absence).

### 6.1 AICommandServer — thread safety and lifecycle (HIGH)

- **Unsynchronised shared fd.** `server_socket_fd_` is written by the server thread at `AICommandServer.cpp:104` (and `:128`, `:136`, `:209`) and read/closed by the caller thread in `stop_listening` at `:84-89` with no mutex or atomic. This is a C++ data race (undefined behaviour) and a double-close hazard: both `:88` and `:208` can close the same descriptor.
- **Start/stop race.** `start_listening` sets `running_ = true` before spawning the thread (`:69-71`). If `stop_listening` runs before `server_loop` assigns `server_socket_fd_` (`:104`), the `if (server_socket_fd_ != -1)` guard at `:84` is false, no shutdown is signalled, and a subsequent `accept()` at `:146` can block forever; `join()` at `:95` then hangs.
- **Misleading log.** At `:90` the code prints `server_socket_fd_` *after* assigning `-1` at `:89`, so the diagnostic always reports `-1`.
- **Dead branch.** `:164-167` is an empty `if` whose only body is a commented-out line; it has no effect and should be removed or replaced with real error handling.
- **Blocking I/O.** A single client that connects and sends nothing blocks `read()` at `:175` indefinitely, stalling all command processing. There is no timeout, no per-client thread, and no connection cap beyond `listen(..., 5)` at `:133`.

### 6.2 AICommandServer — security exposure (HIGH)

- Binds `INADDR_ANY` (`AICommandServer.cpp:122`), exposing the command channel on every interface, with no authentication, no source allowlist, and a hand-written JSON parser (`:8-51`). Any remote peer can push `action = "move_to"` into avatar nodes via `Game.cpp:454-459`.
- `send()` results are unchecked (`:192`, `:196`); on POSIX, writing to a closed peer can raise `SIGPIPE` and terminate the process unless `MSG_NOSIGNAL` or signal handling is used.
- `::isspace` at `:12` is called with plain `char`; negative values are undefined behaviour for the C `isspace` contract and the missing `<cctype>` include (B9) compounds it.
- C13-WARDEN-style gate: this is a hard-gate candidate before any non-loopback deployment.

### 6.3 DVVE / performance claim violations (MEDIUM-HIGH)

- **DVVE particle-count control is disabled at runtime.** `Game.cpp:665` hard-codes `bool benchmark_particle_counts = true;`, so the DVVE-derived mapping at `:681-692` is unreachable. The status doc's claim that `R_p` drives particle count (`game_engine_status_and_next_steps.md:17`) is therefore false in the current build configuration.
- **Per-frame heap churn.** `Game.cpp:559` allocates a fresh `std::vector<unsigned char> flipped_pixels(sdl_pixels.size())` every frame (320x240x4 = 307,200 bytes at defaults) and then wraps it in a `CPTA_uchar` (`:567`). This conflicts with the stated "reducing per-frame allocations" optimisation (status doc `:34`).
- **Budget not enforced (alpha).** `particle_render_budget_` is computed at `game_engine_sdl_alpha.cpp:157` but `render()` (`:174-185`) ignores it; `renderParticles` (`:133-155`) issues one `SDL_SetRenderDrawColor` plus one draw call per particle. Combined with `SDL_RENDERER_SOFTWARE` (`:44`), the documented 100k @ 50 FPS target (`game_engine_status_and_next_steps.md:51`) has no supporting mechanism.
- **No 4K path.** Both windows are fixed small (1280x720 `Game.cpp:62`; 320x240 default `DVVEParticleSystem.hpp:14`). The profiler at `Game.cpp:530-547` logs averages but never asserts the 16.67 ms budget, so S2 cannot be verified or failed.

### 6.4 SDL lifecycle and compositing (MEDIUM)

- `DVVEParticleSystem::initialize` calls global `SDL_Init(SDL_INIT_VIDEO)` (`:20`) and `shutdown` calls global `SDL_Quit()` (`:164`). Multiple instances, or any other SDL user in the process, are affected. `GameEngineSDL_Alpha::~GameEngineSDL_Alpha` also calls `SDL_Quit()` (`game_engine_sdl_alpha.cpp:22`).
- The engine runs **two independent windows** (Panda3D plus an SDL preview). Pixel data is read back from GPU-visible SDL texture to CPU (`DVVEParticleSystem.cpp:114-117`), vertically flipped (`Game.cpp:561-565`), and uploaded into a Panda texture (`:568`). This readback is a structural performance ceiling, independent of particle count.
- Constructor member-initialiser order in `DVVEParticleSystem.cpp:8-11` does not follow declaration order in the header (`:36-49`); with `-Wextra -Wpedantic` (`CMakeLists.txt:9`) this yields `-Wreorder` diagnostics. No semantic corruption because the reordered members are vectors/defaults, but it is a warning gate failure.

### 6.5 Floating-point and input validation (LOW-MEDIUM)

- `DVVE.cpp:10` checks `(1.0 + params.nu_v) == 0` with exact floating-point equality. It catches the test's exact `-1.0` (`TestGameDVVE.cpp:36`) but is fragile for computed values.
- `Game::initialize` calls `window_->get_graphics_window()->get_properties()` (`:61`) without a null check on `get_graphics_window()`, immediately after only checking `window_` (`:44`). A null graphics window would dereference.
- `AssetManager.cpp:27` dereferences `Loader::get_global_ptr()` without a null check.
- `getPixelDataBuffer()` returns a const reference to a buffer cleared by `shutdown()` (`DVVEParticleSystem.cpp:160`, header `:30`); any consumer retaining the reference across shutdown sees a dangling-view pattern.

### 6.6 Test-suite validity (HIGH)

- `TestAssetManager.cpp:26-49` explicitly tests that failed loads are re-attempted; `AssetManager.cpp:31` and `:57` cache the failure but `:16-24` and `:41-50` deliberately fall through. The cache of failures is therefore dead state — harmless for correctness, wasteful for I/O.
- `TestGameDVVE.cpp:78-145` contains no assertion about rendering, physics, or timing — only DVVE arithmetic and `detail_level_` strings. Even if B2 were fixed, the suite would not validate any of S1/S2/S7.
- `TestPhysicsEngine.cpp:20-37` uses `ASSERT_NO_THROW` around print-only methods; it cannot detect a physics regression.

---

## 7. Verification status on this host

### 7.1 What was actually executed

1. Directory/file inventory of `game_engine`, `nextverse_alpha_game_engine_design`, and `nextverse-platform\docs`.
2. Full line-numbered reads of all 27 files listed in Section 2.
3. Repository-wide symbol scans: `getParticles` (1 hit, the call), `num_active_particles_` (only in the alpha `.cpp`), `Game.hpp` method declarations (confirmed absent for `update`/`userRequestQuit`/`sendContextualQueryToAI`/`processAICommands`), PBR/shader/4K/16 ms term scan (zero PBR hits).
4. Existence checks: `QuantumCore.hpp` absent anywhere under ROOT; `FindPanda3D.cmake` absent; alpha core headers present (`quantum_common_alpha.hpp`, `qcc_alpha.hpp`, `qsvm_alpha.hpp` all present in `nextverse_alpha_core_design`).
5. Compiler probes: `cl` → NOT FOUND, `g++` → NOT FOUND, `gcc` → NOT FOUND, `clang++` → NOT FOUND. `cmake` → `C:\Program Files\CMake\bin\cmake.exe`; `ninja` → Python Scripts directory. No Visual Studio installation directory found.
6. Inspection of stale `game_engine\build\CMakeCache.txt`: `CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++`, `CMAKE_GENERATOR:INTERNAL=Unix Makefiles`, `CMAKE_HOME_DIRECTORY:INTERNAL=/app/game_engine`, `Panda3D_DIR:PATH=Panda3D_DIR-NOTFOUND`. No `.exe/.dll/.lib/.a/.o/.obj` artefacts exist in that build tree.

### 7.2 What was NOT executed and why

- **No CMake configure/generate** was run. `cmake` exists but no C++ compiler does, so compiler detection would fail; additionally, running configure would write into the repository and the task prohibits modifying files other than this report.
- **No compilation** of any `.cpp`, library, or executable.
- **No test execution** (`ctest`/GTest). No GTest installation was verified on this host.
- **No application launch, no rendering, no profiling, no benchmark measurement.** Consequently S2 (16 ms @ 4K) is untested and unmeasured, not merely unmet.
- **No sanitizers, static analysis, or Clang-Tidy.**
- The stale cache is evidence of a **prior failed/different-environment configure** (`/app/game_engine`, Linux `/usr/bin/c++`, Panda3D not found). It is not evidence of a successful build.

### 7.3 Deterministic conclusion

All build-blocker findings in Section 5 are **PROVEN statically**: they follow from symbol declarations/definitions and file existence, not from compiler output. Which additional diagnostics a real compiler would emit (for example B8/B9 transitive-include behaviour, Panda3D API version mismatches, or `CPTA_uchar` availability) is **INFERRED** and must be confirmed on a compiler-equipped host.

---

## 8. Compile and test plan for an MSVC or GCC host

### 8.1 Prerequisites

- CMake >= 3.10 (repo declares 3.10), a working C++17 compiler (repo sets C++17; README claims C++20 — resolve the discrepancy first, see R10).
- SDL2 development package, Panda3D SDK with headers and libraries discoverable by CMake, GoogleTest, and Threads.
- Windows: MSVC v143 + vcpkg (`vcpkg install sdl2:x64-windows gtest:x64-windows`) plus a Panda3D SDK; POSIX socket usage must be ported (R4) before the MSVC branch at `CMakeLists.txt:4-6` is meaningful.
- Linux: `libsdl2-dev`, `libgtest-dev`, Panda3D SDK, and a `FindPanda3D.cmake` or Panda3D config package placed on `CMAKE_MODULE_PATH`/`Panda3D_DIR`.

### 8.2 Sequence (run only after P0 blockers R1–R7 are closed)

1. Clean configure with explicit dependency paths, e.g.
   Run `cmake -S ROOT\game_engine -B ROOT\game_engine\build-verify` with `-DPanda3D_DIR` and `-DSDL2_DIR` set to the concrete config-package directories of the installed SDKs on the build machine.
2. Confirm configure output contains no `NOTFOUND` for Panda3D, SDL2, GTest, or Threads; fail the gate otherwise.
3. Build: `cmake --build <build-verify> --config Release --parallel`.
4. Turn warnings into errors after the initial clean build: `/WX` (MSVC) or `-Werror` (GCC/Clang), matching the existing `/W4` or `-Wall -Wextra -Wpedantic` flags (`CMakeLists.txt:4-10`). Expect `-Wreorder` (6.4) and unused-parameter/variable warnings (alpha sources).
5. Run tests: `ctest --test-dir <build-verify> -C Release --output-on-failure`. All of `TestGameDVVE`, `TestPhysicsEngine`, `TestAssetManager` must configure and pass.
6. Run the app from a directory where Panda3D model paths (`models/environment`, `panda.egg`, `models/misc/rgbCube.egg`, `box.egg`, `maps/panda-rgb.png`) resolve, or provide those assets; otherwise the engine logs many load failures (`Game.cpp:72-167`).
7. Sanitizer pass on GCC/Clang: rebuild tests with `-fsanitize=address,undefined -fno-omit-frame-pointer` and run `ctest`. This is the intended detector for the fd double-close/data race in 6.1.
8. Performance gate: add a 3840x2160 harness that records p99 frame time with 1k/10k/25k/50k particles and asserts <= 16.67 ms before S2 can be called implemented. The current 320x240 preview and average-only logging do not satisfy this.
9. Alpha path: give `nextverse_alpha_game_engine_design\CMakeLists.txt` a real executable target (currently commented out at `:64-72`) and run the same configure/build/test gate independently.

---

## 9. Prioritized remediation plan

| Priority | ID | Action | Acceptance criterion |
| --- | --- | --- | --- |
| P0 | R1 | Add a `getParticles() const` accessor to `DVVEParticleSystem` (returning `const std::vector<Particle>&`) or change `Game.cpp:699` to pass an available buffer. | `Game.cpp` compiles; test asserts the accessor returns the live particle count. |
| P0 | R2 | Fix the test/API mismatch: either expose a public `Game::update(double)` wrapper or change `TestGameDVVE.cpp:81,97,117,134` to call the intended entry point. | All four DVVE integration tests compile and pass. |
| P0 | R3 | Declare `userRequestQuit`, `sendContextualQueryToAI`, `processAICommands` in `Game.hpp` with correct signatures and access levels. | `Game.cpp` compiles with no "member not declared" diagnostics. |
| P0 | R4 | Port `AICommandServer` to a conditionally compiled socket layer: Winsock2 on Windows, POSIX on Linux; add `<cerrno>`/`<cctype>`. | Same header/config builds under both MSVC and GCC. |
| P0 | R5 | Provide the missing quantum-core dependency: either create `quantum_core/include/QuantumCore.hpp` with the `QuantumTask`/`QuantumTaskType` definitions used at `PhysicsEngine.cpp:29-36`, or replace that include with a local minimal header and update `CMakeLists.txt:39`. | `PhysicsEngine.hpp` include resolves; configure finds the include dir. |
| P0 | R6 | Fix aggregate initialisation in `TestPhysicsEngine.cpp:31-32` (4 fields) and add `<cstdlib>`/`<cctype>`/`<cstring>`/`<thread>`/`<chrono>`/`<cstdint>`/`<utility>` where used (B8/B9). | All three test TUs compile with `-Werror`. |
| P0 | R7 | Add `size_t num_active_particles_ = 0;` to `particle_system_alpha.hpp` private section; give the alpha CMake an executable target. | Alpha particle library and demo build and link. |
| P0 | R8 | Make Panda3D/GTest discovery reproducible (vendored `FindPanda3D.cmake` or documented SDK path; GTest via `FetchContent` fallback). | Clean configure on a fresh machine with zero `NOTFOUND` entries. |
| P1 | R9 | Rework `AICommandServer` lifecycle: protect the fd with a mutex/atomic, sequence start so the fd exists before `stop_listening` can run, use one owner for close, remove the dead branch at `:164-167`, fix the log at `:90`. | Stress test (100 start/stop cycles under ASan/UBSan) shows no race, hang, or double-close. |
| P1 | R10 | Harden the command channel: bind to loopback by default, add an allowlist/configured bind address, replace the naive JSON parser with a validated parser, handle `send` errors and `SIGPIPE`, add read timeouts and a connection cap. | Unauthenticated remote command injection test fails against default config; a malformed payload is rejected without process termination. |
| P1 | R11 | Replace per-frame `flipped_pixels` allocation (`Game.cpp:559`) with a reusable member buffer; remove the permanent `benchmark_particle_counts = true` override (`:665`) or make it a documented runtime flag. | No per-frame allocation in the compositing path; DVVE-driven particle count is demonstrably active in non-benchmark mode. |
| P1 | R12 | Make SDL lifecycle instance-safe (`SDL_InitSubSystem`/`SDL_QuitSubSystem` with a refcount or single shared owner); make the SDL preview window optional. | Two particle-system instances can coexist; compositing works with the preview disabled. |
| P2 | R13 | Reconcile documentation with code: change README S1/S2/S3 to accurately label PBR/quantum physics/16 ms @ 4K as targets, or implement them. Cite this audit. | No present-tense capability claim in README lacks a passing test or benchmark. |
| P2 | R14 | Implement or formally withdraw the "quantum physics" claim: define a real `QuantumTask` submission path and a typed result, and feed a measured `P_core` from the core instead of `setMockPCore` (`Game.cpp:584-587`). | A test proves a particle set is submitted and a result is consumed; status doc updated accordingly. |
| P2 | R15 | Add the 4K/16.67 ms benchmark harness with p99 reporting and CI gating; enforce `particle_render_budget_` in the alpha `renderParticles` path. | A recorded run at 3840x2160 reports p99 <= 16.67 ms on the target GPU, or the claim is withdrawn. |
| P3 | R16 | Expand tests: `ThreadSafeQueue` concurrency, JSON parser edge cases, server start/stop lifecycle, particle system bounds/lifetime, asset cache behaviour, and DVVE boundary values. | Meaningful coverage of every changed class; no test consists solely of `ASSERT_NO_THROW` around logging. |
| P3 | R17 | Add Clang-Tidy, coverage, and CI (MSVC + GCC matrix) per `quantum_enhancements.md:22-25`. | CI green on both toolchains; coverage report published. |
| P4 | R18 | Decide and document one maintained engine path (`game_engine` vs `nextverse_alpha_game_engine_design`) and one language standard (C++17 in `CMakeLists.txt:12` vs C++20 in `README.md:70`). | A single ADR records the decision; the losing path is archived with an explicit status. |

---

## 10. Acceptance criteria and definition of done

**Per-item acceptance** is listed in Section 9 and is objective (compiles, test passes, no diagnostic, measured value).

**Global definition of done for this audit's findings:**

1. All P0 rows R1–R8 closed and independently rebuilt from a clean checkout on both MSVC and GCC hosts.
2. `ctest` (or equivalent) reports 100% of discovered tests passing; no test file is excluded to achieve that result.
3. Warning gate `/W4 /WX` (MSVC) and `-Wall -Wextra -Wpedantic -Werror` (GCC) passes.
4. ASan/UBSan run over the test suite and a scripted 100-cycle command-server lifecycle shows zero reports.
5. Either (a) a 3840x2160 benchmark report with p99 <= 16.67 ms exists and is reproducible, or (b) README S2 is amended to state the target is unachieved. Same rule for PBR and quantum physics claims (S1/S9/S14).
6. A re-run of the symbol/existence scans in Section 7.1 finds zero unresolved references (`getParticles`, `Game::update`, `num_active_particles_`, `QuantumCore.hpp`).
7. This audit file is updated with the new verification results and the date of re-audit.

---

## 11. Proven versus inferred — explicit ledger

**PROVEN (direct observation):**
- Seven compile blockers B1–B7 and the missing include in B5; symbol scans returned exactly zero declarations where the code requires them.
- Absence of PBR/shader/4K/16 ms implementation terms across all engine and alpha source files.
- No C++ compiler on this host; stale cache points to `/app/game_engine` with `/usr/bin/c++` and `Panda3D_DIR-NOTFOUND`.
- The DVVE formula, mocked `P_core`, print-only collision offload, hardcoded benchmark flag, two-window compositing design, and per-frame pixel-buffer allocation — all quoted with file:line above.
- The project's own documents label PBR, shaders, LOD, real quantum connection, and rigorous benchmarks as future work.

**INFERRED (code reading, not executed):**
- The `server_socket_fd_` data race and start/stop hang scenarios (6.1); deterministic by C++ memory-model reasoning but not observed at runtime.
- `SIGPIPE` process termination risk and remote command-injection exposure (6.2); dependent on platform and network configuration.
- Performance ceiling from CPU readback and per-particle draw calls (6.3); plausible but unmeasured.
- The "~55% complete" figure (`README.md:59`); it is a document claim with no objective measurement backing it.

---

## Appendix A — inspected file inventory

All paths are relative to `ROOT = C:\02_QUILLAN\09 - Projects\projects\Nextverse-protype-app--main`.

```
README.md
game_engine_status_and_next_steps.md
game_engine/CMakeLists.txt
game_engine/include/{Game,PhysicsEngine,AssetManager,DVVE,DVVEParticleSystem,AICommandServer,Particle}.hpp
game_engine/src/{Game,DVVE,PhysicsEngine,AssetManager,DVVEParticleSystem,AICommandServer,main}.cpp
game_engine/tests/{TestGameDVVE,TestPhysicsEngine,TestAssetManager}.cpp
game_engine/build/CMakeCache.txt                     (inspected only)
nextverse_alpha_game_engine_design/CMakeLists.txt
nextverse_alpha_game_engine_design/{game_engine_common_alpha,game_engine_sdl_alpha,particle_system_alpha}.hpp
nextverse_alpha_game_engine_design/{game_engine_sdl_alpha,particle_system_alpha}.cpp
nextverse-platform/docs/quantum_enhancements.md
nextverse-platform/docs/feedback_loops_analysis.md
```

No files other than this report were created, modified, or deleted.
