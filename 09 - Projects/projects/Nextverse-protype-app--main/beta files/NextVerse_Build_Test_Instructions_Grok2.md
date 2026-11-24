# NextVerse Build and Test Instructions

Follow these steps to build, run, and test the NextVerse MVP on your Windows machine using Visual Studio 2022 and VS Code.

## Setup Environment
- **OS**: Windows 10/11 (2025-compatible).
- **Dependencies**:
  - **Visual Studio 2022**: Ensure C++ Desktop Development workload is installed.
  - **CMake**: Install via Visual Studio or standalone (`winget install Kitware.CMake`).
  - **Boost**: Download and install (e.g., `vcpkg install boost:x64-windows`).
  - **ZeroMQ**: Install via vcpkg (`vcpkg install zeromq:x64-windows`).
  - **JUCE**: Download from [juce.com](https://juce.com), configure with CMake.
  - **Panda3D**: Already installed (ensure SDK is in PATH).
  - **Qt6**: Already installed (ensure CMake can find it).
  - **Python**: Install Flask and pytest (`pip install flask pytest`).
- **Hardware**: 1.1 GHz 4-core CPU, 8GB RAM.

## Directory Structure
```
nextverse/
├── src/
│   ├── quantum_core.hpp
│   ├── quantum_core_test.cpp
│   ├── quantum_vm.hpp
│   ├── quantum_vm_test.cpp
│   ├── mmorpg.hpp
│   ├── mmorpg_test.cpp
│   ├── daw.hpp
│   ├── daw_test.cpp
│   ├── network.hpp
│   ├── network_test.cpp
│   ├── ethical_framework.hpp
│   ├── ethical_framework_test.cpp
│   ├── main.cpp
├── python/
│   ├── ai_assistant.py
│   ├── test_ai_assistant.py
│   ├── game_engine.py
│   ├── test_game_engine.py
├── CMakeLists.txt
```

## Build with Visual Studio 2022
1. Open Visual Studio 2022.
2. Select "Open a local folder" and choose the `nextverse` directory.
3. CMake should auto-configure. If not, run:
   ```cmd
   mkdir build
   cd build
   cmake -G "Visual Studio 17 2022" -A x64 ..
   ```
4. Open the generated `NextVerse.sln` in Visual Studio.
5. Build the `nextverse` and `nextverse_tests` targets (Release mode).

## Run Tests
- **C++ Tests**:
  ```cmd
  cd build
  ctest -C Release
  ```
- **Python Tests** (in VS Code):
  ```cmd
  cd python
  pytest test_ai_assistant.py test_game_engine.py
  ```

## Run MVP
- **AI Assistant** (in VS Code):
  ```cmd
  cd python
  python ai_assistant.py
  ```
- **Game Engine** (in VS Code):
  ```cmd
  cd python
  python game_engine.py
  ```
- **Main Program** (in Visual Studio or cmd):
  ```cmd
  cd build
  .\Release\nextverse.exe
  ```
- **Test AI Endpoint** (in cmd or VS Code terminal):
  ```cmd
  curl -X POST http://localhost:5000/chat -H "Content-Type: application/json" -d "{\"message\":\"Hello\"}"
  ```

## Verify Output
- **Game Engine**: Panda3D window shows 1,000 spheres scattered in 3D space.
- **AI Assistant**: Returns `{"response":"Echo: Hello"}` for valid messages.
- **MMORPG**: Accepts network requests (test with a ZeroMQ client).
- **DAW**: Modifies audio buffer (check test results).
- **Network**: Sends messages (verify with a subscriber).
- **Ethical Framework**: Passes/fails based on input (check test logs).

## Testable Features
- **Quantum Core**: Computes power, performs Grover's search, simulates Rowen's physics (tested via Google Test).
- **Quantum VM**: Offloads tasks with measurable efficiency (tested via Google Test).
- **AI Assistant**: Handles chat requests with ethical validation (tested via pytest).
- **Game Engine**: Renders 1,000 particles in Panda3D (visually verifiable, tested via pytest).
- **MMORPG**: Simulates 10 players/NPCs via network (network test placeholder).
- **DAW**: Processes audio samples (tested via Google Test).
- **Network**: Sends test messages (network test placeholder).
- **Ethical Framework**: Enforces ethical checks (tested via Google Test).

## Limitations
- **Scale**: The MVP handles 1,000 particles and 10 players/NPCs (vs. 100,000 and 500). Scaling requires further optimization.
- **Rowen's Algorithm**: Placeholder implementation; you'll need to provide the full version.
- **Performance**: Needs benchmarking to confirm 50 FPS or <50ms audio latency on your hardware.
- **JUCE/Panda3D**: Assumes correct installation; licensing may need review.
- **Qt6**: Included in CMake but not used in MVP; ready for future UI integration. 