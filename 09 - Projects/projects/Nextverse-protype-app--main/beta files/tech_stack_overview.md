# NextVerse Platform Tech Stack Overview

## Core Technologies
- **C++17/20** - Primary development language for performance-critical components
- **Python 3** - Used for AI components and high-level scripting
- **CMake** - Build system (version 3.10+)
- **Visual Studio 2022** - Primary IDE/compiler
- **MinGW** - Alternative build toolchain
- **vcpkg** - Package manager for C++ dependencies
- **WebAssembly** - For performance-critical web components
- **SQLite** - Local database for offline data storage

## Key Dependencies

1. **Core Libraries**:
   - **spdlog** (v1.13.0) - Fast C++ logging
   - **nlohmann-json** (v3.11.2) - Modern JSON handling
   - **eigen3** - Linear algebra operations
   - **glm** (v0.9.9.8) - OpenGL Mathematics for 3D operations
   - **boost** - Used for various utilities

2. **Networking (Optional)**:
   - **cpp-httplib** - HTTP client/server functionality (for optional features)
   - **WebRTC** - Peer-to-peer networking for multiplayer
   - **libcurl** - For web search functionality
   - **openssl** - Encryption and security

3. **Graphics/UI**:
   - **glfw3** - Window and input handling
   - **Qt6** - For advanced desktop UI components (already integrated)
   - **Panda3D** - 3D game engine with C++/Python integration (partially integrated)

4. **Testing/Performance**:
   - **benchmark** (v1.9.2) - Performance benchmarking
   - **gtest** - For unit tests

## Architecture Overview

### Quantum Core Layer (Phase 1)
- Implemented in C++ for maximum performance
- Python tooling for benchmarking and analysis
- Contains quantum-inspired algorithms for exponential performance gains
- Includes security sandboxing and anomaly detection
- Fully local execution with no cloud dependencies
- Optimized for single-threaded performance on modest hardware

### VM Layer (Phase 1)
- C++ abstraction over quantum computations
- Provides virtualized computing environment
- Enables cascading performance boosts through 1:1 copy scaling
- Integrates securely with the quantum core

### AI Assistant Module (Phase 1)
- Hybrid architecture:
  - C++ core components for performance-critical operations
  - Python implementation for NLU and advanced reasoning capabilities
  - Basic UI interface for testing and interaction
- Implements:
  - Advanced reasoning components (chain-of-thought, meta-cognition)
  - Dual-stream language processing (semantic & syntactic)
  - Memory systems with episodic buffers
  - Ethical directives and monitoring systems

### Game Engine Module (Phase 2)
- Custom C++ rendering pipeline
- Panda3D integration for advanced 3D capabilities
- Supports physics simulation and avatar control
- Targets high performance (50k particles, 8 high-quality objects at ≥35 FPS)

### MMORPG Module (Phase 3)
- Built on top of the Game Engine
- Provides social, interactive, and narrative components
- Includes AI-driven NPC behaviors
- Real-time narrative synthesis and world management

### DAW Module (Phase 4)
- Real-time audio processing
- AI-assisted creative production
- Spatial audio capabilities for 3D environments

### Network Module
- Secure, low-latency communication between components
- Local networking with end-to-end encryption
- Dynamic routing and optimization

## Integration Strategy
- **C++/Python Bridge**: Use native integration between C++ core components and Python AI modules
- **Performance Focus**: C++ for performance-critical paths, Python for AI reasoning
- **Qt6 for UI**: Leverage existing Qt6 integration for desktop interfaces
- **Panda3D for 3D**: Use the existing Panda3D integration for game engine features
- **Phase-Based Development**: Follow the established testing phases for component integration

## Testing Architecture
- Phase-based testing framework:
  - **Test Phase 1**: Core + VM + AI Assistant with basic UI
  - **Test Phase 2**: + Game Engine integration
  - **Test Phase 3**: + MMORPG integration
  - **Test Phase 4**: + DAW integration
  - **Test Phase 5**: Complete system integration testing

## Build System
- CMake with custom scripts for component integration
- Windows batch and PowerShell scripts for build automation
- GTest-based test suite organization

## Integration Strategy
- Leverage existing Python AI components where they're already working
- Use C++ for performance-critical components (Core, VM Layer, Game Engine)
- Keep existing component boundaries to minimize disruption
- Use Qt6 for desktop UI components where appropriate
- Integrate Panda3D for 3D visualization and game mechanics 