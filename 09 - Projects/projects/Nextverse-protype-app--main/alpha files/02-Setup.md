<!--
Consolidation: Merged and refactored from docs/SETUP.md, setup_env.ps1, setup_dependencies.bat, and relevant sections of the original README.md.
Purpose: Provides comprehensive instructions for setting up the development environment, installing prerequisites and dependencies, and building the NextVerse platform.
-->

# NextVerse Setup Guide

This guide provides detailed instructions for setting up your development environment to build and run the NextVerse platform.

## Quick Start

For most users, the fastest way to get started is:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/nextverse.git
   cd nextverse
   ```

2. **Run the automated setup script (Windows):**
   - Open a terminal as Administrator and run:
     ```cmd
     setup_build_environment.bat
     ```
   - This will install all required tools and dependencies.

3. **Verify your environment:**
   - Run:
     ```cmd
     check_environment.bat
     ```
   - Follow any instructions if tools are missing.

4. **Build the project:**
   - Use the unified build script:
     ```cmd
     build.bat
     ```
   - Or, for quantum-accelerated builds:
     ```cmd
     build_nextverse_quantum.bat
     ```

5. **Run a demo or test:**
   - Launch the platform or AI assistant:
     ```cmd
     launch_assistant.bat
     ```

For Linux/macOS, follow the manual steps in the main guide below.

---

## Platform-Specific Tips

### Windows

- Use `setup_build_environment.bat` for automated setup.
- Use `check_environment.bat` to verify all tools are installed.
- Use `build.bat` or `build_nextverse.ps1` for building.
- If you encounter issues with Visual Studio or Python, ensure they are in your PATH and installed as per the prerequisites.

### Linux

- Use the provided apt commands for dependencies.
- Use `cmake` and `make` as described in the main guide.
- For Python, always use a virtual environment (`python3 -m venv venv`).

### macOS

- Use Homebrew to install dependencies.
- Refer to the `vcpkg/scripts/azure-pipelines/osx/setup-box.sh` for advanced setup.

---

## Testing Your Setup

After building, you can verify your setup by running:

- **Frontend tests:**
  ```bash
  cd frontend/
  npm run test
  ```

- **C++/Python tests:**
  ```bash
  ctest
  # or for Python
  pytest
  ```

- **AI Assistant test:**
  ```bash
  python -c "from ai_assistant import test_assistant; test_assistant()"
  ```

If all tests pass, your environment is ready for development.

---

## Further Reading

- [16-Roadmap.md](./16-Roadmap.md): Full development workflow and roadmap.
- [18-Contributing.md](./18-Contributing.md): Coding standards and contribution guidelines.
- [01-PlatformArchitecture.md](./01-PlatformArchitecture.md): Technical and architectural overview.

---

## Tech Stack Overview

NextVerse uses a modular, multi-language tech stack for performance, security, and extensibility:

- **C++17/20**: Performance-critical modules (Quantum Core, Game Engine, DAW)
- **Python 3.8+**: AI Assistant, scripting, and rapid prototyping
- **Rust**: Security-critical networking (Network Module)
- **Node.js/ReactJS**: Frontend UI
- **Panda3D 1.10.15+**: Game engine and 3D rendering
- **TensorFlow/PyTorch**: AI/ML frameworks
- **PortAudio**: Real-time audio (DAW)
- **ZeroMQ/liboqs**: Networking and post-quantum cryptography
- **SQLite**: Local storage
- **CMake/vcpkg**: Build system and C++ dependency management
- **Docker**: (Optional) Containerization for deployment
- **GoogleTest/Pytest**: Unit testing
- **Valgrind/gprof/cProfile**: Profiling and memory analysis
- **Doxygen/Markdown**: Documentation

For a detailed breakdown by module, see the architecture doc and roadmap.

## 1. Prerequisites

### Hardware Requirements
- **CPU**: 1.1 GHz 4-core processor (minimum)
- **RAM**: 32GB system memory (64GB recommended for larger models)
- **Storage**: 20GB+ free disk space for models and dependencies
- **GPU**: NVIDIA CUDA-compatible GPU with compute capability 7.0+ (Ampere, Turing, or newer) recommended for enhanced rendering and AI acceleration

### Software Requirements

Ensure you have the following software installed:

-   **Git**: For cloning the repository. Download from [https://git-scm.com/download/](https://git-scm.com/download/).
-   **Node.js**: LTS version (18.x or newer) is required for frontend development and build tools. Download from [https://nodejs.org/](https://nodejs.org/).
-   **CMake**: Version 3.15 or higher is required for configuring and building the C++ projects. Download from [https://cmake.org/download/](https://cmake.org/download/).
-   **C++ Compiler**: A C++20 compatible compiler is required. MSVC 2022 or later is recommended on Windows. Ensure the C++ development tools are installed with Visual Studio.
-   **Python 3.8+**: Required for AI and scripting components. Download from [https://www.python.org/downloads/](https://www.python.org/downloads/). It is recommended to use a virtual environment.
-   **vcpkg**: Used for managing C++ dependencies. Follow the installation instructions at [https://github.com/microsoft/vcpkg](https://github.com/microsoft/vcpkg).
-   **Panda3D 1.10.15**: The primary game engine framework. The SDK must be installed locally. The build system expects it at `C:/development platform/Nextverse/Panda3D-1.10.15-x64` by default, but this can be adjusted in the CMake configuration.
-   **PortAudio (Optional)**: For DAW functionality. Will be used if available.
-   **CUDA**: 11.8 or newer (for GPU acceleration)
-   **Docker**: 20.10+ (for containerized deployment, optional)

### Linux/Ubuntu Setup

- **Recommended OS**: Ubuntu 20.04 or later
- **Install core dependencies**:
  ```bash
  sudo apt update
  sudo apt install git cmake g++ clang python3 python3-venv python3-pip libpanda3d-dev libzmq3-dev
  # For Rust
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  # For Docker (optional)
  sudo apt install docker.io
  ```

### Windows Setup

1. **Install Visual Studio 2022**:
   - Ensure C++ Desktop Development workload is installed
   - Include Windows SDK and C++ CMake tools

2. **Install Dependencies via Chocolatey**:
   ```powershell
   # Install Chocolatey if not already installed
   Set-ExecutionPolicy Bypass -Scope Process -Force
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
   iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

   # Install core dependencies
   choco install -y visualstudio2022buildtools visualstudio2022-workload-vctools cmake git python
   ```

## 2. Initial Repository Setup

1.  **Clone the repository**: Open a terminal or command prompt and run:
    ```bash
    git clone https://github.com/yourusername/nextverse.git
    cd nextverse
    ```
2.  **Initialize submodules**: Pull in required external libraries:
    ```bash
    git submodule update --init --recursive
    ```

## 3. Setting up the Environment

### Windows (using PowerShell)

Execute the `setup_env.ps1` script to configure the Visual Studio environment variables:

```powershell
./setup_env.ps1
```

This script will attempt to locate your Visual Studio 2022 installation and set up the necessary environment for the build.

### Setting up Dependencies

Execute the `setup_dependencies.bat` script to download and set up external C++ dependencies like GLM, GLFW, GLAD, STB, and Assimp:

```bash
./setup_dependencies.bat
```

Alternatively, you can use vcpkg to install dependencies if you have it set up:

```bash
# Ensure you are in the project root directory
vcpkg install
```

## 4. Building the Project

The NextVerse project uses CMake for its build system. A unified build script is provided for convenience.

### Using the Unified Build Script

Navigate to the project root directory in your terminal and use the `unified_build.bat` script:

```bash
scripts\unified_build.bat [target]
```

**Available Targets:**

-   `all`: Builds all components of Nextverse.
-   `quantum`: Builds Quantum components.
-   `platform`: Builds Platform components.
-   `demo`: Builds the Demo.
-   `complete`: Builds the complete Nextverse platform.
-   `network`: Builds Network components.
-   `test_phase1`: Builds Test Phase 1.
-   `clean`: Cleans build directories.

If no target is specified, the script defaults to building `all` components.

### Manual CMake Build

If you prefer to use CMake directly:

1.  **Configure the project**: Create a build directory and run CMake from the project root:
    ```bash
    cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=[path_to_vcpkg]/scripts/buildsystems/vcpkg.cmake
    ```
    Replace `[path_to_vcpkg]` with the actual path to your vcpkg installation if you used it for dependencies.
2.  **Build the project**: Navigate to the build directory and run the build command:
    ```bash
    cmake --build build --config Release
    ```
    You can replace `Release` with `Debug` for a debug build.

### CMake Build System Details

The project uses CMake for cross-platform builds and dependency management. Key configuration points:
- **C++ Standard**: C++17/20 is used for most modules. Some components (e.g., Eigen with Panda3D) may require C++14 for compatibility.
- **Panda3D Integration**: Ensure the Panda3D SDK is installed and the path is set in CMake.
- **PortAudio (Optional)**: If available, PortAudio will be linked for DAW features.
- **Dependency Linking**: All libraries are linked via CMake's `target_link_libraries`.

Example CMake configuration:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Panda3D
set(PANDA3D_DIR "C:/development platform/Nextverse/Panda3D-1.10.15-x64")
find_path(PANDA3D_INCLUDE pandaFramework.h PATHS ${PANDA3D_INCLUDE_DIR})
find_library(PANDA3D_FRAMEWORK_LIBRARY NAMES p3framework libp3framework)
# PortAudio
find_package(portaudio CONFIG QUIET)
if(portaudio_FOUND)
    set(PORTAUDIO_LIBRARIES portaudio_static)
else()
    set(PORTAUDIO_LIBRARIES "")
endif()
target_link_libraries(NextVerse PRIVATE
    ${PANDA3D_LIBRARIES}
    $<$<BOOL:${portaudio_FOUND}>:${PORTAUDIO_LIBRARIES}>
    opengl32
)
```

#### Troubleshooting
- **Header File Issues**: If Panda3D headers aren't found, verify the installation path and header files.
- **Linker Errors**: Ensure all required libraries are available at the specified paths.
- **Eigen Issues**: Use C++14 for compatibility with Eigen if needed.
- **General**: See the main contributing doc for more troubleshooting tips.

## 5. Frontend Dependencies

Navigate to the `frontend/` directory to install Node.js dependencies for the web-based UI:

```bash
cd frontend/
npm install
```

## 6. Verification and Development

After building, you can verify your setup and begin development:

-   `npm run typecheck` (in `frontend/`) - Check frontend code style
-   `npm run test` (in `frontend/`) - Run frontend tests
-   `npm run lint` (in `frontend/`) - Check frontend code style
-   `npm run lint:fix` (in `frontend/`) - Fix frontend code style issues
-   `npm run dev` (in `frontend/`) - Start the frontend development server

Refer to the documentation for specific modules for instructions on running tests and applications for each component (e.g., running the QuantumCore Benchmark or the Quantum-Enhanced Avatar Test).

## 7. Troubleshooting

If you encounter issues during setup:

-   Verify Node.js and npm installations: `node --version` and `npm --version`.
-   Clear npm cache: `npm cache clean --force`.
-   Delete `node_modules` and reinstall frontend dependencies: `rm -rf node_modules && npm install` (in `frontend/`).
-   Ensure Visual Studio C++ tools are installed and that `setup_env.ps1` runs correctly.
-   Check CMake and vcpkg configurations.

## 8. Coding Guidelines

- **C++**: Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
- **Python**: Adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/).
- **Rust**: Use [Rust Style Guidelines](https://github.com/rust-dev-tools/fmt-rfcs).
- **Document** all functions and classes with clear comments/docstrings.
- **Error Handling**: Implement robust error handling and logging.
- **Performance**: Prioritize low-latency algorithms and modular code structure.
- **Ethical Safeguards**: Integrate ECC and LMCB in AI code for compliance.
- **Version Control**: Use Git with a feature branch workflow; commit frequently with descriptive messages; conduct code reviews.
- **Testing**: Write unit tests (GoogleTest for C++, Pytest for Python); implement integration tests for module interactions; test edge cases and error conditions.

For detailed coding standards, see the main contributing doc (`docs/18-Contributing.md`). For module-specific build and test instructions, see the roadmap (`docs/16-Roadmap.md`).

## 8. AI Module Setup

After the main setup, configure the AI module:

1. **Create AI Environment**:
   ```bash
   cd NextVerseAI
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install AI Dependencies**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Install Quantum Dependencies**:
   ```bash
   pip install qiskit qiskit-aer qiskit-ibm-runtime
   pip install pennylane pennylane-qiskit
   pip install qulacs
   ```

4. **Verify AI Installation**:
   ```bash
   python -c "import quantum_boost; import nextverse_vm; print('Dependencies verified!')"
   ```

## 9. Extended Troubleshooting

In addition to the basic troubleshooting steps, here are solutions for common issues:

### CMake Issues
- **Panda3D Not Found**: Verify the installation path in CMakeLists.txt matches your system
- **Eigen Compatibility**: Some components may require C++14 for Eigen compatibility
- **vcpkg Integration**: Ensure vcpkg toolchain file is properly referenced

### Build Issues
- **Linker Errors**: Check that all required libraries are available at specified paths
- **Compilation Errors**: Verify C++ standard compatibility across all components
- **Python Integration**: Ensure Python paths and virtual environments are correctly configured

### Runtime Issues
- **Missing DLLs**: Add library paths to system PATH or copy DLLs to executable directory
- **GPU Acceleration**: Verify CUDA installation and GPU compatibility
- **Audio Issues**: Check PortAudio configuration and audio device permissions

For more detailed troubleshooting, refer to the main contributing documentation. 