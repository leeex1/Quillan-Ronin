# NextVerse Alpha - Quantum Core Design

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
> **Documentation provenance.** This repo/doc is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.
<!-- QUILLAN-DOCS-PROVENANCE:END -->

## Overview

This directory contains the source code for the Quantum Core Alpha, a software simulation of a conceptual quantum computational core (QCC) and a quantum super virtual machine (QSVM). The primary purpose of this module is to simulate performance characteristics and task management related to theoretical quantum computing resources, rather than executing actual quantum algorithms.

It serves as a foundational component in the NextVerse ecosystem, providing a simulated measure of quantum processing power that higher-level services can conceptually leverage.

## Components

The Quantum Core Alpha is comprised of the following main components:

*   **`QuantumComputationalCore` (QCC)**
    *   **Role**: The QCC simulates the fundamental quantum processing unit. Its main responsibility in this Alpha version is to calculate the **JQLD (J-Quantum Algorithmic Logic Density)**. JQLD is a simulated performance metric representing the raw computational power of the core, derived from parameters like classical base performance, algorithmic optimization factors, and task efficiencies. It also includes stubs for submitting "quantum tasks" and managing "qubit resources" conceptually.
    *   **Key Files**: `qcc_alpha.hpp`, `qcc_alpha.cpp`

*   **`QuantumSuperVM` (QSVM)**
    *   **Role**: The QSVM acts as a layer above the QCC. It takes the JQLD output from the QCC and applies another simulated performance metric called **LVVM (Layered Virtualization & Viability Metric)**. LVVM models factors such as replication benefits, virtualization overheads, adaptive boosts, and synchronization costs to determine an "effective" or "amplified" performance for a VM instance. The QSVM also simulates task offloading via a threaded queue and conceptual management of multiple VM instances (e.g., the "11 Copy Principle").
    *   **Key Files**: `qsvm_alpha.hpp`, `qsvm_alpha.cpp`

*   **`quantum_common_alpha.hpp`**
    *   **Role**: This header file defines common data structures, enumerations, and utility types used across the QCC and QSVM components. This includes definitions for:
        *   `QuantumAlgorithmTypeAlpha`: Enum for different types of (simulated) quantum algorithms.
        *   `QuantumTaskAlpha`: Structure defining a simulated quantum task.
        *   `QuantumTaskStatusAlpha`: Enum for the status of a task processed by the QSVM.
        *   `QuantumTaskResultAlpha`: Structure for the result of a simulated quantum task.

## Key Formulas

The core of the simulation revolves around two key performance metrics:

*   **JQLD (J-Quantum Algorithmic Logic Density)**:
    *   Calculated by the `QuantumComputationalCore`.
    *   Conceptually represents the raw processing power of the quantum core.
    *   The detailed formula is: `Q = C * 2^(sum(N_jq * eta_task * theta_j)) * (1/omega_q)`
        *   `C`: Classical base performance.
        *   `sum(N_jq * eta_task * theta_j)`: Sum of contributions from various algorithm optimizations, efficiencies, and impact factors.
        *   `omega_q`: An overall divisor representing overheads or scaling limitations.
    *   Simpler variants are also available for different simulation needs.

*   **LVVM (Layered Virtualization & Viability Metric)**:
    *   Calculated by the `QuantumSuperVM`, using the QCC's JQLD (Q) as input.
    *   Conceptually represents the effective performance of a virtualized quantum instance, after considering VM-layer effects.
    *   The detailed formula is: `VM_eff = Q * (R_vm + psi_vm * (1 - mu_vm)) / (1 + tau_vm)`
        *   `Q`: JQLD from the QCC.
        *   `R_vm`: Replication factor.
        *   `psi_vm`: Adaptive boost factor.
        *   `mu_vm`: Virtualization overhead.
        *   `tau_vm`: Synchronization cost.
    *   Simpler variants are also available.

## Build Process

This project uses CMake for building.

To build the project (generic CMake workflow example):

1.  Ensure you have CMake and a compatible C++ compiler installed.
2.  Navigate to the root directory containing this `nextverse_alpha_core_design` directory.
3.  Create a build directory and change into it:
    ```bash
    mkdir build
    cd build
    ```
4.  Run CMake to configure the project (assuming the `CMakeLists.txt` for this component is in the parent directory relative to `build`):
    ```bash
    cmake ..
    ```
    (If the main `CMakeLists.txt` is in a different location, adjust the path accordingly.)
5.  Compile the project:
    ```bash
    make
    ```

The build process will likely produce a library or a test binary (e.g., `quantum-core-binary` or similar, depending on the `CMakeLists.txt` configuration) that would be used or linked by higher-level components in the NextVerse simulation, such as the main VM layer that orchestrates these core simulations.

## Simulation Nature

It is crucial to understand that this module is a **software simulation**. It models conceptual quantum computational characteristics and does not interact with or control any physical quantum hardware. The "quantum operations," "qubits," and "tasks" are all simulated entities and processes designed to provide performance indicators based on theoretical formulas.

## Alpha Stage

This is an **alpha version** of the Quantum Core design. Many features are simplified, represented as stubs, or are conceptual placeholders for more complex logic that might be developed in future iterations. The primary focus of this version is the calculation of the JQLD and LVVM metrics and basic task flow simulation.
