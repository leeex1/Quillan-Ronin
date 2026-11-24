# NextVerse Formula Validation Suite

<!-- QUILLAN-DOCS-PROVENANCE:BEGIN -->
> **Documentation provenance.** This repo/doc is part of the Quillan-Ronin ecosystem (CrashOverrideX / `leeex1`). Canonical sources live in the root repository: `README.md` (**READ FIRST** — Track A harness-on-frontier vs. Track B native WSP models vs. Track B′ custom Mini) and `docs/MODEL_CARDS.md` (fleet identity, verified facts, open gaps). The large 95–100% benchmark table in the root README is a **harness-on-frontier** result, not a score of any Quillan-owned checkpoint. Creator artifacts (books, music, game demo) have primary-source links in the root README's **Creator and verifiable artifacts** section.  Four-donor design: Flagship and Mini receive all four donors (Qwen ~0.6B + Llama ~1B + BitNet 2B-4T + Falcon3 ~1B) **as a group**; Dragon is Falcon3 with an Ollama system message running Quillan's system prompt. Cross-match fix in progress. Canonical official links (site, social, repos, Ollama, HF, Grokipedia, NotebookLM, DeepWiki, Genius, YouTube/Suno) are in the root README's **Official links** directory. Historical framing is real Japanese history (Edo period, rōnin, bushidō) used as an ethical reference frame — not roleplay, game mechanics, or fantasy lore.
<!-- QUILLAN-DOCS-PROVENANCE:END -->

This application is a test suite designed to validate and benchmark the 11 Joshua Don Lee formulas,
as described in the "NextVerse Replit Setup Guide". It provides a web-based dashboard
to run tests, view results in real-time, and export validation data.

## Features

*   Calculates 11 different formulas (JQLD, LVVM, DESS, JRRN, LRPP, DVVE, JSSC, LSSS, DNNL, JHFR, LMCB).
    *   JQLD and LVVM have their specific theoretical calculations implemented.
    *   The other 9 formulas currently return dummy placeholder data.
*   Web dashboard for interactive testing:
    *   Validate all formulas at once.
    *   Run live tests with real-time progress updates via WebSockets.
    *   View individual formula results (speedup, status, execution time).
    *   See a compound "turbo effect" (overall speedup).
*   Export validation results as a JSON file.
*   Placeholder benchmarking functions for CPU and VM operations.

## Setup and Running

### Prerequisites

*   Python 3.7+
*   `pip` for installing dependencies

### Installation

1.  **Clone the repository (if applicable) or ensure you have this `nextverse_validation_suite` directory.**

2.  **Navigate to the suite's directory:**
    ```bash
    cd path/to/nextverse_validation_suite
    ```

3.  **(Optional but Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the server:**
    From within the `nextverse_validation_suite` directory, run:
    ```bash
    python main.py
    ```
    Or, for debug mode (recommended for development):
    ```bash
    python main.py --debug
    ```
    You can also specify host and port:
    ```bash
    python main.py --host 0.0.0.0 --port 5001 --debug
    ```

2.  **Access the Dashboard:**
    Open your web browser and go to `http://127.0.0.1:5000` (or the host/port you specified).

## How it Works

*   **Backend:** A Flask server (`ui/backend/app.py`) handles API requests and WebSocket communication.
*   **Frontend:** An HTML dashboard (`ui/frontend/templates/index.html`) with JavaScript interacts with the backend.
*   **Core Logic:** The `core/formula_engine.py` contains the `NextVerseFormulaEngine` class, which implements the formula calculations and (placeholder) benchmarking.
*   **Entry Point:** `main.py` is used to launch the application.

## Further Development

*   Implement the actual theoretical calculations and benchmarking logic for the 9 placeholder formulas.
*   Enhance the benchmarking functions (`_benchmark_cpu_operations`, etc.) with realistic workloads.
*   Expand test coverage (e.g., using `pytest` for unit and integration tests).
