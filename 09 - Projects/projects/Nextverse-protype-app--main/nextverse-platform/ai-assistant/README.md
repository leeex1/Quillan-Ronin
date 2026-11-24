# NextVerse AI Assistant Module

## Overview

The AI Assistant module is a sophisticated component designed for the NextVerse platform, providing intelligent, context-aware, and ethically responsible assistance. It integrates various AI sub-systems to process user input, manage dialogue, ensure ethical compliance, and generate appropriate responses.

This module now attempts to use real NLP models for NLU tasks and a local Language Model (LLM) for response generation, with robust fallback mechanisms to simpler placeholder logic if models cannot be loaded (e.g., due to resource constraints or missing dependencies). It features a modular architecture allowing for future enhancements, including more advanced LLMs and actual quantum-enhanced processing.

## Core Components

The AI Assistant comprises several key components that work in concert:

1.  **AI Brain (`core/ai_brain.py`)**:
    *   The central orchestrator of the assistant.
    *   It receives user input and coordinates the flow between NLU, Dialogue Management, Ethical Framework, response generation logic (LLM or placeholder), and conceptual Quantum VM calls.
    *   Manages the overall state, interaction sequence, and integrates user preferences into response generation.
    *   *Current Status*: Defaults to using `distilgpt2` for response generation, but can be configured (via `utils/config.py`) to use other Hugging Face models like "microsoft/Phi-3-mini-4k-instruct". It falls back to rule-based placeholders if the selected model cannot be loaded. Makes conceptual calls to the Quantum VM interface for tasks like reasoning and ethical consistency.

2.  **Natural Language Understanding (NLU) Engine (`core/nlu.py`)**:
    *   Responsible for processing raw text input from users.
    *   Extracts key information such as user intent, entities mentioned, and sentiment of the input.
    *   *Current Status*: Attempts to load and use pre-trained Hugging Face Transformer models:
        *   **Sentiment Analysis:** `distilbert-base-uncased-finetuned-sst-2-english`
        *   **Named Entity Recognition (NER):** `dbmdz/bert-large-cased-finetuned-conll03-english`
        *   **Intent Classification:** `facebook/bart-large-mnli` (via zero-shot classification pipeline).
    *   If any model fails to load, it gracefully falls back to simpler rule-based placeholder logic for the respective NLU task.

3.  **Dialogue Management System (`core/dialogue_manager.py`)**:
    *   Maintains the state and history of conversations.
    *   Features short-term memory (in-memory list) and long-term memory using an SQLite database (`data/long_term_memory.db`) for persistent conversation logs, user preferences, and conversation embeddings.
    *   **Semantic Search:** If the `sentence-transformers` library and `all-MiniLM-L6-v2` model are available, it generates embeddings for conversation turns and can retrieve relevant past turns based on semantic similarity to a query. Otherwise, it falls back to recency-based history retrieval.
    *   Manages user preferences which can influence AI response style and content.

4.  **Ethical Framework (`core/ethical_framework.py`)**:
    *   Ensures the AI's responses and processing adhere to predefined ethical guidelines.
    *   Features expanded keyword lists for moral factors (truthfulness, harmlessness, fairness, privacy) and refined context weights.
    *   Checks for an expanded list of forbidden topics using robust regex matching.
    *   Calculates a Dynamic Ethical Safety Score (DESS) and validates content against LMCB thresholds.
    *   Includes a conceptual "ethical nudge" mechanism in `AIBrain` for borderline responses.

5.  **Quantum Virtual Machine (QVM) Interface (`core/quantum_vm_interface.py`)**:
    *   A placeholder interface for potential future integration of quantum computing capabilities.
    *   Simulates more specific conceptual quantum tasks:
        *   `quantum_semantic_disambiguation`
        *   `quantum_contextual_reasoning_step`
        *   `quantum_pattern_analysis_for_nlu`
        *   `quantum_ethical_consistency_check`
    *   *Current Status*: This is a mock interface. `AIBrain` includes conceptual calls to these tasks at relevant points in its logic.

6.  **Chat API (`api/chat_api.py`)**:
    *   Provides the external interface to the AI Assistant via Flask and Flask-SocketIO.
    *   Handles real-time, bidirectional communication, routing user messages to the AI Brain and relaying AI responses.
    *   Serves a basic HTML/CSS/JS frontend for direct interaction.

## Getting Started

Follow these instructions to set up and run the AI Assistant module locally.

**Prerequisites:**
*   Python 3.8 or higher.
*   `pip` for package management.

**Setup and Installation:**

1.  **Clone the Repository:**
    ```bash
    # If you haven't already, clone the main NextVerse platform repository
    git clone <repository_url>
    cd <repository_path>/nextverse-platform/ai-assistant
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    This includes `transformers`, `torch`, `sentencepiece`, `sentence-transformers`, `numpy`, `einops`, `accelerate`, and potentially `optimum[onnxruntime-openvino]` if you plan to use ONNX-optimized models. These libraries (`einops`, `accelerate`, `optimum`) can be beneficial or required for running certain advanced transformer models or optimized model formats.
    *   **Important Note on Model Downloads:** The system attempts to download NLP/LLM models (e.g., the configured LLM's base Hugging Face model, `all-MiniLM-L6-v2` for semantic search, and various NLU models) from Hugging Face on the first run of components that use them. ONNX models are typically local and need to be generated beforehand (see "Using ONNX Models for Optimized Inference" below).
    *   If Hugging Face model downloads fail (network, disk space), affected components fall back to placeholder logic.
    *   For full functionality, ensure internet access for initial Hugging Face model downloads and sufficient disk space.

4.  **Configuring the Language Model:**
    The AI Assistant uses a variant-based system to manage LLM configurations, controlled via `utils/config.py`:

    *   **`SELECTED_LLM_VARIANT_KEY`**: This variable in `Config` class (e.g., `Config.SELECTED_LLM_VARIANT_KEY = "phi_3_mini_variant"`) specifies which LLM variant to load.
    *   **`AVAILABLE_LLM_VARIANTS`**: This dictionary maps variant keys to detailed configurations. Each configuration includes:
        *   `description` (str): A human-readable description.
        *   `base_model_name` (str): The Hugging Face model identifier (e.g., `"microsoft/Phi-3-mini-4k-instruct"`). This is used for loading the tokenizer and as a fallback if an ONNX model isn't available or fails to load.
        *   `onnx_model_directory` (str, optional): Path (relative to the `ai-assistant` root, e.g., `"models/phi-3-mini_onnx"`) to a directory containing an ONNX-exported version of the model for optimized inference (e.g., with OpenVINO).
        *   Other keys like `reasoning_adapters_simulated_source` can be added for more complex variants.

    **Example Variant Configuration in `utils/config.py`:**
    ```python
    class Config:
        # ...
        SELECTED_LLM_VARIANT_KEY = "phi_3_mini_variant"

        AVAILABLE_LLM_VARIANTS = {
            "default_distilgpt2_variant": {
                "description": "Standard distilgpt2 model.",
                "base_model_name": "distilgpt2",
                "onnx_model_directory": "models/distilgpt2_onnx" # Optional ONNX path
            },
            "phi_3_mini_variant": {
                "description": "Standard Microsoft Phi-3 Mini 4k Instruct model, potentially ONNX optimized.",
                "base_model_name": "microsoft/Phi-3-mini-4k-instruct",
                "onnx_model_directory": "models/phi-3-mini_onnx" # Path to ONNX model files
            },
            # ... other variants ...
        }
        # ...
    ```

    The system will first attempt to load an ONNX model if `onnx_model_directory` is specified and valid for the selected variant. If that fails or is not specified, it falls back to loading the base Hugging Face model. If all loading attempts fail, it uses placeholder responses.

5.  **Using ONNX Models for Optimized Inference (Optional):**
    To potentially improve inference speed and reduce resource usage, you can convert Hugging Face models to the ONNX format and run them using ONNX Runtime, optionally with providers like OpenVINO.

    *   **Exporting a Model to ONNX:**
        A utility script `utils/export_to_onnx.py` is provided. Use it from the `ai-assistant` directory:
        ```bash
        python utils/export_to_onnx.py --model_id <huggingface_model_id> --output_path <path_to_save_onnx_model_directory> [--fp16]
        ```
        Examples:
        ```bash
        # For Phi-3 Mini (requires trust_remote_code, ensure export script handles this if needed)
        python utils/export_to_onnx.py --model_id microsoft/Phi-3-mini-4k-instruct --output_path models/phi-3-mini_onnx --fp16

        # For distilgpt2
        python utils/export_to_onnx.py --model_id distilgpt2 --output_path models/distilgpt2_onnx
        ```
        The `--output_path` should match the `onnx_model_directory` specified in `utils/config.py` for the corresponding variant. The script uses `optimum.exporters.onnx` for conversion. You might need to install `optimum` with specific providers: `pip install optimum[onnxruntime-openvino]`.
        *Note on Phi-3 export*: Models like Phi-3 might require `trust_remote_code=True` to be passed to the exporter. You may need to modify `export_to_onnx.py` to include this argument in `main_export(...)` if you encounter issues.

    *   **Running with ONNX Models:**
        If an `onnx_model_directory` is correctly configured for the `SELECTED_LLM_VARIANT_KEY` in `utils/config.py`, and the ONNX model files exist at that path, `AIBrain` will attempt to load it using `ORTModelForCausalLM` with the `OpenVINOExecutionProvider`. The original Hugging Face model ID specified in `base_model_name` is still used to load the correct tokenizer.

**Running the Application:**

1.  **Start the Server:**
    Execute the `main.py` script from the `nextverse-platform/ai-assistant/` directory:
    ```bash
    python main.py
    ```
    This will start the Flask-SocketIO server, typically on `http://0.0.0.0:5000/`. Pay attention to console logs for messages about model loading status.

2.  **Access the Chat Interface:**
    Open your web browser and navigate to `http://localhost:5000/` (or the address shown in the server logs). You should see a basic chat interface to interact with the AI Assistant.

**Running Tests:**

To ensure all components are functioning as expected (including fallback mechanisms if models don't load):
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
This command should be run from the `nextverse-platform/ai-assistant/` directory. Tests for API functionality might be skipped if Flask dependencies are unavailable due to installation issues.

## Project Structure

A brief overview of the main directories within the `ai-assistant` module:

-   `api/`: Contains the Flask-SocketIO backend (`chat_api.py`) for handling client communication.
-   `core/`: Houses the core logic of the AI Assistant: `ai_brain.py`, `nlu.py`, `dialogue_manager.py`, `ethical_framework.py`, and `quantum_vm_interface.py`.
-   `data/`: Intended for persistent data storage. The SQLite database (`long_term_memory.db`) used by the Dialogue Manager is created here.
-   `static/`: Contains static frontend assets:
    -   `css/style.css`: Stylesheets for the chat interface.
    -   `js/chat.js`: JavaScript for client-side chat functionality and SocketIO communication.
-   `templates/`: HTML templates for the web interface (e.g., `index.html`).
-   `tests/`: Contains all unit and integration tests for the module.
-   `utils/`: For utility functions and configuration.
    -   `config.py`: For managing configurations like model names, API keys (future), thresholds, etc.

## Future Enhancements

-   **Broader LLM Integration**: Continue expanding `AVAILABLE_LLM_VARIANTS` with more models and configurations, including testing various ONNX optimization strategies (quantization, different providers).
-   Fine-tuning NLU models for domain-specific intents and entities.
-   Implementation of actual quantum algorithms or more sophisticated simulations for the QVM interface tasks.
-   **Enhanced Configuration**: Further improvements to dynamic loading and configuration of models and parameters, potentially through environment variables or a more detailed configuration schema.
-   **ONNX/OpenVINO Optimizations**: Deeper integration and exploration of advanced features of ONNX Runtime and OpenVINO, such as I/O binding, quantization, and performance tuning for specific hardware.
-   More advanced error handling, logging, and monitoring.
-   Scalable deployment architecture using WSGI servers (like Gunicorn) and potentially container orchestration.

---
This README provides a comprehensive guide to understanding, running, and developing the AI Assistant module.
