"""
Configuration module for the AI Assistant.

This file is intended to hold various configuration settings required by the
AI Assistant module. This includes, but is not limited to:
- Model paths for NLU, LLMs.
- API keys for external services.
- Database connection strings or paths (if not hardcoded for development).
- Thresholds for ethical framework or other AI decision-making processes.
- Feature flags.

Currently, it contains placeholder configurations. For production, sensitive
values like SECRET_KEY should be loaded from environment variables or a
secure configuration management system.
"""

import os  # Ensure os is imported for getenv


class Config:
    """
    Configuration class for the AI Assistant.

    Attributes:
        DEBUG (bool): Enables or disables debug mode.
        SECRET_KEY (str): A secret key for session management and security.
                          IMPORTANT: This should be overridden with a strong, unique
                          value for production, preferably from an environment variable.
        # Example for a future database URI:
        # DATABASE_URI = os.environ.get('DATABASE_URI') or \
        #                'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'app.db')
        # Example for LLM model path:
        # LLM_MODEL_PATH = os.environ.get('LLM_MODEL_PATH') or './models/default_llm'
    """

    # Safe defaults: debug is OFF unless explicitly enabled; no secret is embedded.
    DEBUG = os.getenv("NEXTVERSE_DEBUG", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    SECRET_KEY = os.environ.get(
        "FLASK_SECRET_KEY", ""
    )  # REQUIRED in production; never hardcode a key

    # Add other application-wide configurations as needed below.
    # For example:
    # NLU_MODEL_NAME = "distilbert-base-uncased"
    # DEFAULT_LANGUAGE = "en"
    # LOG_LEVEL = "INFO"

    SELECTED_LLM_VARIANT_KEY = "QLRC-6B_simulated"  # Renamed and updated

    # AVAILABLE_LLMS dictionary removed

    AVAILABLE_LLM_VARIANTS = {
        "default_distilgpt2_variant": {
            "description": "Standard distilgpt2 model.",
            "base_model_name": "distilgpt2",
            "onnx_model_directory": "models/distilgpt2_onnx",  # Example path
        },
        "phi_3_mini_variant": {
            "description": "Standard Microsoft Phi-3 Mini 4k Instruct model.",
            "base_model_name": "microsoft/Phi-3-mini-4k-instruct",
            "onnx_model_directory": "models/phi-3-mini_onnx",
        },
        "QLRC-6B_simulated": {
            "description": "Simulated Quantum-Lite Reasoning Core. Loads Phi-3 Mini as base, simulates DeepSeek R1 adapters with distilgpt2.",
            "base_model_name": "microsoft/Phi-3-mini-4k-instruct",
            "reasoning_adapters_simulated_source": "distilgpt2",
            "other_components_placeholders": [
                "TinyLlama-1.1B",
                "DeepSeekR1_reasoning_adapters",
            ],
        },
    }

    # Game Engine Communication Settings
    GAME_ENGINE_HOST = os.getenv("GAME_ENGINE_HOST", "localhost")
    GAME_ENGINE_COMMAND_PORT = int(os.getenv("GAME_ENGINE_COMMAND_PORT", 8081))


# For more complex configurations, consider using libraries like python-dotenv
# to load from .env files, or integrating with a configuration server.
