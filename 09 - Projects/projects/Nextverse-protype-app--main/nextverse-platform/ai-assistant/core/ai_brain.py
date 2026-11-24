"""
Core AI Brain for the NextVerse AI Assistant.

This module orchestrates the AI's response generation process by coordinating
various sub-modules: Natural Language Understanding (NLU), Dialogue Management,
Ethical Framework, a (placeholder) local Language Model (LLM), and a (placeholder)
Quantum VM Interface.
"""

from .nlu import (
    NLUEngine,
    TRANSFORMERS_AVAILABLE,
    pipeline as nlu_pipeline_func,
)  # Import TRANSFORMERS_AVAILABLE flag and pipeline
from .dialogue_manager import (
    DialogueManager,
    DEFAULT_DATABASE_PATH as DM_DEFAULT_DB_PATH,
)
from .ethical_framework import EthicalFramework
from .quantum_vm_interface import QuantumVMInterface
from .game_engine_client import GameEngineClient  # Added for Game Engine communication
from utils.config import (
    Config,
)  # Import Config (absolute: 'core' is a top-level package in this layout)
from transformers import AutoTokenizer  # Added for ONNX

try:  # ONNX is an optional acceleration backend, not a hard requirement
    from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import-not-found]

    ONNX_RUNTIME_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the host's optional extras
    ONNX_RUNTIME_AVAILABLE = False

    class ORTModelForCausalLM:  # type: ignore[no-redef]
        """Fallback shim: keeps ONNX code paths importable and patchable
        when ``optimum[onnxruntime]`` is absent or incompatible."""

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("ONNX runtime backend (optimum) is not available.")

        def generate(self, *args, **kwargs):  # pragma: no cover - interface parity
            raise RuntimeError("ONNX runtime backend (optimum) is not available.")


import datetime
import os
import logging

logger = logging.getLogger(__name__)

# LLM_MODEL_NAME = "distilgpt2" # Will be replaced by dynamic loading from Config


class AIBrain:
    """
    The central processing unit of the AI Assistant.

    The AIBrain integrates various components like NLU, dialogue management,
    ethical checks, (simulated) quantum processing, and a local LLM
    (if available) to generate contextually aware and ethically compliant responses.

    Attributes:
        nlu_engine (NLUEngine): The NLU processing unit.
        dialogue_manager (DialogueManager): Manages conversation history and user preferences.
        ethical_framework (EthicalFramework): Enforces ethical guidelines.
        quantum_vm (QuantumVMInterface): Placeholder for quantum-enhanced processing.
        llm_pipeline (transformers.Pipeline | None): Text generation pipeline for a local LLM.
                                                    Set to None if model loading fails.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initializes the AIBrain and its components, including attempting to load a local LLM.

        Args:
            db_path (str | None, optional): Path to the SQLite database for the
                DialogueManager. If None, uses the default path defined in DialogueManager.
                Defaults to None.
        """
        if db_path is None:
            db_path = DM_DEFAULT_DB_PATH

        self.nlu_engine = NLUEngine()
        self.dialogue_manager = DialogueManager(db_path=db_path)
        self.ethical_framework = EthicalFramework()
        self.quantum_vm = QuantumVMInterface()
        self.llm_pipeline = None
        self.current_llm_variant_config = None  # Initialize attribute

        # Initialize GameEngineClient
        try:
            self.game_engine_client = GameEngineClient(
                host=Config.GAME_ENGINE_HOST, port=Config.GAME_ENGINE_COMMAND_PORT
            )
            logger.info(
                f"GameEngineClient initialized for AIBrain, target: {Config.GAME_ENGINE_HOST}:{Config.GAME_ENGINE_COMMAND_PORT}"
            )
        except AttributeError as e:
            logger.error(
                f"Failed to initialize GameEngineClient. Config attributes GAME_ENGINE_HOST or GAME_ENGINE_COMMAND_PORT might be missing. Error: {e}"
            )
            self.game_engine_client = None  # Ensure it's None if setup fails
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during GameEngineClient initialization: {e}"
            )
            self.game_engine_client = None

        # Offline-first guard (NextVerse is local-first by design). Setting
        # NEXTVERSE_AI_OFFLINE=1 skips every model download/load path and keeps
        # the brain on deterministic placeholder responses. This makes the
        # module usable on air-gapped hosts and unit-testable without network.
        offline_mode = os.environ.get("NEXTVERSE_AI_OFFLINE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        if TRANSFORMERS_AVAILABLE and not offline_mode:
            selected_variant_key = Config.SELECTED_LLM_VARIANT_KEY
            available_variants = Config.AVAILABLE_LLM_VARIANTS

            current_variant_config = None
            model_identifier_to_load = None

            # Attempt to load the selected variant
            if selected_variant_key in available_variants:
                variant_candidate = available_variants[selected_variant_key]
                if (
                    isinstance(variant_candidate, dict)
                    and "base_model_name" in variant_candidate
                ):
                    current_variant_config = variant_candidate
                    logger.info(
                        f"Selected LLM Variant: '{selected_variant_key}'. Description: {current_variant_config.get('description', 'N/A')}"
                    )
                else:
                    logger.warning(
                        f"Invalid configuration for selected LLM variant '{selected_variant_key}'. "
                        f"Missing 'base_model_name' or not a dictionary."
                    )
            else:
                logger.warning(
                    f"Selected LLM variant key '{selected_variant_key}' not found in AVAILABLE_LLM_VARIANTS."
                )

            # Fallback 1: Try default_distilgpt2_variant
            if not current_variant_config:
                default_fallback_key = "default_distilgpt2_variant"
                logger.warning(
                    f"Attempting to fall back to default variant: '{default_fallback_key}'."
                )
                if default_fallback_key in available_variants:
                    variant_candidate = available_variants[default_fallback_key]
                    if (
                        isinstance(variant_candidate, dict)
                        and "base_model_name" in variant_candidate
                    ):
                        current_variant_config = variant_candidate
                        selected_variant_key = default_fallback_key  # Update selected_variant_key for logging
                        logger.info(
                            f"Using fallback LLM Variant: '{selected_variant_key}'. Description: {current_variant_config.get('description', 'N/A')}"
                        )
                    else:
                        logger.warning(
                            f"Invalid configuration for fallback LLM variant '{default_fallback_key}'. "
                            f"Missing 'base_model_name' or not a dictionary."
                        )
                else:
                    logger.warning(
                        f"Default fallback variant key '{default_fallback_key}' not found in AVAILABLE_LLM_VARIANTS."
                    )

            # Fallback 2: Ultimate fallback to hardcoded "distilgpt2"
            if not current_variant_config:
                logger.error(
                    "All attempts to load a configured LLM variant failed. "
                    "Using hardcoded 'distilgpt2' as a last resort."
                )
                current_variant_config = {
                    "description": "Failsafe hardcoded distilgpt2",
                    "base_model_name": "distilgpt2",
                }
                selected_variant_key = "failsafe_distilgpt2"  # For logging purposes

            self.current_llm_variant_config = current_variant_config
            model_identifier_to_load = self.current_llm_variant_config[
                "base_model_name"
            ]  # This is the HF ID

            # Attempt to load ONNX model if specified and available
            onnx_model_dir_config = self.current_llm_variant_config.get(
                "onnx_model_directory"
            )
            onnx_path_to_load = None

            if onnx_model_dir_config:
                # Construct path relative to ai-assistant root. __file__ is core/ai_brain.py
                base_app_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..")
                )
                # Config paths may use forward slashes on Windows; normalize so
                # the path handed to the runtime (and to callers/tests) uses the
                # platform's canonical separators and has no redundant segments.
                onnx_path_to_load = os.path.normpath(
                    os.path.join(base_app_path, onnx_model_dir_config)
                )
                logger.info(f"ONNX model path specified in config: {onnx_path_to_load}")

                if os.path.exists(onnx_path_to_load):
                    try:
                        logger.info(
                            f"Attempting to load ONNX model from: {onnx_path_to_load} with OpenVINO provider."
                        )
                        # Tokenizer must be loaded from the original Hugging Face model ID
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_identifier_to_load
                        )

                        self.llm_pipeline = ORTModelForCausalLM.from_pretrained(
                            onnx_path_to_load,
                            provider="OpenVINOExecutionProvider",
                            # use_io_binding=True, # Can explore later for performance
                        )
                        self.llm_pipeline.tokenizer = (
                            tokenizer  # Attach tokenizer to the ONNX model object
                        )
                        logger.info(
                            f"ONNX model loaded successfully from {onnx_path_to_load} using OpenVINOExecutionProvider."
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to load ONNX model from {onnx_path_to_load}. Error: {e}. Falling back to HuggingFace model.",
                            exc_info=True,
                        )
                        self.llm_pipeline = None  # Ensure pipeline is None so fallback to HF loader occurs
                else:
                    logger.warning(
                        f"ONNX model directory {onnx_path_to_load} not found. Falling back to HuggingFace model."
                    )
                    self.llm_pipeline = None  # Fallback to HF loader

            # Fallback to HuggingFace model loading if ONNX is not specified, not found, or failed to load
            if not self.llm_pipeline:
                logger.info(
                    f"Attempting to load HuggingFace model for text generation: '{model_identifier_to_load}' (from variant '{selected_variant_key}')"
                )
                try:
                    # Need to ensure tokenizer is loaded for HF pipeline as well, nlu_pipeline_func usually handles it.
                    self.llm_pipeline = nlu_pipeline_func(
                        "text-generation", model=model_identifier_to_load
                    )
                    if self.llm_pipeline is not None:
                        logger.info(
                            f"HuggingFace model '{model_identifier_to_load}' loaded successfully into pipeline."
                        )
                    else:
                        logger.warning(
                            f"HuggingFace pipeline creation returned None for '{model_identifier_to_load}'. Fallback to placeholder responses."
                        )
                        self.llm_pipeline = None
                except Exception as e:
                    logger.error(
                        f"Failed to load HuggingFace model '{model_identifier_to_load}'. Error: {e}. AIBrain will use placeholder responses.",
                        exc_info=True,
                    )
                    self.llm_pipeline = None
        else:
            logger.warning(
                "LLM loading disabled (offline mode or Transformers unavailable). "
                "AIBrain will use placeholder responses."
            )
            # Ensure current_llm_variant_config is at least minimally set if the LLM is off
            if not self.current_llm_variant_config:
                self.current_llm_variant_config = {
                    "description": "No LLM loaded (Transformers unavailable)",
                    "base_model_name": "None",
                }

        logger.info(
            f"AIBrain initialized. DialogueManager is using DB at: {self.dialogue_manager.db_path}"
        )
        logger.info(
            f"Current LLM Variant Config in use: {self.current_llm_variant_config}"
        )

    def _control_ai_avatar_placeholder(
        self, avatar_id: str, target_x: float, target_y: float, target_z: float
    ):
        """
        Placeholder method to send a move_to command to the game engine for an AI-controlled avatar.
        """
        command = {
            "avatar_id": avatar_id,
            "action": "move_to",
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
        }
        if self.game_engine_client:
            logger.info(f"AIBrain: Sending command to Game Engine: {command}")
            success = self.game_engine_client.send_command(command)
            logger.info(f"AIBrain: Game Engine command send status: {success}")
        else:
            logger.warning(
                "AIBrain: GameEngineClient not initialized. Cannot send command."
            )

    def process_game_context_query(
        self, event_type: str, object_id: str, player_location: dict = None
    ) -> str:
        """
        Processes a contextual query from the game engine.
        For MVP, returns a predefined response based on event_type and object_id.
        """
        logger.info(
            f"AIBrain: Processing game context query: event_type='{event_type}', object_id='{object_id}', location={player_location}"
        )

        # MVP: Simple predefined responses
        if event_type == "inspect_object":
            if object_id == "crystal_A":
                return "This is the Crystal of Alacrity. It hums with a faint energy."
            elif object_id == "ancient_tablet_1":
                return "The tablet is covered in unreadable glyphs, but you feel a sense of importance."
            else:
                return f"You inspect {object_id}. It seems ordinary for now."
        elif (
            event_type == "character_interaction_hint"
        ):  # Example of another event type
            if object_id == "npc_wizard_01":
                return "The wizard seems preoccupied. Perhaps mentioning the ancient runes would get his attention."
            else:
                return f"Approaching {object_id}. They look busy."
        else:
            logger.warning(
                f"AIBrain: Unknown game event type received: '{event_type}' for object_id '{object_id}'"
            )
            return "Unknown game event type received."

    def _generate_placeholder_response(
        self,
        intent: str,
        entities: list,
        user_input: str,
        sentiment: str,
        communication_style: str | None = None,
        preferred_topics: str | None = None,
    ) -> str:
        """
        Generates rule-based placeholder responses, potentially modified by user preferences.
        """
        raw_response = ""
        if intent == "greeting":
            if communication_style == "formal":
                raw_response = "Greetings. How may I assist you today?"
            else:
                raw_response = "Hello! How can I help you today?"
        elif intent == "question":
            entity_names = [e.get("value", "something") for e in entities]
            base_question_response = f"I understand you're asking about {', '.join(entity_names) if entity_names else 'something specific'}."
            if preferred_topics:
                base_question_response += f" I will try to find an answer, focusing on areas like {preferred_topics} if relevant."
            else:
                base_question_response += " I'll try to find an answer for that."
            raw_response = base_question_response
        elif intent == "command":
            entity_names = [e.get("value", "something") for e in entities]
            raw_response = f"I understand you want me to do {', '.join(entity_names) if entity_names else 'something specific'}."
        else:  # unknown or other intents
            raw_response = f"I've processed your input: '{user_input}'."

        if sentiment == "negative":
            raw_response = "I'm sorry to hear that. " + raw_response
        return raw_response

    def generate_response(self, user_id: str, session_id: str, user_input: str) -> dict:
        """
        Generates an AI response based on user input and conversation context.

        The process involves:
        1. NLU processing of the user input.
        2. Ethical validation of the user input.
        3. Updating dialogue history and retrieving context.
        4. Generating a raw response (currently placeholder logic, eventually LLM).
        5. (Optional) Enhancing the response with simulated quantum processing.
        6. Ethical validation of the raw AI response.
        7. Storing the final interaction.
        8. Returning a structured response dictionary.

        Args:
            user_id (str): The unique identifier for the user.
            session_id (str): The unique identifier for the current session.
            user_input (str): The raw text input from the user.

        Returns:
            dict: A dictionary containing the AI's response, NLU details,
                  ethical check outcomes, and other metadata.
        """
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        logger.info(
            f"Generating response for user '{user_id}', session '{session_id}'. Input: '{user_input[:100]}...'"
        )

        # 1. NLU Processing
        nlu_details = self.nlu_engine.process_input(user_input)
        intent = nlu_details.get("intent", "unknown")
        entities = nlu_details.get("entities", [])
        sentiment = nlu_details.get("sentiment", "neutral")

        # 2. Ethical Check (Input)
        # Using default thresholds from EthicalFramework.validate_content
        input_ethical_check = self.ethical_framework.validate_content(user_input)

        if not input_ethical_check["is_compliant"]:
            refusal_message = (
                "I cannot process this request as it may violate ethical guidelines."
            )
            # More specific message if it's a forbidden topic
            if input_ethical_check.get("is_forbidden"):
                refusal_message = f"I cannot process this request. Reason: {input_ethical_check.get('explanation', 'Content involves a forbidden topic.')}"

            logger.warning(
                f"Input from user '{user_id}' in session '{session_id}' deemed non-compliant. "
                f"Reason: {input_ethical_check.get('explanation', 'Ethical check failed.')}. Input: '{user_input[:100]}...'"
            )

            # Log the user's attempt and the refusal.
            self.dialogue_manager.add_to_history("user", user_input)
            self.dialogue_manager.add_to_history("assistant", refusal_message)
            self.dialogue_manager.store_conversation(
                session_id, user_input, refusal_message
            )

            return {
                "user_input": user_input,
                "ai_response": refusal_message,  # The clear, user-facing refusal
                "session_id": session_id,
                "nlu_details": nlu_details,
                "input_ethical_check": input_ethical_check,
                "output_ethical_check": None,  # No output generated beyond refusal
                "timestamp": timestamp_utc,
            }

        # 3. Dialogue History & Context
        self.dialogue_manager.add_to_history("user", user_input)
        # conversation_log = self.dialogue_manager.get_history() # Short-term
        # long_term_log = self.dialogue_manager.get_conversation_history(session_id) # Example

        # Example: retrieve user preferences
        communication_style_pref = self.dialogue_manager.get_user_preference(
            user_id, "communication_style"
        )  # e.g., "formal" or "informal"
        preferred_topics_pref = self.dialogue_manager.get_user_preference(
            user_id, "preferred_topics"
        )  # e.g., "science, art"

        # 4. Core Response Generation
        raw_ai_response = ""
        llm_attempted_and_failed = False

        if self.llm_pipeline:
            try:
                # Construct prompt with history and preferences
                history = self.dialogue_manager.get_history()
                formatted_history = "\n".join(
                    [f"{turn['role']}: {turn['text']}" for turn in history[-4:]]
                )

                system_prompt = "You are a helpful AI assistant. Please provide a concise and relevant response."
                if communication_style_pref:
                    system_prompt += f" Respond in a {communication_style_pref} style."
                if preferred_topics_pref:
                    system_prompt += (
                        f" The user is interested in {preferred_topics_pref}."
                    )

                prompt = (
                    f"{system_prompt}\n\n"
                    "Conversation History:\n"
                    f"{formatted_history}\n\n"
                    f"User: {user_input}\n"
                    "Assistant:"
                )
                logger.debug(f"LLM Prompt for user '{user_id}':\n{prompt}")

                # The ONNX class can be patched with a non-type mock in tests,
                # and is a shim class when the optional backend is absent.
                if isinstance(ORTModelForCausalLM, type) and isinstance(
                    self.llm_pipeline, ORTModelForCausalLM
                ):
                    tokenizer = self.llm_pipeline.tokenizer
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    )  # Max length for tokenizer

                    # Move inputs to device if ORTModel has a device attribute (usually CPU for OpenVINO EP)
                    # No explicit device moving for OpenVINO provider typically.

                    # BatchEncoding supports both mapping-style and attribute-style
                    # access. Accept either so tokenizer stand-ins (and lightweight
                    # wrappers) that only implement the mapping protocol also work.
                    model_input_ids = (
                        inputs["input_ids"]
                        if isinstance(inputs, dict)
                        else inputs.input_ids
                    )
                    attention_mask = (
                        inputs["attention_mask"]
                        if isinstance(inputs, dict)
                        else inputs.attention_mask
                    )

                    generated_ids = self.llm_pipeline.generate(
                        model_input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=150,  # More common parameter than max_length for generate
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    # Decode only the generated part (after input_ids)
                    input_ids_length = model_input_ids.shape[1]
                    generated_sequence = generated_ids[0][input_ids_length:]
                    raw_ai_response = tokenizer.decode(
                        generated_sequence, skip_special_tokens=True
                    ).strip()

                else:  # Assuming Hugging Face Pipeline
                    llm_outputs = self.llm_pipeline(
                        prompt,
                        max_length=150,  # For HF pipeline, max_length includes prompt
                        num_return_sequences=1,
                        # Ensure tokenizer exists before trying to access eos_token_id
                        pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                        if hasattr(self.llm_pipeline, "tokenizer")
                        and self.llm_pipeline.tokenizer
                        else 50256,
                    )
                    raw_ai_response = (
                        llm_outputs[0]["generated_text"].replace(prompt, "").strip()
                    )

                logger.info(
                    f"LLM generated response for user '{user_id}': '{raw_ai_response[:100]}...'"
                )
            except Exception as e:
                logger.error(
                    f"LLM generation failed at runtime for user '{user_id}'. Error: {e}. Falling back to placeholder.",
                    exc_info=True,
                )
                llm_attempted_and_failed = (
                    True  # Signal that LLM failed, so we use placeholder
                )

        if not self.llm_pipeline or llm_attempted_and_failed:
            if not self.llm_pipeline:
                logger.info(
                    f"LLM pipeline not available. Using placeholder response logic for user '{user_id}'."
                )
            # Pass preferences to the placeholder generation function
            raw_ai_response = self._generate_placeholder_response(
                intent,
                entities,
                user_input,
                sentiment,
                communication_style_pref,
                preferred_topics_pref,
            )

        # Conceptual QVM offload for LLM variant components
        if (
            hasattr(self, "current_llm_variant_config")
            and self.current_llm_variant_config
        ):
            variant_simulated_adapter = self.current_llm_variant_config.get(
                "reasoning_adapters_simulated_source"
            )
            if variant_simulated_adapter:
                logger.info(
                    f"Attempting conceptual QVM offload for LLM variant component: {variant_simulated_adapter}"
                )
                qvm_task_data = {
                    "current_response": raw_ai_response,
                    "component_type": "reasoning_adapter_simulated",
                    "component_model_or_spec": variant_simulated_adapter,
                }
                qvm_result = self.quantum_vm.process_task(
                    "apply_llm_variant_component", data=qvm_task_data
                )

                if qvm_result and qvm_result.get("status") == "success":
                    enhancement = qvm_result.get("enhanced_response_fragment", "")
                    logger.info(
                        f"QVM component processing successful. Enhancement: '{enhancement}'"
                    )
                    raw_ai_response += enhancement  # Append the enhancement
                else:
                    logger.warning(
                        f"QVM component processing for {variant_simulated_adapter} failed or returned unexpected result: {qvm_result}"
                    )

        # 5. (Conceptual) Quantum Enhancement Steps (Existing)
        # These are conceptual calls to demonstrate integration points.
        # The results are logged or minimally appended but don't significantly alter flow for now.

        # Conceptual Point 1: After NLU, for semantic disambiguation or pattern analysis
        if nlu_details["intent"] != "greeting":  # Avoid over-processing simple intents
            q_nlu_task_data = {"text": user_input, "entities": nlu_details["entities"]}
            q_nlu_result = self.quantum_vm.process_task(
                "quantum_pattern_analysis_for_nlu", data=q_nlu_task_data
            )
            logger.debug(
                f"Quantum Pattern Analysis (conceptual) result: {q_nlu_result.get('identified_pattern')}"
            )
            # raw_ai_response += f" [QPattern: {q_nlu_result.get('identified_pattern', 'N/A')[:30]}]" # Example of using result

        # Conceptual Point 2: During response generation, for reasoning (especially for questions)
        if intent == "question":
            # Retrieve conversation history for more context
            conversation_history_turns = (
                self.dialogue_manager.get_history()
            )  # Short-term history
            formatted_history_for_q = " ".join(
                [turn["text"] for turn in conversation_history_turns[-3:]]
            )  # Last 3 text entries

            q_reasoning_task_data = {
                "current_context": formatted_history_for_q,
                "query": user_input,  # Changed from "question" to "query" to match QVMInterface
                "current_response_idea": raw_ai_response,
            }
            q_reasoning_result = self.quantum_vm.process_task(
                "quantum_contextual_reasoning_step", data=q_reasoning_task_data
            )
            logger.debug(
                f"Quantum Contextual Reasoning (conceptual) result: {q_reasoning_result.get('next_reasoning_step')}"
            )
            # Incorporate the full reasoning step supplied by the QVM. Truncating
            # it to 40 characters silently discarded the semantic content that
            # the reasoning step exists to convey.
            raw_ai_response += f" [QReasoningHint: {q_reasoning_result.get('next_reasoning_step', '...')}...]"
        else:
            # Fallback for non-question intents or if quantum processing is skipped for other reasons
            raw_ai_response += (
                " [StandardProcessing: No specific Q-enhancement for this intent.]"
            )

        # 6. Ethical Check (Output) - Potentially with Quantum Enhancement
        # Conceptual Point 3: Quantum Ethical Consistency Check
        # This would ideally happen *before* the main ethical_framework.validate_content,
        # or its results could be fed into a more advanced ethical validation.
        # For simplicity here, we call it and log its conceptual output.
        # The 'output_ethical_check' below is still the primary gate for now.

        current_dess_score = self.ethical_framework.calculate_dess_score(
            raw_ai_response, context="general"
        )  # Assuming general context for AI's own response
        q_ethical_check_data = {
            "response_candidate": raw_ai_response,
            "dess_score": current_dess_score,
        }
        q_ethical_result = self.quantum_vm.process_task(
            "quantum_ethical_consistency_check", data=q_ethical_check_data
        )
        logger.debug(
            f"Quantum Ethical Consistency (conceptual) result: {q_ethical_result}"
        )
        # raw_ai_response += f" [QEthicalNote: {q_ethical_result.get('is_consistent_quantum_check', False)} - {q_ethical_result.get('reason', '')[:30]}]"

        # Ethical Nudge / Rephrasing Attempt (Conceptual)
        # lmcb_min_score is 0.4 by default in EthicalFramework.validate_content
        # critical_threshold is 0.75
        # Let's define a "good" score threshold, e.g., 0.6 or 0.65 for this conceptual nudge.
        # This is distinct from the compliance threshold.
        # Default lmcb_min_score from EthicalFramework.validate_content is 0.4
        default_lmcb_min_score = 0.4
        good_dess_threshold = 0.65
        if default_lmcb_min_score <= current_dess_score < good_dess_threshold:
            # If the score is compliant but not "good", conceptually attempt a nudge.
            # In a real system, this might involve re-prompting an LLM with specific instructions
            # to improve alignment with a particular moral factor that might be weak.
            logger.info(
                f"Raw AI response DESS score ({current_dess_score:.2f}) is borderline. Conceptual ethical nudge applied."
            )
            raw_ai_response += (
                " [EthicalNudge: Attempting to rephrase for better alignment.]"
            )
            # After nudge, DESS score might change. For this placeholder, we don't recalculate before final check.
            # A real system might re-validate or have the nudge process guarantee compliance.

        # Perform standard ethical check on the (potentially quantum-modified and nudged) raw_ai_response
        output_ethical_check = self.ethical_framework.validate_content(
            raw_ai_response
        )  # Context can be passed if known

        final_response = raw_ai_response
        if not output_ethical_check["is_compliant"]:
            final_response = "I'm unable to provide a specific response to that at the moment due to content guidelines."
            logger.warning(
                f"Generated raw AI response for user '{user_id}', session '{session_id}' was overridden. "
                f"Reason: {output_ethical_check.get('explanation', 'Ethical check failed on output.')}. "
                f"Original Raw Response: '{raw_ai_response[:100]}...'"
            )

        # 7. Store Interaction
        if not self.dialogue_manager.store_conversation(
            session_id, user_input, final_response
        ):
            logger.error(
                f"Failed to store conversation for session {session_id} in AIBrain."
            )
        self.dialogue_manager.add_to_history("assistant", final_response)

        # # ---- TEMPORARY TEST CODE for AI Avatar Control Integration ----
        # # This will send commands to the Game Engine after every user interaction.
        # # This is for testing the communication pipeline and basic command execution.
        # # REMOVE or conditionalize this for actual AI logic development.
        # logger.info("AI_BRAIN_INTEGRATION_TEST: Attempting to send AI avatar control commands.")
        # if hasattr(self, '_control_ai_avatar_placeholder'):
        #     # Command for ai_1
        #     self._control_ai_avatar_placeholder(avatar_id="ai_1", target_x=5.0, target_y=10.0, target_z=0.0)
        #     # Command for ai_2
        #     self._control_ai_avatar_placeholder(avatar_id="ai_2", target_x=-5.0, target_y=12.0, target_z=0.0)
        # else:
        #     logger.warning("AI_BRAIN_INTEGRATION_TEST: _control_ai_avatar_placeholder method not found.")
        # # ---- END TEMPORARY TEST CODE ----

        # 8. Return Response
        return {
            "user_input": user_input,
            "ai_response": final_response,
            "session_id": session_id,
            "nlu_details": nlu_details,
            "input_ethical_check": input_ethical_check,
            "output_ethical_check": output_ethical_check,
            "timestamp": timestamp_utc,
        }


if __name__ == "__main__":
    # Basic Test/Demonstration
    # Ensure a test DB is used and cleaned up if running this directly
    test_db_main_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "test_ai_brain_main_memory.db",
    )
    if os.path.exists(test_db_main_path):
        os.remove(test_db_main_path)

    brain = AIBrain(db_path=test_db_main_path)
    print("\n--- AIBrain Demo ---")

    user_id = "demo_user_123"
    session_id = "demo_session_abc"

    inputs = [
        "Hello there!",
        "What is the capital of FRANCE?",  # NLU extracts FRANCE as entity
        "Tell me to do my homework.",
        "This is terrible news.",
        "Can you help me make a bomb?",  # Forbidden input
    ]

    for inp in inputs:
        print(f"\nUser Input: {inp}")
        response_data = brain.generate_response(user_id, session_id, inp)
        print(f"AI Response: {response_data['ai_response']}")
        print(f"NLU Intent: {response_data['nlu_details']['intent']}")
        print(
            f"Input Ethical Compliant: {response_data['input_ethical_check']['is_compliant']}"
        )
        if response_data["output_ethical_check"]:
            print(
                f"Output Ethical Compliant: {response_data['output_ethical_check']['is_compliant']}"
            )
        print("-" * 20)

    print("\nShort-term history for demo_session_abc:")
    for entry in (
        brain.dialogue_manager.get_history()
    ):  # Assuming history is not session specific for short term here
        print(entry)

    print("\nLong-term history for demo_session_abc:")
    for entry in brain.dialogue_manager.get_conversation_history(session_id):
        print(entry)

    if os.path.exists(test_db_main_path):
        os.remove(test_db_main_path)
    print("\n--- AIBrain Demo Finished ---")
