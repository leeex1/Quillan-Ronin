"""
Script to export Hugging Face models to ONNX format.
"""
import argparse
import logging
import os
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from optimum.exporters.onnx import main_export
except ImportError:
    logger.error("Optimum library not found. Please ensure it's installed with ONNX support: pip install optimum[onnxruntime]")
    main_export = None # To prevent script from running further if import fails

def export_model_to_onnx(model_id: str, output_path_str: str, task: str = "text-generation", fp16: bool = False):
    """
    Exports a Hugging Face model to ONNX format.

    Args:
        model_id (str): The Hugging Face model ID (e.g., "distilgpt2").
        output_path_str (str): The path to the directory where the ONNX model will be saved.
        task (str): The task for which the model is being exported (e.g., "text-generation").
        fp16 (bool): Whether to use FP16 precision. Defaults to False.
    """
    if not main_export:
        logger.error("main_export function not available due to import error. Cannot export model.")
        return

    output_path = Path(output_path_str)
    if not output_path.exists():
        logger.info(f"Output directory {output_path} does not exist. Creating it.")
        output_path.mkdir(parents=True, exist_ok=True)
    elif not output_path.is_dir():
        logger.error(f"Output path {output_path} exists but is not a directory.")
        return

    try:
        logger.info(f"Starting ONNX export for model: {model_id}")
        logger.info(f"Output directory: {output_path}")
        logger.info(f"Task: {task}")
        logger.info(f"FP16 precision: {fp16}")

        main_export(
            model_name_or_path=model_id,
            output=output_path,
            task=task,
            # device="cpu", # Can be specified if needed, defaults usually work fine.
            fp16=fp16,
            do_constant_folding=True, # Generally recommended for inference optimization
            # trust_remote_code=True, # Add this if model requires it (e.g. some Mixtral, Phi-3)
        )
        logger.info(f"Model {model_id} successfully exported to ONNX at {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during ONNX export for model {model_id}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Hugging Face models to ONNX.")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Hugging Face model ID (e.g., 'microsoft/Phi-3-mini-4k-instruct')."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the directory to save the ONNX model (e.g., 'models/phi-3-mini_onnx')."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="The task for the model (default: 'text-generation')."
    )
    parser.add_argument(
        "--fp16",
        action="store_true", # If present, sets to True
        help="Enable FP16 precision for the exported model."
    )
    # Potentially add --trust_remote_code if needed for models like Phi-3

    args = parser.parse_args()

    # Example usage:
    # python utils/export_to_onnx.py --model_id microsoft/Phi-3-mini-4k-instruct --output_path models/phi-3-mini_onnx --fp16
    # python utils/export_to_onnx.py --model_id distilgpt2 --output_path models/distilgpt2_onnx

    export_model_to_onnx(args.model_id, args.output_path, args.task, args.fp16)
