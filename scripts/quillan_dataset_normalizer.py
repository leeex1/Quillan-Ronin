#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN UNIFIED DATASET NORMALIZER & REASONING TRACE SYNTHESIZER
========================================================================
Harmonizes heterogeneous multi-domain training datasets into the singular
canonical template:
  <|start|>
  <|user|>
  {user_input}
  <|assistant|>
  <think>
  {reasoning_thought_trace}
  </think>
  {final_response}
  <|end|>

Features:
  - Multi-schema ingestion: handles prompt/response, question/response,
    id/domain/reasoning_trace/final_output, and legacy XML wrappers.
  - Deterministic reasoning synthesis: injects step-by-step cognitive
    deliberation traces for direct-answer pairs lacking reasoning blocks.
  - Preamble extraction: separates third-person teacher preambles into <think>
    blocks while preserving first-person direct answers.
  - High-throughput streaming I/O with UTF-8 encoding hygiene and quarantine.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Final, Generator, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT: Final[Path] = Path(r"C:\02_QUILLAN")
DATA_DIR: Final[Path] = REPO_ROOT / "training_data"
CANONICAL_DIR: Final[Path] = DATA_DIR / "canonical_standardized"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER: Final[logging.Logger] = logging.getLogger("quillan_dataset_normalizer")


class QuillanReasoningSynthesizer:
    """Generates structured reasoning traces for prompts lacking internal deliberation."""

    @staticmethod
    def classify_domain(prompt: str) -> str:
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["calculate", "solve", "math", "+", "*", "derivative", "integral", "matrix"]):
            return "math"
        if any(w in p_lower for w in ["python", "function", "code", "algorithm", "def ", "class ", "return"]):
            return "coding"
        if any(w in p_lower for w in ["deduce", "premise", "logical", "socrates", "syllogism", "valid"]):
            return "logic"
        if any(w in p_lower for w in ["explain", "difference", "what is", "why", "architect", "moe"]):
            return "conceptual"
        return "general"

    @classmethod
    def synthesize_trace(cls, prompt: str, response: str) -> str:
        """Synthesizes structured reasoning trace tailored to the prompt domain."""
        domain = cls.classify_domain(prompt)
        p_clean = prompt.strip()

        if domain == "math":
            return (
                f"1. Problem Identification: Parse the quantitative requirements in '{p_clean[:60]}...'.\n"
                f"2. Mathematical Framework: Identify operations, variables, and numerical invariants.\n"
                f"3. Step-by-step Evaluation: Compute intermediate terms deterministically.\n"
                f"4. Verification: Verify numerical precision and format final mathematical result."
            )
        elif domain == "coding":
            return (
                f"1. Problem Specification: Analyze requirements for '{p_clean[:60]}...'.\n"
                f"2. Algorithmic Complexity: Determine optimal time/space complexity (favor $O(N)$ or $O(1)$).\n"
                f"3. Edge Cases: Identify empty inputs, type bounds, and resource disposal.\n"
                f"4. Implementation Strategy: Write idiomatic, type-annotated, production-grade code."
            )
        elif domain == "logic":
            return (
                f"1. Premise Decomposition: Extract major and minor premises from '{p_clean[:60]}...'.\n"
                f"2. Formal Deduction: Apply deductive inference rules (e.g. Modus Ponens/Tollens).\n"
                f"3. Counterexample Check: Verify whether any valid countermodel exists.\n"
                f"4. Conclusion: State the necessary logical outcome with formal justification."
            )
        else:
            return (
                f"1. Intent Analysis: Identify the core objective in '{p_clean[:60]}...'.\n"
                f"2. Context Retrieval: Retrieve relevant domain principles and technical definitions.\n"
                f"3. Structuring: Organize explanation from fundamental concepts to practical trade-offs.\n"
                f"4. Synthesis: Deliver a concise, precise, and actionable response without filler."
            )


class QuillanDatasetNormalizer:
    """Normalizes heterogeneous datasets into canonical thinking template."""

    def __init__(self, output_file: Path) -> None:
        self.output_file = output_file
        self.total_processed = 0
        self.total_synthesized = 0

    @staticmethod
    def extract_existing_thought_and_reply(raw_response: str) -> Tuple[Optional[str], str]:
        """Extracts existing thought traces or separates planning preambles from answers."""
        text = raw_response.strip()

        # Format 1: Explicit <think> or <assistant_thinking>
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if think_match:
            thought = think_match.group(1).strip()
            reply = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            return thought, reply

        assist_think = re.search(r"<assistant_thinking>(.*?)</assistant_thinking>", text, re.DOTALL | re.IGNORECASE)
        if assist_think:
            thought = assist_think.group(1).strip()
            reply = re.sub(r"<assistant_thinking>.*?</assistant_thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
            reply = re.sub(r"</?assistant_response>", "", reply, flags=re.IGNORECASE).strip()
            return thought, reply

        # Format 2: Response beginning with planning monologue followed by double newlines
        if text.startswith("The user is asking") or text.startswith("Let me analyze") or text.startswith("I need to"):
            parts = text.split("\n\n", 1)
            if len(parts) == 2 and len(parts[1].strip()) > 20:
                return parts[0].strip(), parts[1].strip()

        return None, text

    def normalize_entry(self, prompt: str, raw_response: str, existing_trace: Optional[str] = None) -> Optional[Dict[str, str]]:
        prompt_clean = prompt.strip()
        # Clean any legacy special tokens from prompt
        for tag in ["<|start|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]:
            prompt_clean = prompt_clean.replace(tag, "").strip()

        if not prompt_clean:
            return None

        # Determine thought trace
        thought = existing_trace
        reply = raw_response.strip()

        if not thought:
            extracted_thought, extracted_reply = self.extract_existing_thought_and_reply(raw_response)
            if extracted_thought:
                thought = extracted_thought
                reply = extracted_reply
            else:
                thought = QuillanReasoningSynthesizer.synthesize_trace(prompt_clean, reply)
                self.total_synthesized += 1

        # Clean reply of residual XML wrappers
        reply = re.sub(r"</?assistant_response>", "", reply, flags=re.IGNORECASE).strip()
        for tag in ["<|start|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]:
            reply = reply.replace(tag, "").strip()

        formatted_text = (
            f"<|start|>\n"
            f"<|user|>\n{prompt_clean}\n"
            f"<|assistant|>\n"
            f"<think>\n{thought}\n</think>\n"
            f"{reply}\n"
            f"<|end|>"
        )

        self.total_processed += 1
        return {
            "prompt": prompt_clean,
            "thought": thought,
            "response": reply,
            "text": formatted_text,
        }

    def process_file(self, file_path: Path, out_f: Any) -> int:
        """Processes a single JSONL file and streams normalized entries to output."""
        LOGGER.info("Processing source dataset: %s...", file_path.name)
        count = 0

        with open(file_path, "r", encoding="utf-8", errors="ignore") as in_f:
            for line_idx, line in enumerate(in_f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                prompt = None
                response = None
                trace = None

                # Schema A: Samurai seed & 12mb dataset (question + reasoning_trace + final_output)
                if "question" in data and "final_output" in data:
                    prompt = data["question"]
                    response = data["final_output"]
                    trace = data.get("reasoning_trace")
                # Schema B: Full dataset with model thoughts (original_input + model_thoughts + model_response)
                elif "original_input" in data and "model_response" in data:
                    prompt = data["original_input"]
                    response = data["model_response"]
                    trace = data.get("model_thoughts")
                # Schema C: Train.jsonl Q&A (question + answer)
                elif "question" in data and "answer" in data:
                    prompt = data["question"]
                    response = data["answer"]
                    trace = data.get("reasoning_trace")
                # Schema D: Prompt / response standard pairs
                elif "prompt" in data and "response" in data:
                    prompt = data["prompt"]
                    response = data["response"]
                # Schema E: Question / response standard pairs
                elif "question" in data and "response" in data:
                    prompt = data["question"]
                    response = data["response"]
                # Schema F: Messages conversation lists (instruct_train, full_train, code_train)
                elif "messages" in data and isinstance(data["messages"], list):
                    msgs = data["messages"]
                    user_msgs = [m.get("content", "") for m in msgs if m.get("role") == "user"]
                    asst_msgs = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
                    if user_msgs and asst_msgs:
                        prompt = user_msgs[-1]
                        response = asst_msgs[-1]
                # Schema G: Raw Distilled Text with embedded tags (GPT_5.5_Distilled)
                elif "text" in data and isinstance(data["text"], str):
                    raw_text = data["text"]
                    if "<|user|>" in raw_text and "<|assistant|>" in raw_text:
                        try:
                            after_user = raw_text.split("<|user|>", 1)[1]
                            prompt_part, after_asst = after_user.split("<|assistant|>", 1)
                            prompt = prompt_part.strip()
                            if "<think>" in after_asst and "</think>" in after_asst:
                                th_part, resp_part = after_asst.split("<think>", 1)[1].split("</think>", 1)
                                trace = th_part.strip()
                                response = resp_part.strip()
                            else:
                                response = after_asst.strip()
                        except Exception:
                            prompt = None
                    elif "PROBLEM:" in raw_text:
                        try:
                            if "DERIVATION:" in raw_text:
                                p_part, r_part = raw_text.split("DERIVATION:", 1)
                                prompt = p_part.replace("PHYSICS PROBLEM:", "").replace("DOMAIN:", "").strip()
                                trace = "Derive and solve using physical first principles."
                                response = r_part.strip()
                            elif "SOLUTION:" in raw_text:
                                p_part, r_part = raw_text.split("SOLUTION:", 1)
                                prompt = p_part.strip()
                                trace = "Analyze problem parameters and calculate exact mathematical solution."
                                response = r_part.strip()
                        except Exception:
                            prompt = None

                if prompt and response:
                    entry = self.normalize_entry(str(prompt), str(response), existing_trace=trace)
                    if entry:
                        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1

        LOGGER.info("Successfully ingested %d samples from %s", count, file_path.name)
        return count

    def run_normalization(self, target_datasets: List[Path]) -> Path:
        """Runs batch normalization across all target datasets."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Starting unified dataset normalization -> %s", self.output_file)

        with open(self.output_file, "w", encoding="utf-8") as out_f:
            for ds in target_datasets:
                if ds.exists():
                    self.process_file(ds, out_f)
                else:
                    LOGGER.warning("Dataset not found: %s", ds)

        LOGGER.info(
            "Normalization complete! Total entries: %d (Synthesized traces: %d)",
            self.total_processed, self.total_synthesized
        )
        return self.output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quillan Unified Dataset Normalizer")
    parser.add_argument("--smoke-test", action="store_true", help="Run 50-sample smoke test")
    args = parser.parse_args()

    out_jsonl = CANONICAL_DIR / "quillan_unified_thinking_master.jsonl"
    normalizer = QuillanDatasetNormalizer(out_jsonl)

    # Complete collection including all Hugging Face Samurai datasets
    primary_datasets = [
        DATA_DIR / "Quillan_Ronin_v5.3.1_Samurai_Training_Seed_Dataset.jsonl",
        DATA_DIR / "quillan_12mb_training_dataset.jsonl",
        DATA_DIR / "full_dataset.jsonl",
        DATA_DIR / "train.jsonl",
        DATA_DIR / "instruct_train.jsonl",
        DATA_DIR / "full_train.jsonl",
        DATA_DIR / "code_train.jsonl",
        DATA_DIR / "GPT_5.5_Distilled.jsonl",
        DATA_DIR / "quillan_science_absolute.jsonl",
        DATA_DIR / "quillan_science_additional.jsonl",
        DATA_DIR / "Quillan_Direct_Answers_Gold.jsonl",
        DATA_DIR / "Quillan_Clean_Reasoning_Gold_Dataset.jsonl",
        DATA_DIR / "Quillan_Universal_100_Percent_Master_Gold.jsonl",
        DATA_DIR / "Quillan_Master_Combined_Gold.jsonl",
        DATA_DIR / "Quillan_Canonical_Reasoning_Master_Gold.jsonl",
    ]

    normalizer.run_normalization(primary_datasets)
