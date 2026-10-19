#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚔️ QUILLAN-RONIN v5.4.0-ONI — STRICT JSON SCHEMA TOOLS FOR QWEN
================================================================
Architect: CrashOverrideX
Provides strictly-typed JSON Schema tool definitions for Qwen models (Qwen2.5-Coder / Qwen3).
No untyped 'Any' fields — full type, properties, and required validation.
"""

from typing import Dict, Any, List

# Strict JSON Schema definitions for the 19 MCP Tools
QWEN_STRICT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_quillan_pipeline",
            "description": "Execute the deterministic 6-stage Quillan-Ronin v5.4.0-ONI MCP Chain Bus across all custom and system MCPs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user query or task instruction to process through the 6-stage pipeline."
                    },
                    "context_json": {
                        "type": "string",
                        "description": "Optional serialized JSON string containing active context or state variables."
                    },
                    "max_retries": {
                        "type": "integer",
                        "description": "Maximum RCI self-healing iteration attempts (default: 5)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prism_decompose",
            "description": "Deconstruct a query into the 9-Vector Semantic Prism (Language, Sentiment, Context, Intent, Meta, Creativity, Ethics, Strategy, Constraint).",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The target text or instruction to decompose across all 9 semantic vectors."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "governor_compute_pacing",
            "description": "Compute Venturi PID token velocity and thermodynamic pacing under Lee-Mach-6 governor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent_score": {
                        "type": "number",
                        "description": "Estimated intent scalar in [0.0, 1.0]."
                    },
                    "constraint_score": {
                        "type": "number",
                        "description": "Estimated constraint scalar in [0.0, 1.0]."
                    }
                },
                "required": ["intent_score", "constraint_score"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "council_deliberate_34",
            "description": "Execute 34-expert council deliberation across Cognitive, Systems, Meta, and Communication clusters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal": {
                        "type": "string",
                        "description": "Candidate proposal or reasoning hypothesis to deliberate upon."
                    },
                    "active_cluster": {
                        "type": "string",
                        "enum": ["all", "cognitive", "systems", "meta", "communication"],
                        "description": "Target council cluster for pull-weight aggregation."
                    }
                },
                "required": ["proposal", "active_cluster"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "samurai_gate_check",
            "description": "Evaluate a candidate code modification or output against all 5 Operational Gates (Warden, Shepherd, CodeWeaver, Predator, Aurelion).",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": "Code snippet, diff, or proposal to evaluate against the 5 operational gates."
                    }
                },
                "required": ["payload"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "codeweaver_ast_inspect",
            "description": "Perform static AST inspection, syntax validation, and test-seam analysis on Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_content": {
                        "type": "string",
                        "description": "Raw Python code content to inspect."
                    },
                    "target_file": {
                        "type": "string",
                        "description": "Optional file path being analyzed."
                    }
                },
                "required": ["code_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "quillan_rag_search",
            "description": "Query the local Quillan knowledge base and vector database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of relevant chunks to retrieve (default: 5)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "model_lab_eval_offload",
            "description": "Check model hardware offload boundaries for GTX 1050 Ti (4GB VRAM) and 28GB System RAM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_params_millions": {
                        "type": "number",
                        "description": "Parameter count in millions (e.g. 455.7)."
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Inference or training batch size."
                    }
                },
                "required": ["model_params_millions", "batch_size"]
            }
        }
    }
]


def get_qwen_tools() -> List[Dict[str, Any]]:
    """Return all strictly-typed JSON Schema tool definitions."""
    return QWEN_STRICT_TOOLS


if __name__ == "__main__":
    import json
    print(f"Exported {len(QWEN_STRICT_TOOLS)} strictly-typed Qwen tools:")
    for t in QWEN_STRICT_TOOLS:
        fn = t["function"]
        print(f"  - {fn['name']}: {len(fn['parameters']['properties'])} properties, required: {fn['parameters']['required']}")
