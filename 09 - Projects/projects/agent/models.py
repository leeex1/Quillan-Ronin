"""Quillan-Ronin model registry.

All endpoints below were verified working as free on NVIDIA NIM
(integrate.api.nvidia.com/v1, 2026-08). Context windows from HF config.json
where available; otherwise conservative published values.
Quillan may pick a model per reply. Rotation list used as failsafe.
"""
import json

MODELS = [
    {"id": "meta/llama-3.1-8b-instruct", "context": 131072, "strength": "balanced all-rounder, strong tool use"},
    {"id": "meta/llama-3.1-70b-instruct", "context": 131072, "strength": "bigger brain, deeper reasoning, slower"},
    {"id": "meta/llama-3.2-11b-vision-instruct", "context": 131072, "strength": "vision-capable + strong text"},
    {"id": "deepseek-ai/deepseek-v4-flash-0731", "context": 1048576, "strength": "1M context, fast, code + reasoning"},
    {"id": "nvidia/llama-3.3-nemotron-super-49b-v1", "context": 131072, "strength": "49B super model, strong reasoning"},
    {"id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "context": 131072, "strength": "49B super v1.5, reasoning"},
    {"id": "z-ai/glm-5.2", "context": 1048576, "strength": "1M context, long-form philosophical depth"},
    {"id": "openai/gpt-oss-20b", "context": 131072, "strength": "20B open model, capable all-rounder"},
    {"id": "minimaxai/minimax-m3", "context": 262144, "strength": "creative, strong prose + persona voice"},
    {"id": "nvidia/nemotron-3-nano-30b-a3b", "context": 131072, "strength": "30B active 3B, fast + smart"},
    {"id": "nvidia/nemotron-3-super-120b-a12b", "context": 131072, "strength": "120B active 12B, very strong"},
    {"id": "nvidia/nemotron-mini-4b-instruct", "context": 4096, "strength": "small + fast, lightweight replies only"},
    {"id": "nvidia/nemotron-nano-12b-v2-vl", "context": 32768, "strength": "nano size, vision-capable"},
    {"id": "nvidia/nvidia-nemotron-nano-9b-v2", "context": 131072, "strength": "9B nano v2, balanced"},
    {"id": "stepfun-ai/step-3.7-flash", "context": 131072, "strength": "flash fast, strong reasoning"},
    {"id": "thinkingmachines/inkling", "context": 131072, "strength": "reasoning-focused, thoughtful"},
]

# Failsafe rotation order — best/most capable first, small fast ones for retries.
ROTATION = [m["id"] for m in MODELS]

# Cheat sheet the agent can review.
CHEAT_SHEET = json.dumps(
    [
        {"model": m["id"], "context": m["context"], "strength": m["strength"]}
        for m in MODELS
    ],
    indent=2,
)


def model_info():
    """Return the cheat sheet for quillan to pick a model."""
    return CHEAT_SHEET


def get_model(model_id):
    for m in MODELS:
        if m["id"] == model_id:
            return m
    return None
