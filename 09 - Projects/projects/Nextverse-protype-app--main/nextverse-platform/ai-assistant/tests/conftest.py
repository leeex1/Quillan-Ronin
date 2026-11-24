"""Pytest harness for the NextVerse AI assistant.

Keeps the suite hermetic and deterministic:
  * forces CPU (this host's CUDA runtime has no compatible kernel image);
  * replaces the HuggingFace text-generation pipeline with a small fake that
    returns the same shape as a real pipeline, so no model is downloaded and
    response strings stay real strings (tests exercising loading patch it again);
  * puts the ai-assistant root on sys.path so ``core.*`` imports resolve;
  * aliases the historical ``nextverse_platform.ai_assistant`` import path to
    the real ``core.ai_brain`` module so legacy patch targets still resolve.
"""

from __future__ import annotations

import os
import sys
import types

# Must happen before torch/transformers are imported by any test module.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

AI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AI_ROOT not in sys.path:
    sys.path.insert(0, AI_ROOT)


class _FakeTokenizer:
    eos_token_id = 50256


class _FakeTextGenerationPipeline:
    """Minimal stand-in for transformers' text-generation pipeline."""

    tokenizer = _FakeTokenizer()

    def __call__(self, prompt, **kwargs):  # noqa: ANN002, ANN003 - pipeline API
        if isinstance(prompt, (list, tuple)):
            prompt = prompt[0] if prompt else ""
        return [{"generated_text": f"{prompt} offline test response"}]


def _fake_pipeline_factory(task, model=None, **kwargs):  # noqa: ANN002, ANN003
    return _FakeTextGenerationPipeline()


def pytest_configure(config: object) -> None:  # noqa: ANN001 - pytest hook
    import core.ai_brain as brain  # noqa: WPS433 - hook-local import is intentional

    brain.nlu_pipeline_func = _fake_pipeline_factory

    # Alias the legacy package path used by some @patch decorators.
    if "nextverse_platform.ai_assistant.core.ai_brain" not in sys.modules:
        npkg = types.ModuleType("nextverse_platform")
        npkg.__path__ = []  # type: ignore[attr-defined]
        apkg = types.ModuleType("nextverse_platform.ai_assistant")
        apkg.__path__ = []  # type: ignore[attr-defined]
        cpkg = types.ModuleType("nextverse_platform.ai_assistant.core")
        cpkg.__path__ = []  # type: ignore[attr-defined]
        npkg.ai_assistant = apkg  # type: ignore[attr-defined]
        apkg.core = cpkg  # type: ignore[attr-defined]
        cpkg.ai_brain = brain  # type: ignore[attr-defined]
        sys.modules.update(
            {
                "nextverse_platform": npkg,
                "nextverse_platform.ai_assistant": apkg,
                "nextverse_platform.ai_assistant.core": cpkg,
                "nextverse_platform.ai_assistant.core.ai_brain": brain,
            }
        )
