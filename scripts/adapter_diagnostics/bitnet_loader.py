#!/usr/bin/env python3
"""Load BitNet-2B-4T via transformers with torch.compile monkeypatched OUT.
HuggingFace's bitnet integration decorates unpack_weights/forward with @torch.compile,
which needs a C++ compiler (cl.exe) absent on this box. We replace the compiled
functions with eager equivalents before model load.
"""

from __future__ import annotations
import torch


def patch_bitnet_eager():
    import transformers.integrations.bitnet as b

    def unpack_eager(packed, dtype=torch.float32):
        packed = packed.to(torch.uint8)
        orig = packed.shape[0] * 4
        unpacked = torch.zeros(
            orig, *packed.shape[1:], dtype=torch.uint8, device=packed.device
        )
        for i in range(4):
            start = i * packed.shape[0]
            end = start + packed.shape[0]
            unpacked[start:end] = (packed & (3 << (2 * i))) >> (2 * i)
        return unpacked.to(dtype) - 1

    b.unpack_weights = unpack_eager
    try:
        import transformers.quantizers.quantizer_bitnet as qb

        qb.unpack_weights = unpack_eager
    except Exception:
        pass
    return unpack_eager


def load_bitnet(path):
    from transformers import AutoModelForCausalLM

    patch_bitnet_eager()
    import os

    try:
        torch._dynamo.config.disable = True
    except Exception:
        pass
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    m = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32
    )
    m = m.eval()
    for mod in m.modules():
        f = getattr(mod, "forward", None)
        if f is not None and hasattr(f, "__wrapped__"):
            try:
                mod.forward = f.__wrapped__.__get__(mod, type(mod))
            except Exception:
                pass
    return m
