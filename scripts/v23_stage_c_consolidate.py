#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage C: Consolidation & Long-Context GGUF Export for WSP v23.

Starts strictly from verified Stage B checkpoints (exports/v23_stageb_<model>_hf),
runs VRAM-safe chunked-CE LoRA consolidation on strictly zero-contamination
non-test training splits (assistant-only masked QA/MC/GSM8K/Code/BFCL/Sovereign
plus raw technical prose LM fluency), merges adapters cleanly into float32 Stage B
weights (param_diff == 0, zero adapter keys), and exports standalone Q8_0 GGUFs.

Usage:
  python -u v23_stage_c_consolidate.py --model flagship
  python -u v23_stage_c_consolidate.py --model flagship --convert-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, r"C:\02_QUILLAN\scripts")
import v23_common as c  # noqa: E402
import v23_stage_b_precombine as stage_b  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("V23_StageC")


SOVEREIGN_SYSTEM_ANCHOR = (
    "You are Quillan (C0-QUILLAN), Sovereign Samurai Collective. "
    "Prioritize rectitude, honesty, and technical precision over flattery. "
    "Verify every constraint inside deliberate reasoning before answering."
)


def format_chat_example(
    tokenizer: Any,
    user_text: str,
    assistant_text: str,
    system_text: Optional[str] = None,
    max_len: int = 240,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Tokenizes a (system?, user, assistant) pair with assistant-only loss masking (-100
    on prompt tokens). Preserves any system prefix and left-trims only excess user tokens
    if total sequence exceeds max_len.
    """
    messages_prompt: List[Dict[str, str]] = []
    if system_text:
        messages_prompt.append({"role": "system", "content": system_text})
    messages_prompt.append({"role": "user", "content": user_text})

    sys_keep = 0
    if getattr(tokenizer, "chat_template", None):
        prompt_str = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        if system_text:
            sys_str = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_text}],
                tokenize=False,
                add_generation_prompt=False,
            )
            sys_keep = len(tokenizer(sys_str, add_special_tokens=False)["input_ids"])
    else:
        sys_str = f"{system_text}\n" if system_text else ""
        prompt_str = sys_str + f"User: {user_text}\nAssistant: "
        if sys_str:
            sys_keep = len(tokenizer(sys_str, add_special_tokens=False)["input_ids"])

    eos = tokenizer.eos_token or ""
    full_str = prompt_str + assistant_text + eos

    p_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    if len(f_ids) <= len(p_ids) + 1:
        return None

    # Keep full assistant target and system prefix; trim only middle/user context if needed
    if len(f_ids) > max_len:
        ans_ids = f_ids[len(p_ids) :]
        if len(ans_ids) >= max_len - 24:
            ans_ids = ans_ids[: max_len - 24]
        keep_p = max_len - len(ans_ids)
        if 0 < sys_keep < keep_p:
            tail_p = keep_p - sys_keep
            p_ids = p_ids[:sys_keep] + p_ids[-tail_p:]
        else:
            p_ids = p_ids[-keep_p:]
        f_ids = p_ids + ans_ids

    input_ids = torch.tensor([f_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(p_ids) + f_ids[len(p_ids) :]], dtype=torch.long)
    return input_ids, labels


def format_lm_prose_example(
    tokenizer: Any, text: str, max_len: int = 240
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Tokenizes plain technical prose for direct next-token LM PPL reduction."""
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
    if len(ids) < 16:
        return None
    t = torch.tensor([ids], dtype=torch.long)
    return t, t.clone()


def build_stage_c_curriculum(
    tokenizer: Any, data_path: Path, max_len: int = 240
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, str]], List[str]]:
    """
    Constructs a strictly non-test balanced training dataset + held-out non-test dev MC/NLL split.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Disjoint dev sets (never trained on)
    _, val_texts = stage_b.build_calib_and_val_splits(data_path)
    dev_mc = data["mmlu_direct"][-60:] + data["arc_direct"][-40:]

    rng = np.random.default_rng(202603)
    train_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def _add_chat(
        items: List[Dict[str, str]],
        limit: int,
        weight_reps: int = 1,
        sys_prob: float = 0.20,
        think_wrap_prob: float = 0.0,
    ) -> int:
        pool = list(items)
        idx = rng.permutation(len(pool))[: min(limit, len(pool))]
        added = 0
        for i in idx:
            row = pool[int(i)]
            sys_txt = row.get("system")
            if not sys_txt and rng.random() < sys_prob:
                sys_txt = SOVEREIGN_SYSTEM_ANCHOR
            ans_txt = row["assistant"]
            if think_wrap_prob > 0.0 and "<think>" not in ans_txt and "\n#### " in ans_txt:
                if rng.random() < think_wrap_prob:
                    reason, final_ans = ans_txt.rsplit("\n#### ", 1)
                    ans_txt = f"<think>\n{reason.strip()}\n</think>\n\n#### {final_ans.strip()}"
            ex = format_chat_example(
                tokenizer,
                row["user"],
                ans_txt,
                system_text=sys_txt,
                max_len=max_len,
            )
            if ex is not None:
                for _ in range(weight_reps):
                    train_pairs.append(ex)
                added += weight_reps
        return added

    # 1. Exclude any rows already used in calib_prose/val_texts/dev_mc
    mmlu_train = data["mmlu_direct"][18:-60]
    arc_train = data["arc_direct"][10:-40]
    gsm_train = data["gsm8k"][8:]

    c_counts: Dict[str, int] = {}
    c_counts["mmlu_chat"] = _add_chat(mmlu_train, 550, sys_prob=0.20)
    c_counts["arc_chat"] = _add_chat(arc_train, 400, sys_prob=0.20)
    c_counts["gsm8k_chat"] = _add_chat(gsm_train, 450, sys_prob=0.20, think_wrap_prob=0.30)
    c_counts["code_chat"] = _add_chat(data.get("code", []), 260, weight_reps=2, sys_prob=0.20)
    c_counts["bfcl_chat"] = _add_chat(data.get("bfcl", []), 120, weight_reps=3, sys_prob=0.25)
    c_counts["sov_chat"] = _add_chat(
        data.get("sovereign_adherence", []) + data.get("identity", []),
        80,
        weight_reps=4,
        sys_prob=0.0,
    )

    # 2. STEM/technical prose SFT + direct next-token LM fluency (lowers PPL directly)
    stem_rows = data.get("stem_prose", [])
    c_counts["stem_chat"] = _add_chat(stem_rows, 350, sys_prob=0.15)

    lm_added = 0
    for prose in data["calib_prose"][:3]:
        ex = format_lm_prose_example(tokenizer, prose, max_len=max_len)
        if ex is not None:
            for _ in range(3):
                train_pairs.append(ex)
            lm_added += 3
    for row in stem_rows[:550]:
        ex = format_lm_prose_example(
            tokenizer, f"{row['user']}\n\n{row['assistant']}", max_len=max_len
        )
        if ex is not None:
            train_pairs.append(ex)
            lm_added += 1
    for row in data.get("code", [])[:250]:
        ex = format_lm_prose_example(
            tokenizer, f"{row['user']}\n\n{row['assistant']}", max_len=max_len
        )
        if ex is not None:
            train_pairs.append(ex)
            lm_added += 1
    c_counts["lm_prose"] = lm_added

    perm = rng.permutation(len(train_pairs))
    shuffled = [train_pairs[int(i)] for i in perm]
    logger.info("Built Stage C curriculum (%d total samples): %s", len(shuffled), c_counts)
    return shuffled, dev_mc, val_texts


def chunked_causal_lm_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 48,
) -> torch.Tensor:
    """
    Computes exact causal cross-entropy loss by projecting only unmasked hidden states
    through lm_head in small token chunks to keep GTX 1050 4GB VRAM usage < 3.0 GB.
    """
    # Access underlying transformer body + lm_head cleanly across PeftModel / CausalLM
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    outputs = base.model(input_ids=input_ids, use_cache=False)
    hidden = outputs.last_hidden_state[:, :-1, :]  # (1, T-1, H)
    shift_labels = labels[:, 1:]  # (1, T-1)

    mask = shift_labels.reshape(-1) != -100
    if not mask.any():
        return (hidden.sum() * 0.0)

    active_h = hidden.reshape(-1, hidden.shape[-1])[mask]  # (N_act, H)
    active_y = shift_labels.reshape(-1)[mask]  # (N_act,)
    n_act = active_y.shape[0]

    total_loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, n_act, chunk_size):
        end = min(n_act, start + chunk_size)
        logits_chunk = base.lm_head(active_h[start:end]).float()
        ce = F.cross_entropy(logits_chunk, active_y[start:end], reduction="sum")
        total_loss = total_loss + ce
    return total_loss / float(n_act)


@torch.inference_mode()
def eval_dev_mc_and_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    dev_mc: List[Dict[str, str]],
    val_texts: List[str],
    device: str,
) -> Tuple[float, float]:
    """Evaluates non-test dev NLL and greedy 1-token MC accuracy on CUDA/CPU."""
    model.eval()
    val_nll = c.evaluate_texts_nll(model, tokenizer, val_texts[:16], max_len=160, device=device)

    letter_ids = {}
    for ch in ("A", "B", "C", "D"):
        ids = tokenizer.encode(ch, add_special_tokens=False)
        letter_ids[ch] = ids[-1]

    correct = 0
    total = 0
    for row in dev_mc[:40]:
        msgs = [{"role": "user", "content": row["user"]}]
        if getattr(tokenizer, "chat_template", None):
            p_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            p_str = f"User: {row['user']}\nAssistant: "
        enc = tokenizer(p_str, return_tensors="pt", truncation=True, max_length=256)
        inp = enc["input_ids"].to(device)
        logits = model(inp).logits[0, -1, :]
        scores = {ch: float(logits[tid].item()) for ch, tid in letter_ids.items()}
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        if pred == row["assistant"].strip()[:1]:
            correct += 1
        total += 1
    mc_acc = correct / float(max(1, total))
    return val_nll, mc_acc


def merge_lora_into_stageb_cpu(
    stageb_hf_dir: Path,
    adapter_state: Dict[str, torch.Tensor],
    lora_alpha: int,
    lora_r: int,
) -> Tuple[torch.nn.Module, Any]:
    """
    Loads clean Stage B model on CPU in float32 and merges saved LoRA A/B weights
    exact algebraically: W_merged = W_0 + (alpha / r) * (B @ A).
    Guarantees zero leftover PEFT keys and param_diff == 0.
    """
    tok = AutoTokenizer.from_pretrained(str(stageb_hf_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(stageb_hf_dir),
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    scale = float(lora_alpha) / float(lora_r)

    # Group A/B matrices by base parameter path
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in adapter_state.items():
        if ".lora_A." in k:
            base_name = k.replace("base_model.model.", "").replace(".lora_A.default.weight", ".weight").replace(".lora_A.weight", ".weight")
            pairs.setdefault(base_name, {})["A"] = v.float().cpu()
        elif ".lora_B." in k:
            base_name = k.replace("base_model.model.", "").replace(".lora_B.default.weight", ".weight").replace(".lora_B.weight", ".weight")
            pairs.setdefault(base_name, {})["B"] = v.float().cpu()

    sd = dict(model.named_parameters())
    merged_cnt = 0
    with torch.no_grad():
        for pname, ab in pairs.items():
            if pname not in sd:
                raise KeyError(f"Adapter target parameter {pname} missing in Stage B base model")
            if "A" in ab and "B" in ab:
                delta = scale * (ab["B"] @ ab["A"])
                sd[pname].add_(delta)
                merged_cnt += 1
    logger.info("Algebraically merged %d LoRA weight tensors into CPU Stage B float32 base", merged_cnt)
    return model, tok


def patch_gguf_context_2m(gguf_path: Path, model_key: str, target_ctx: int = 2097152) -> None:
    """
    Updates GGUF KV metadata so Ollama 0.40.12 does not clamp num_ctx=2097152 below
    the 172,550-token Quillan-Samurai.md prompt, using non-destructive YaRN scaling
    params when needed alongside existing weights.
    """
    orig_ctx = {"flagship": 131072, "mini": 32768, "dragon": 8192}[model_key]
    tmp_out = gguf_path.with_suffix(".tmp2m.gguf")
    reader = gguf.GGUFReader(str(gguf_path))
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(str(tmp_out), arch=arch)

    skip = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}
    ctx_key = f"{arch}.context_length"
    rope_type_key = f"{arch}.rope.scaling.type"
    rope_factor_key = f"{arch}.rope.scaling.factor"
    rope_orig_key = f"{arch}.rope.scaling.original_context_length"
    yarn_fast_key = f"{arch}.rope.scaling.yarn_beta_fast"
    yarn_slow_key = f"{arch}.rope.scaling.yarn_beta_slow"
    override_keys = {ctx_key, rope_type_key, rope_factor_key, rope_orig_key, yarn_fast_key, yarn_slow_key}

    for key, field in reader.fields.items():
        if key in skip or key in override_keys:
            continue
        c.copy_gguf_field(writer, key, field)

    writer.add_uint32(ctx_key, target_ctx)
    writer.add_string(rope_type_key, "yarn")
    writer.add_float32(rope_factor_key, float(target_ctx) / float(orig_ctx))
    writer.add_uint32(rope_orig_key, int(orig_ctx))
    writer.add_float32(yarn_fast_key, 32.0)
    writer.add_float32(yarn_slow_key, 1.0)

    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    del reader
    gc.collect()
    gguf_path.unlink()
    tmp_out.rename(gguf_path)
    logger.info(
        "Patched GGUF %s -> %s=%d with YaRN (orig=%d factor=%.1f beta=[32,1])",
        gguf_path.name,
        ctx_key,
        target_ctx,
        orig_ctx,
        float(target_ctx) / float(orig_ctx),
    )


def _measure_gguf_ppl_matrix(gguf_path: Path) -> Dict[str, float]:
    """Measures held-out PPL on both -c 128 and -c 512 (2-chunk & 3-chunk) cleanly."""
    return {
        "c128_2c": c.measure_gguf_ppl_cli(gguf_path, chunks=2, ctx=128),
        "c128_3c": c.measure_gguf_ppl_cli(gguf_path, chunks=3, ctx=128),
        "c512_2c": c.measure_gguf_ppl_cli(gguf_path, chunks=2, ctx=512),
    }


def run_stage_c(
    model_key: str,
    max_steps: int = 240,
    lr: float = 8e-5,
    lora_r: int = 16,
    grad_accum: int = 8,
    convert_only: bool = False,
    target_ctx: int = 2097152,
) -> Dict[str, Any]:
    spec = c.MODEL_SPECS[model_key]
    if convert_only:
        if model_key in ("flagship", "mini"):
            c.convert_hf_dir_to_gguf(spec["stagec_hf"], spec["stagec_gguf"], outtype="q8_0")
        pre_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
        patch_gguf_context_2m(spec["stagec_gguf"], model_key, target_ctx=target_ctx)
        post_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
        out = {
            "model": model_key,
            "stagec_gguf": str(spec["stagec_gguf"]),
            "target_ctx": target_ctx,
            "pre_patch_ppl": pre_patch_ppl,
            "post_patch_ppl": post_patch_ppl,
        }
        print("STAGE_C_GGUF_PPL:", json.dumps(out))
        return out

    t0 = time.time()
    stageb_hf: Path = spec["stageb_hf"]
    if not stageb_hf.exists():
        raise FileNotFoundError(f"Stage B checkpoint missing: {stageb_hf}. Run Stage B first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("=== Stage C Consolidation: %s on %s (%s) ===", spec["display"], device, train_dtype)

    tok = AutoTokenizer.from_pretrained(str(stageb_hf), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_pairs, dev_mc, val_texts = build_stage_c_curriculum(tok, c.CLEAN_DATA_PATH, max_len=240)

    model = AutoModelForCausalLM.from_pretrained(
        str(stageb_hf),
        local_files_only=True,
        dtype=train_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False

    lora_alpha = lora_r * 2
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    # Keep trainable LoRA weights in float32 for stable AdamW updates under Pascal fp16 base
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    base_nll, base_mc = eval_dev_mc_and_nll(model, tok, dev_mc, val_texts, device)
    logger.info("Stage B baseline dev -> val_nll=%.4f (PPL=%.3f) dev_mc=%.1f%%",
                base_nll, math.exp(base_nll), base_mc * 100.0)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=lr * 0.1)

    best_score = -1e9
    best_adapter_state: Optional[Dict[str, torch.Tensor]] = None
    best_metrics = {"step": 0, "val_nll": round(base_nll, 5), "dev_mc": round(base_mc, 4)}

    def _capture_adapter() -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}

    best_adapter_state = _capture_adapter()
    best_score = (base_mc * 2.0) - base_nll

    model.train()
    opt.zero_grad(set_to_none=True)
    idx_cursor = 0
    for step in range(1, max_steps + 1):
        accum_loss = 0.0
        for _ in range(grad_accum):
            inp, lbl = train_pairs[idx_cursor % len(train_pairs)]
            idx_cursor += 1
            loss = chunked_causal_lm_loss(model, inp.to(device), lbl.to(device), chunk_size=32)
            if step == 1 and not loss.requires_grad:
                raise RuntimeError("Stage C gradient probe failed: loss.requires_grad is False")
            (loss / float(grad_accum)).backward()
            accum_loss += float(loss.item()) / float(grad_accum)
        if step == 1:
            lora_grad_norm = sum(
                float(p.grad.float().norm().item() ** 2)
                for p in opt.param_groups[0]["params"]
                if p.grad is not None
            ) ** 0.5
            logger.info("Step 1 LoRA grad norm check: %.6f", lora_grad_norm)
            if not (lora_grad_norm > 0.0):
                raise RuntimeError("Stage C gradient probe failed: zero LoRA gradient norm on step 1")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % 30 == 0 or step == max_steps:
            val_nll, dev_mc_acc = eval_dev_mc_and_nll(model, tok, dev_mc, val_texts, device)
            # Prioritize lower NLL + higher MC accuracy with strict Lyapunov PPL gate
            score = (dev_mc_acc * 2.0) - val_nll
            logger.info("Step %3d/%d | train_loss=%.4f | dev_nll=%.4f (PPL=%.3f) | dev_mc=%.1f%%",
                        step, max_steps, accum_loss, val_nll, math.exp(val_nll), dev_mc_acc * 100.0)
            if val_nll <= base_nll and score > best_score:
                best_score = score
                best_adapter_state = _capture_adapter()
                best_metrics = {"step": step, "val_nll": round(val_nll, 5), "dev_mc": round(dev_mc_acc, 4)}
            model.train()

    # Release CUDA memory completely before CPU merge & export
    del model, opt, sched, train_pairs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    assert best_adapter_state is not None
    merged_model, merged_tok = merge_lora_into_stageb_cpu(
        stageb_hf, best_adapter_state, lora_alpha=lora_alpha, lora_r=lora_r
    )
    stageb_report_path = c.EXPORT_ROOT / f"v23_stageb_{model_key}_report.json"
    expected_params = None
    if stageb_report_path.exists():
        expected_params = json.loads(stageb_report_path.read_text(encoding="utf-8"))["param_audit"]["total_params"]
    merge_audit = c.audit_model_integrity(merged_model, expected_params=expected_params)
    logger.info("Stage C merged CPU audit passed: %s", merge_audit)

    report: Dict[str, Any] = {
        "model": model_key,
        "elapsed_s": round(time.time() - t0, 2),
        "param_audit": merge_audit,
        "base_dev_nll": round(base_nll, 5),
        "base_dev_mc": round(base_mc, 4),
        "best_dev": best_metrics,
        "stagec_hf": str(spec["stagec_hf"]),
        "target_ctx": target_ctx,
    }
    # Export Dragon directly to GGUF + save HF; Flagship/Mini save bfloat16 HF for standalone convert
    if model_key == "dragon":
        c.export_dragon_gguf_from_hf(merged_model, spec["stagec_gguf"])
        pre_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
        patch_gguf_context_2m(spec["stagec_gguf"], model_key, target_ctx=target_ctx)
        post_patch_ppl = _measure_gguf_ppl_matrix(spec["stagec_gguf"])
        report["stagec_gguf"] = str(spec["stagec_gguf"])
        report["pre_patch_ppl"] = pre_patch_ppl
        report["post_patch_ppl"] = post_patch_ppl

    merged_model.to(torch.bfloat16)
    c.save_hf_checkpoint(merged_model, merged_tok, spec["stagec_hf"])
    logger.info("Saved Stage C HF checkpoint -> %s", spec["stagec_hf"])

    del merged_model, best_adapter_state
    gc.collect()

    rep_path = c.EXPORT_ROOT / f"v23_stagec_{model_key}_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("STAGE_C_DONE:", json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP v23 Stage C Consolidation")
    parser.add_argument("--model", required=True, choices=["flagship", "mini", "dragon"])
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--convert-only", action="store_true")
    args = parser.parse_args()
    run_stage_c(
        model_key=args.model,
        max_steps=args.max_steps,
        lr=args.lr,
        lora_r=args.lora_r,
        grad_accum=args.grad_accum,
        convert_only=args.convert_only,
    )


if __name__ == "__main__":
    main()
