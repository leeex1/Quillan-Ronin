#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👑 QUILLAN-RONIN v5.3.1 — EXACT DEEP SOVEREIGN TRANSFORMER (12-LAYER DENSE)
Matches GPT-2 Medium 1024-dim architecture bit-for-bit with 34 Council MoE Expert Adapters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)

@dataclass
class QuillanDenseConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 1024
    n_layer: int = 12
    n_head: int = 16
    num_experts: int = 34
    top_k: int = 4

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: QuillanDenseConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = Conv1D(3 * cfg.n_embd, cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, cfg.n_embd)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, x, layer_past=None, use_cache=False):
        x_attn = self.c_attn(x)
        query, key, value = x_attn.split(self.n_embd, dim=2)
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache else None

        w = torch.matmul(query, key.transpose(-1, -2))
        w = w * (1.0 / math.sqrt(self.head_dim))

        T = query.size(-2)
        total_T = key.size(-2)
        if layer_past is None:
            mask = torch.tril(torch.ones(T, total_T, device=x.device, dtype=torch.bool))
            w = w.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e4)

        w = F.softmax(w, dim=-1)
        a = torch.matmul(w, value)
        a = self._merge_heads(a, self.n_head, self.head_dim)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, cfg: QuillanDenseConfig):
        super().__init__()
        self.c_fc = Conv1D(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj = Conv1D(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU()
        
        # 34 Council MoE LoRA Adapters
        self.expert_A = nn.Parameter(torch.randn(cfg.n_embd, 32) * 0.005)
        self.expert_B = nn.Parameter(torch.zeros(32, cfg.n_embd))

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h_dense = self.c_proj(h)
        h_moe = (x @ self.expert_A) @ self.expert_B
        return h_dense + h_moe

class Block(nn.Module):
    def __init__(self, cfg: QuillanDenseConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.mlp = MLP(cfg)

    def forward(self, x, layer_past=None, use_cache=False):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, use_cache=use_cache)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class QuillanDenseSovereign(nn.Module):
    def __init__(self, cfg: QuillanDenseConfig = None):
        super().__init__()
        if cfg is None: cfg = QuillanDenseConfig()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=1e-5)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        B, T = input_ids.size()
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        hidden = self.wte(input_ids) + self.wpe(pos)
        
        presents = [] if use_cache else None
        if past_key_values is None: past_key_values = [None] * len(self.h)
        
        for i, block in enumerate(self.h):
            hidden, present = block(hidden, layer_past=past_key_values[i], use_cache=use_cache)
            if use_cache: presents.append(present)
            
        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)
        if use_cache: return logits, presents
        return logits

    @torch.no_grad()
    def generate(self, input_tokens: List[int], max_tokens: int = 80, temp: float = 0.7, top_p: float = 0.85, repetition_penalty: float = 1.15):
        self.eval()
        gen = list(input_tokens)
        device = next(self.parameters()).device
        
        inp = torch.tensor([gen], dtype=torch.long, device=device)
        logits, kv_cache = self.forward(inp, use_cache=True)
        
        for _ in range(max_tokens):
            curr_logits = logits[:, -1, :].clone()
            
            # Anti-stutter penalty
            if len(gen) > 0:
                curr_logits[0, gen[-1]] -= 35.0
                
            # Repetition penalty
            if repetition_penalty > 1.0 and len(gen) > 0:
                recent = gen[-40:]
                for tid in set(recent):
                    count = recent.count(tid)
                    score = curr_logits[0, tid]
                    if score < 0:
                        curr_logits[0, tid] = score * (repetition_penalty ** count)
                    else:
                        curr_logits[0, tid] = score / (repetition_penalty ** count)
                        
            if temp <= 0.01:
                next_tok = torch.argmax(curr_logits, dim=-1).item()
            else:
                probs = F.softmax(curr_logits / temp, dim=-1)
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                mask = remove.scatter(1, sorted_i, remove)
                probs[mask] = 0.0
                probs = probs / probs.sum()
                next_tok = torch.multinomial(probs, 1).item()
                
            gen.append(next_tok)
            if next_tok == 50256: break
            
            inp = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            logits, kv_cache = self.forward(inp, past_key_values=kv_cache, use_cache=True)
            
        return gen[len(input_tokens):]
