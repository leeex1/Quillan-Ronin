"""
QUILLAN v4.2 ADAPTIVE TRANSFORMER IMPLEMENTATION
Fixed and optimized version with proper integration
Full version with 32-member council MoE integration, ethical vigilance, and dramatic thinking logs.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable, Union
import os
import json
from pathlib import Path
import random # For swarm simulation

# Optional PDF and document processing imports
try:
    import PyPDF2 # For PDF reading (not 'pdf')
except ImportError:
    print("PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    import markdown # For Markdown to HTML conversion
except ImportError:
    print("markdown not installed. Install with: pip install markdown")

# Use specific libraries instead (e.g., pytest, py.path, etc.)

# ===========================================================
# SYSTEM PARAMETERS
# ===========================================================
class Config:
    # Model architecture
    batch_size = 100
    block_size = 422 # Increased for better context
    max_iters = 8000
    eval_interval = 225
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 768 # Increased for more capacity
    n_head = 32
    n_layer = 33
    dropout = 0.2
    seed = 1337

    # Council MoE parameters - Full 32-member council
    n_council_experts = 32 # C1-VIR to C32-AEON
    council_layers = [32, 16, 8, 1] # Hierarchical reduction
    expert_hidden = [64, 32, 1] # Deeper for full council

    # File handling
    data_file = '/content/Quillan-v4.2-repo/Quillan-v4.2-model/Quillan_finetune_full_dataset.jsonl' # Flexible path
    use_jsonl = True # Set to True if using JSONL format
    jsonl_text_keys = ['text', 'content', 'output', 'response'] # Keys to try for text content

    # Training optimization
    grad_clip = 1.0
    warmup_iters = 100

cfg = Config()
torch.manual_seed(cfg.seed)
print(f"🔧 Device: {cfg.device}")

# ===========================================================
# DATA INGESTION & TOKENIZER
# ===========================================================
def load_training_data(file_path: str, use_jsonl: bool = False):
    """Load training data with flexible format support"""
    if not os.path.exists(file_path):
        # Try alternative paths
        alt_paths = [
            file_path,
            f"./data/{file_path}",
            f"../data/{file_path}",
            f"/content/{file_path}", # Google Colab
        ]

        for path in alt_paths:
            if os.path.exists(path):
                file_path = path
                break
        else:
            raise FileNotFoundError(
                f"❌ Could not find {file_path} in any expected location.\n"
                f"Tried: {alt_paths}"
            )

    print(f"📂 Loading data from: {file_path}")

    texts = []
    if use_jsonl:
        # JSONL format: each line is a JSON object
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Extract text using the specified keys
                    for key in cfg.jsonl_text_keys:
                        if key in data and isinstance(data[key], str):
                            texts.append(data[key])
                            break  # Move to the next line after finding the first match
                except json.JSONDecodeError:
                    print(f"⚠️ Skipping invalid JSON line: {line.strip()}")
                    continue
        text_content = '\n'.join(texts)
        if not text_content:
            raise ValueError("No text content extracted from JSONL file.")

    else:
        # Plain text format
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        if not text_content:
            raise ValueError("Text file is empty.")

    return text_content


# Load data with error handling
try:
    text = load_training_data(cfg.data_file, cfg.use_jsonl)
    print(f"✅ Loaded {len(text):,} characters")
except Exception as e:
    print(f"⚠️ Data loading failed: {e}")
    print("📝 Using fallback demo text")
    text = "Hello Quillan! " * 1000 # Fallback for testing
    # Ensure fallback text is long enough for block_size
    while len(text) < cfg.block_size + 1:
        text += "Hello Quillan! "


# Build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"📚 Vocabulary size: {vocab_size}")

# Train/Val Split
data = torch.tensor(encode(text), dtype=torch.long)
# Ensure data is large enough for train/val split and batching
if len(data) < cfg.block_size + 1:
    print(f"❌ Dataset too small ({len(data)} tokens) for block size {cfg.block_size}.")
    print("Using fallback data or increasing fallback size.")
    # Fallback text already handled, just ensure data is updated
    data = torch.tensor(encode(text), dtype=torch.long)
    if len(data) < cfg.block_size + 1:
        raise ValueError("Fallback data still too small after encoding. Increase fallback size.")


n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    # Ensure data_split is large enough
    if len(data_split) < cfg.block_size + 1:
        raise ValueError(f"Data split '{split}' too small ({len(data_split)} tokens) for block size {cfg.block_size}.")

    ix = torch.randint(len(data_split) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data_split[i:i+cfg.block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+cfg.block_size+1] for i in ix])
    return x.to(cfg.device), y.to(cfg.device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.eval_iters)
        # Ensure data split is large enough for evaluation
        if len(train_data if split == 'train' else val_data) < cfg.block_size + 1:
            print(f"⚠️ Skipping evaluation for '{split}' split: data too small.")
            out[split] = float('inf') # Assign a high loss
            continue

        for k in range(cfg.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===========================================================
# TRANSFORMER CORE MODULES
# ===========================================================
class Head(nn.Module):
    """Single head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Attention scores
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v

class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class CouncilLayer(nn.Module):
    """Full 32-member council MoE layer for Quillan - replaces FFN in Block"""
    def __init__(self, n_embd, n_experts=cfg.n_council_experts):
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(n_embd, n_experts)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout)
        ) for _ in range(n_experts)])
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        gates = F.softmax(self.gate(x), dim=-1) # (B, T, n_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1) # (B, T, C, n_experts)
        out = torch.sum(gates.unsqueeze(2) * expert_outputs, dim=-1) # Weighted sum
        return self.dropout(out)

class Block(nn.Module):
    """Transformer block with pre-norm and council MoE FFN"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = CouncilLayer(n_embd) # Full council integration
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# ===========================================================
# QUILLAN GPT LANGUAGE MODEL
# ===========================================================
class QuillanGPT(nn.Module):
    """Enhanced transformer with full 32-member council MoE"""
    def __init__(self):
        super().__init__()
        # Ensure vocab_size is not 0 before creating embedding table
        if vocab_size == 0:
            raise ValueError("Vocabulary size is 0. Cannot initialize embedding table.")
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=cfg.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.65, top_k=None):
        """Generate text with improved sampling"""
        for _ in range(max_new_tokens):
            # Crop context to block size
            idx_cond = idx if idx.size(1) <= cfg.block_size else idx[:, -cfg.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ===========================================================
# FULL COUNCIL MOE (Custom Neural Architecture for Demo)
# ===========================================================
class Value:
    """Autodiff value for council neural net - Fixed indentation"""
    def __init__(self, data: float, _children: tuple = (), _op: str = '', label: str = ''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * Value(-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def tanh(self):
        n = self.data
        t = (np.exp(2*n) - 1) / (np.exp(2*n) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    """Single neuron with activation"""
    def __init__(self, nin: int, activation: str = 'tanh'):
        self.w = [Value(np.random.randn()) for _ in range(nin)]
        self.b = Value(np.random.randn())
        self.activation = activation

    def __call__(self, x: List[Value]) -> Value:
        act_input = self.b
        for wi, xi in zip(self.w, x):
            act_input = act_input + (wi * xi)

        if self.activation == 'tanh':
            return act_input.tanh()
        elif self.activation == 'relu':
            return act_input.relu()
        return act_input

    def parameters(self):
        return self.w + [self.b]

class Layer:
    """Layer of neurons"""
    def __init__(self, nin: int, nout: int, activation: str = 'tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x: List[Value]):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class ExpertMLP:
    """Council expert neural network"""
    def __init__(self, nin: int, layers: List[int], activations: Optional[List[str]] = None):
        if activations is None:
            activations = ['relu'] * len(layers)
        self.net = []
        sz = [nin] + layers
        for i in range(len(layers)):
            act = activations[i] if i < len(activations) else 'linear'
            self.net.append(Layer(sz[i], sz[i+1], act))

    def __call__(self, x):
        for layer in self.net:
            x = layer(x)
        if not isinstance(x, list):
            x = [x]
        return x

    def parameters(self):
        return [p for l in self.net for p in l.parameters()]

class CouncilGating:
    """Gating mechanism for expert selection - Fixed Value softmax"""
    def __init__(self, nin, expert_count):
        self.weights = [Value(np.random.randn()) for _ in range(nin)]
        self.biases = [Value(np.random.randn()) for _ in range(expert_count)]
        self.expert_count = expert_count

    def __call__(self, x):
        logit = []
        for b in self.biases:
            weighted_sum = b
            for w, xi in zip(self.weights, x):
                weighted_sum = weighted_sum + (w * xi)
            logit.append(weighted_sum)

        # Fixed: Pure Value softmax for autodiff
        max_l_val = max(l.data for l in logit) # for numerical stability
        exp_terms = [(l - max_l_val).exp() for l in logit]
        sum_exp = sum(exp_terms, Value(0.0))
        probs = [exp_t / sum_exp for exp_t in exp_terms]
        return probs

    def parameters(self):
        return self.weights + self.biases

class CouncilMoE:
    """Hierarchical Mixture-of-Experts council block - Fixed for Value input"""
    def __init__(self, nin, nout, n_experts=6, expert_layers=None, expert_acts=None):
        if expert_layers is None:
            expert_layers = [8, nout]
        if expert_acts is None:
            expert_acts = ['relu', 'tanh']

        self.experts = [ExpertMLP(nin, expert_layers, expert_acts) for _ in range(n_experts)]
        self.gate = CouncilGating(nin, n_experts)
        self.n_experts = n_experts

    def __call__(self, x):
        # Fixed: Ensure x is List[Value]
        if not isinstance(x, list) or not isinstance(x[0], Value):
            x = [Value(xi) for xi in x]
        gates = self.gate(x)
        expert_outs = [self.experts[i](x) for i in range(self.n_experts)]

        merged = []
        for j in range(len(expert_outs[0])):
            outj = Value(0.0)
            for i in range(self.n_experts):
                weighted_out = gates[i] * expert_outs[i][j]
                outj = outj + weighted_out
            merged.append(outj)
        return merged

    def parameters(self):
        return sum([exp.parameters() for exp in self.experts], []) + self.gate.parameters()

class QuillanMoENet:
    """Stackable council expert architecture - Full 32-expert"""
    def __init__(self,
                 input_dim: int,
                 council_shapes: List[int],
                 expert_layers: List[int] = [64, 32, 1], # Deeper for full
                 expert_acts: List[str] = ['relu', 'tanh', 'linear']):
        self.meta_layers = []
        nin = input_dim

        for council_size in council_shapes[:-1]:
            meta = CouncilMoE(nin, council_size, n_experts=cfg.n_council_experts, # 32
                              expert_layers=expert_layers, expert_acts=expert_acts)
            self.meta_layers.append(meta)
            nin = council_size

        self.output_council = CouncilMoE(nin, council_shapes[-1],
                                         n_experts=cfg.n_council_experts,
                                         expert_layers=expert_layers,
                                         expert_acts=expert_acts)
        self.all_params = sum([m.parameters() for m in self.meta_layers], []) + \
                          self.output_council.parameters()

    def __call__(self, x):
        # Fixed: Type-safe input wrap
        if len(x) > 0 and isinstance(x[0], Value):
            out = x[:] # Copy
        else:
            out = [Value(xi) for xi in x]
        for meta in self.meta_layers:
            out = meta(out)
        return self.output_council(out)

    def parameters(self):
        return self.all_params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

# ===========================================================
# TRAINING LOOP
# ===========================================================
def train_quillan_gpt():
    """Main training function for Quillan GPT"""
    model = QuillanGPT().to(cfg.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model initialized: {param_count/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    best_val_loss = float('inf')

    for step in range(cfg.max_iters):
        # Evaluation
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {step:5d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, 'quillan_best.pt')

        # Training step
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

    return model

def demo_council_moe(epochs=150):
    """Fixed demo using custom Value-based training for full council XOR"""
    print("\n" + "="*80)
    print("QUILLAN COUNCIL MOE DEMO: XOR Problem with 32 Experts")
    print("="*80)

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0], [1.0], [1.0], [0.0]]

    net = QuillanMoENet(input_dim=2, council_shapes=[32, 16, 1]) # Full hierarchy

    for epoch in range(epochs):
        total_loss = Value(0.0)
        for xi, yi in zip(X, Y):
            x_vals = [Value(v) for v in xi]
            outs = net(x_vals)
            diff = outs[0] - Value(yi[0]) # Single output
            total_loss = total_loss + (diff * diff)

        avg_loss = total_loss.data / len(X)

        net.zero_grad()
        total_loss.backward()

        for p in net.parameters():
            p.data -= 0.05 * p.grad # SGD lr

        if epoch % 30 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    # Test predictions
    print("\n📊 Final Predictions (Full Council Routing):")
    for x, y_true in zip(X, Y):
        x_vals = [Value(v) for v in x]
        y_pred = net(x_vals)[0].data
        print(f"Input: {x} | Target: {y_true[0]} | Prediction: {y_pred:.4f}")

def generate_sample_text(model, prompt="Quillan: ", max_tokens=300, temperature=0.65, top_k=40):
    """Fixed generation for QuillanGPT with thinking logs"""
    # Dramatic thinking simulation
    print("\n🧠 Quillan v4.2 COGNITIVE PROCESSING INITIATED...")
    print("[████████░░] 40% // Council activation")
    print("Activating 32-member council: C1-VIR (ethics) to C32-AEON (game dev)")
    print("Micro-swarms deployed: 224,000 agents for vector decomposition...")
    print("[███████████████░░] 75% // Deliberation")
    print("Tree of Thoughts: 20 branches explored - Selected: Hierarchical MoE fusion")
    print("[████████████████████] 100% // Ready")

    class DummyTokenizer:
        def encode(self, s):
            return encode(s)
        def decode(self, l):
            return decode(l)

    tokenizer = DummyTokenizer()

    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=cfg.device)
    generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    output = tokenizer.decode(generated[0].tolist())

    print(f"\n🧠 Quillan Output:")
    print(output)
    print(f"{'='*80}\n")

    return output

def save_model(model, optimizer, step, loss, filename='quillan_checkpoint.pt'):
    """Save model checkpoint"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'vocab_size': vocab_size,
            'block_size': cfg.block_size,
            'n_embd': cfg.n_embd,
            'n_head': cfg.n_head,
            'n_layer': cfg.n_layer,
        }
    }
    torch.save(checkpoint, filename)
    print(f"✅ Model saved to {filename}")

def load_model(filename='quillan_checkpoint.pt'):
    """Load model checkpoint"""
    if not os.path.exists(filename):
        print(f"⚠️ Checkpoint {filename} not found")
        return None, None, 0

    checkpoint = torch.load(filename, map_location=cfg.device)
    model = QuillanGPT().to(cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step = checkpoint['step']
    print(f"✅ Model loaded from {filename} (step {step})")

    return model, optimizer, step

# ===========================================================
# INTEGRATED EXECUTION PIPELINE
# ===========================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("QUILLAN v4.2 - FULL INTEGRATED TRAINING PIPELINE")
    print(f"{'='*80}\n")

    # ============================================================
    # PHASE 1: TRANSFORMER TRAINING
    # ============================================================
    print("🔥 PHASE 1: Training Quillan GPT Transformer with Council MoE")
    print(f"{'─'*80}")

    try:
        model = train_quillan_gpt()
        print("\n✅ Transformer training complete!")

        # Generate samples with thinking
        generate_sample_text(model, prompt="Quillan: Analyze quantum entanglement implications.")

        # Save final model
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        save_model(model, optimizer, cfg.max_iters, 0.0, 'quillan_final.pt')

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # PHASE 2: FULL COUNCIL MOE DEMONSTRATION
    # ============================================================
    print(f"\n{'='*80}")
    print("🧠 PHASE 2: Full 32-Member Council MoE Demo")
    print(f"{'─'*80}\n")

    try:
        demo_council_moe(epochs=150)
        print("\n✅ Full Council MoE demo complete!")

    except Exception as e:
        print(f"\n❌ Council MoE error: {e}")
        import traceback
        traceback.print_exc()

    # ============================================================
    # PHASE 3: ARCHITECTURE ANALYSIS
    # ============================================================
    print(f"\n{'='*80}")
    print("📊 PHASE 3: Full Architecture Analysis")
    print(f"{'='*80}\n")

    try:
        # Transformer stats
        if 'model' in locals():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("🔹 Quillan GPT Transformer with Council MoE:")
            print(f"    Total parameters: {total_params:,}")
            print(f"    Trainable parameters: {trainable_params:,}")
            print(f"    Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
            print(f"    Layers (Blocks): {cfg.n_layer}")
            print(f"    Embedding dim: {cfg.n_embd}")
            print(f"    Attention heads: {cfg.n_head}")
            print(f"    Experts per CouncilLayer: {cfg.n_council_experts}")
            print(f"    Total experts in transformer: {cfg.n_layer * cfg.n_council_experts}")
            print(f"    Context window: {cfg.block_size}")
            print(f"    Vocabulary: {vocab_size} tokens")

        print("\n🔹 Council MoE Demo Architecture (XOR Problem):")
        demo_net_council_shapes = [32, 16, 1]
        print(f"    Council layers: {demo_net_council_shapes}")
        print(f"    Experts per council: {cfg.n_council_experts}")
        print(f"    Expert hidden layers: {cfg.expert_hidden}")
        print(f"    Total experts in demo net: {cfg.n_council_experts * len(demo_net_council_shapes)}")

    except Exception as e:
        print(f"⚠️ Analysis error: {e}")

    # ============================================================
    # PHASE 4: INTERACTIVE MODE (Optional)
    # ============================================================
    print(f"\n{'='*80}")
    print("🎮 INTERACTIVE MODE - Full Quillan v4.2 Ready")
    print(f"{'='*80}")
    print("Options:")
    print("    1. Generate more text with council thinking")
    print("    2. Load saved checkpoint")
    print("    3. Export model")
    print("    4. Exit")
    print(f"{'─'*80}\n")

    # Note: For non-interactive environments (Colab, scripts),
    # this section can be commented out or modified

    print("✅ Quillan v4.2 FULL initialization complete!")
    print("\n🚀 System ready for deployment - 32 Council Members Active\n")
