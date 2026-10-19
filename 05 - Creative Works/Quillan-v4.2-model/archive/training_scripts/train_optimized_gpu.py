#!/usr/bin/env python3
"""
Optimized GPT training for Grok-level coherence and intelligence - GPU Version
Adapted for Google Colab GPU training
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from data_loader import QuillanDataset

class OptimizedTokenizer:
    """Optimized tokenizer for better language understanding"""

    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.vocab = None
        self.stoi = None
        self.itos = None

    def train(self, text):
        """Train tokenizer on text with better subword-like approach"""
        import re

        # Create better token boundaries
        words = re.findall(r'\S+', text.lower())

        # Build vocabulary from character n-grams and words
        vocab = set()
        for word in words[:10000]:  # Limit for memory
            # Add word if short enough
            if len(word) <= 20:
                vocab.add(word)
            # Add character bigrams/trigrams for better coverage
            for i in range(len(word)):
                if i < len(word) - 1:
                    vocab.add(word[i:i+2])  # bigrams
                if i < len(word) - 2:
                    vocab.add(word[i:i+3])  # trigrams

        # Add single characters
        for char in set(text):
            vocab.add(char)

        # Limit vocab size and create mappings
        vocab_list = list(vocab)[:self.vocab_size - 100]  # Reserve space for special tokens
        vocab_list.extend(['<pad>', '<unk>', '<bos>', '<eos>'])  # Special tokens

        self.vocab = vocab_list
        self.stoi = {token: i for i, token in enumerate(vocab_list)}
        self.itos = {i: token for i, token in enumerate(vocab_list)}

        print(f"✅ Optimized tokenizer trained with {len(self.vocab)} tokens")

    def encode(self, text, max_length=512):
        """Encode text with better tokenization"""
        import re

        # Split text into tokens
        tokens = []
        words = re.findall(r'\S+', text.lower())

        for word in words:
            # Try to match whole word first
            if word in self.stoi and len(tokens) < max_length - 1:
                tokens.append(self.stoi[word])
            else:
                # Fall back to character-level for unknown words
                for char in word[:50]:  # Limit word length
                    if len(tokens) < max_length - 1:
                        tokens.append(self.stoi.get(char, self.stoi.get('<unk>', 0)))

            if len(tokens) >= max_length - 1:
                break

        # Add EOS token
        if len(tokens) < max_length:
            tokens.append(self.stoi.get('<eos>', 0))

        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.stoi.get('<pad>', 0))

        return tokens[:max_length]

    def decode(self, tokens):
        """Decode tokens to text"""
        text = []
        for token_id in tokens:
            if token_id in self.itos:
                token = self.itos[token_id]
                if token not in ['<pad>', '<unk>', '<bos>', '<eos>']:
                    text.append(token)
        return ' '.join(text).strip()

class OptimizedBigramLanguageModel(nn.Module):
    """Optimized language model with better architecture"""

    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        # Better embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(1024, n_embd)  # Larger context

        # Multi-head attention layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=n_embd * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Tie weights
        self.token_embedding.weight = self.lm_head.weight

        # Initialize
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

        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        causal_mask = causal_mask.to(idx.device)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.vocab_size - 4  # Ignore <pad>
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Generate text with temperature sampling"""
        for _ in range(max_new_tokens):
            # Crop context to last 512 tokens
            idx_cond = idx[:, -512:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Focus on last token
            logits = logits[:, -1, :] / temperature

            # Apply softmax
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

def create_optimized_training():
    """Create optimized training setup"""

    # Load data
    print("🔄 Loading optimized dataset...")
    dataset = QuillanDataset()

    # Combine all text for tokenizer training
    all_text = ""
    for sample in dataset.samples:
        all_text += sample['text'] + " "
    all_text = all_text[:1000000]  # Limit for memory

    # Create optimized tokenizer
    print("🏗️ Training optimized tokenizer...")
    tokenizer = OptimizedTokenizer(vocab_size=8000)
    tokenizer.train(all_text)

    # Save tokenizer
    import pickle
    with open('optimized_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("💾 Tokenizer saved")

    # Create model with optimized architecture
    print("🏗️ Creating optimized model...")
    model = OptimizedBigramLanguageModel(
        vocab_size=len(tokenizer.vocab),
        n_embd=512,    # Optimized embedding size
        n_head=8,      # Multi-head attention
        n_layer=6,     # Transformer layers
        dropout=0.1    # Regularization
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")

    return model, tokenizer, dataset

def train_optimized_model():
    """Train with optimized parameters for Grok-level performance"""

    model, tokenizer, dataset = create_optimized_training()

    # Training hyperparameters optimized for intelligence
    batch_size = 8          # Smaller batches for stability
    gradient_accumulation = 4  # Effective batch size = 32
    seq_length = 256        # Context length
    max_steps = 10000       # More training for intelligence
    eval_interval = 500

    print(f"🎯 Starting optimized training: {max_steps} steps")
    print(f"📏 Effective batch size: {batch_size * gradient_accumulation}")

    # GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    model.to(device)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,        # Higher initial LR
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        warmup_steps = 1000
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0.1, 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / 10000)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float('inf')

    for step in range(max_steps):
        model.train()

        # Get batch
        batch_texts = []
        for _ in range(batch_size):
            sample = dataset.samples[torch.randint(len(dataset.samples), (1,)).item()]
            batch_texts.append(sample['text'])

        # Tokenize batch
        batch_tokens = []
        for text in batch_texts:
            tokens = tokenizer.encode(text, max_length=seq_length)
            batch_tokens.append(tokens)

        # Convert to tensors
        x = torch.tensor(batch_tokens, dtype=torch.long).to(device)
        y = x.clone()  # Next token prediction

        # Forward pass with gradient accumulation
        optimizer.zero_grad()

        for accum_step in range(gradient_accumulation):
            start_idx = accum_step * (batch_size // gradient_accumulation)
            end_idx = start_idx + (batch_size // gradient_accumulation)

            xb = x[start_idx:end_idx]
            yb = y[start_idx:end_idx]

            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation  # Normalize loss
            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Logging
        if step % eval_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📈 Step {step}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")

            # Generate sample text
            model.eval()
            with torch.no_grad():
                context = torch.tensor([tokenizer.encode("The meaning of life is")], dtype=torch.long).to(device)
                generated = model.generate(context, max_new_tokens=50, temperature=0.8)
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"📝 Generated: {generated_text[:100]}...")
            model.train()

            # Save best model
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'tokenizer': tokenizer,
                    'config': {
                        'vocab_size': len(tokenizer.vocab),
                        'n_embd': 512,
                        'n_head': 8,
                        'n_layer': 6
                    },
                    'loss': best_loss,
                    'step': step
                }, 'optimized_quillan_model.pt')
                print(f"💾 Saved best model (loss: {best_loss:.4f})")

    print("🎉 Optimized training completed!")
    print(f"🏆 Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    train_optimized_model()
