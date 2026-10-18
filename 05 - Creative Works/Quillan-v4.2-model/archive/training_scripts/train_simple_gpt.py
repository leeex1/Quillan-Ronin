#!/usr/bin/env python3
"""
Proper GPT training implementation based on the notebook
"""

import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F

def load_quillan_data():
    """Load and preprocess Quillan data like in the notebook"""
    print("🔄 Loading Quillan dataset...")

    # Load JSONL data
    jsonl_samples = []
    jsonl_path = "Quillan_finetune_full_dataset.jsonl"
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Extract text content (same logic as notebook)
                    def find_text_in_json(data):
                        texts = []
                        if isinstance(data, str):
                            texts.append(data)
                        elif isinstance(data, dict):
                            for key, value in data.items():
                                texts.extend(find_text_in_json(value))
                        elif isinstance(data, list):
                            for item in data:
                                texts.extend(find_text_in_json(item))
                        return texts

                    jsonl_samples.extend(find_text_in_json(data))
                except:
                    continue

        print(f"✅ Loaded {len(jsonl_samples)} JSONL samples")

    # Load text files from directories
    text_files = []
    dirs_to_check = ["Songs Lyrics", "Quillan Knowledge files"]

    for dir_name in dirs_to_check:
        if os.path.exists(dir_name):
            for filename in os.listdir(dir_name):
                filepath = os.path.join(dir_name, filename)
                try:
                    if filename.lower().endswith(('.md', '.txt')):
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content.strip()) > 10:
                                text_files.append(content)
                except:
                    continue

    print(f"✅ Loaded {len(text_files)} text files")

    # Combine all text
    full_corpus = "\n\n".join(jsonl_samples + text_files)
    print(f"📊 Total corpus length: {len(full_corpus)} characters")

    return full_corpus

def create_tokenizer(text):
    """Create character-level tokenizer like in the notebook"""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"📝 Vocab size: {vocab_size}")
    print(f"🔤 Characters: {''.join(chars[:50])}...")

    return encode, decode, vocab_size

def prepare_data(text, encode):
    """Prepare data tensor like in the notebook"""
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"📊 Data shape: {data.shape}")

    # Train/val split (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"🎯 Train data: {train_data.shape}")
    print(f"🎯 Val data: {val_data.shape}")

    return train_data, val_data

class BigramLanguageModel(nn.Module):
    """Simple Bigram model from the notebook"""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, decode_func):
        """Generate text"""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return decode_func(idx[0].tolist())

def get_batch(split, train_data, val_data, block_size, batch_size):
    """Get batch like in the notebook"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train_model():
    """Train a simple model properly"""
    print("🚀 Starting proper GPT training...")

    # Load and prepare data
    text = load_quillan_data()
    encode, decode, vocab_size = create_tokenizer(text)
    train_data, val_data = prepare_data(text, encode)

    # Model and training setup
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training hyperparameters (from notebook)
    batch_size = 32
    block_size = 256
    max_steps = 15000  # Extended training for sub-1 loss
    eval_interval = 500

    print(f"🎯 Training: {max_steps} steps, batch_size={batch_size}, block_size={block_size}")

    for step in range(max_steps):
        # Get batch
        xb, yb = get_batch('train', train_data, val_data, block_size, batch_size)

        # Forward and backward
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate occasionally
        if step % eval_interval == 0:
            # Validation loss
            model.eval()
            with torch.no_grad():
                xb_val, yb_val = get_batch('val', train_data, val_data, block_size, batch_size)
                _, val_loss = model(xb_val, yb_val)

            print(f"📈 Step {step}: Train loss = {loss.item():.4f}, Val loss = {val_loss.item():.4f}")

            # Generate sample text
            context = torch.zeros((1, 1), dtype=torch.long)
            generated = model.generate(context, 100, decode)
            print(f"📝 Generated: {generated[:100]}...")
            print("-" * 50)

            model.train()

    # Save the trained model (without lambda functions)
    model_data = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'block_size': block_size,
        'chars': ''.join(sorted(list(text.replace('\n', ' ').replace('\r', ''))))[:vocab_size],  # Save charset
        'config': {
            'vocab_size': vocab_size,
            'block_size': block_size
        }
    }
    torch.save(model_data, 'simple_gpt_model.pt')
    print("💾 Model saved as simple_gpt_model.pt")
    return model, encode, decode

if __name__ == "__main__":
    train_model()
