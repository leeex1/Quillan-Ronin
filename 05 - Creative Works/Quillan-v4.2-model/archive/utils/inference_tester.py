#!/usr/bin/env python3
"""
Inference Testing Script for Trained Quillan Model
Tests the trained model on provided content to evaluate performance
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import json
import random
import pickle
import time
from collections import defaultdict

class OptimizedBigramLanguageModel(nn.Module):
    """Optimized language model matching training architecture"""

    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=6, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(1024, n_embd)

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
        self.token_embedding.weight = self.lm_head.weight

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
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(idx.device)

        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=self.vocab_size - 4
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        """Generate text with temperature sampling"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -512:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class ModelTester:
    """Test the trained model on various content types"""

    def __init__(self, model_path="optimized_quillan_model.pt", tokenizer_path="optimized_tokenizer.pkl"):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cpu')

        self.load_model_and_tokenizer(model_path, tokenizer_path)

    def load_model_and_tokenizer(self, model_path, tokenizer_path):
        """Load the trained model and tokenizer"""
        try:
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Loaded tokenizer with {len(self.tokenizer.vocab)} tokens")

            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = OptimizedBigramLanguageModel(
                vocab_size=len(self.tokenizer.vocab),
                n_embd=512,
                n_head=8,
                n_layer=6
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print("✅ Loaded trained model")
            print(f"   Best loss achieved: {checkpoint.get('loss', 'Unknown'):.4f}")

        except Exception as e:
            print(f"❌ Failed to load model/tokenizer: {e}")
            self.model = None
            self.tokenizer = None

    def test_text_generation(self, prompt="", max_length=100, temperature=0.8):
        """Test text generation from a prompt"""
        if not self.model or not self.tokenizer:
            return "❌ Model not loaded"

        print(f"\n🔍 Testing generation from prompt: '{prompt}'")

        # Encode prompt
        if prompt:
            tokens = self.tokenizer.encode(prompt, max_length=256)
        else:
            tokens = [0] * 10  # Start with padding

        context = torch.tensor([tokens], dtype=torch.long).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            generated = self.model.generate(context, max_length, temperature)
        end_time = time.time()

        generated_text = self.tokenizer.decode(generated[0].tolist())

        print(f"   Generated {len(generated_text)} characters in {end_time - start_time:.2f}s")

        return generated_text

    def test_content_types(self):
        """Test model on different content types from the dataset"""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return

        print("\n🧪 Testing Model on Different Content Types")

        # Test prompts from different content types
        test_prompts = {
            'narrative': "The AI system began to understand",
            'creative': "In the shadows of the digital realm",
            'factual': "The quantum computer processes information",
            'philosophical': "The meaning of consciousness is",
            'technical': "The neural network architecture consists of"
        }

        results = {}
        for category, prompt in test_prompts.items():
            print(f"\n--- Testing {category.upper()} Content ---")
            generated = self.test_text_generation(prompt, max_length=150, temperature=0.7)
            results[category] = generated

            # Basic quality assessment
            word_count = len(generated.split())
            unique_words = len(set(generated.lower().split()))
            diversity_ratio = unique_words / word_count if word_count > 0 else 0

            print(f"   Quality metrics: {word_count} words, {unique_words} unique, {diversity_ratio:.2f} diversity")

        return results

    def benchmark_generation_speed(self):
        """Benchmark generation speed"""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return

        print("\n⚡ Benchmarking Generation Speed")

        test_prompts = [
            "Hello world",
            "The meaning of life is",
            "Artificial intelligence will",
            "In the future, humans will"
        ]

        total_time = 0
        total_tokens = 0

        for prompt in test_prompts:
            start_time = time.time()
            generated = self.test_text_generation(prompt, max_length=50, temperature=1.0)
            end_time = time.time()

            tokens_generated = len(self.tokenizer.encode(generated)) - len(self.tokenizer.encode(prompt))
            time_taken = end_time - start_time

            total_time += time_taken
            total_tokens += tokens_generated

            print(f"   {tokens_generated} tokens in {time_taken:.2f}s ({tokens_generated/time_taken:.1f} tokens/sec)")

        avg_speed = total_tokens / total_time if total_time > 0 else 0
        print(f"   Average Speed: {avg_speed:.1f} tokens/sec")
    def evaluate_coherence(self):
        """Evaluate text coherence and quality"""
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return

        print("\n🎯 Evaluating Text Coherence")

        # Generate multiple samples
        prompts = [
            "The future of AI",
            "Quantum computing",
            "Consciousness explained",
            "Digital immortality"
        ]

        coherence_scores = []

        for prompt in prompts:
            generated = self.test_text_generation(prompt, max_length=200, temperature=0.8)

            # Simple coherence metrics
            sentences = generated.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

            # Check for repeated patterns (lower score for repetition)
            words = generated.lower().split()
            unique_ratio = len(set(words)) / len(words) if words else 0

            # Basic coherence score (0-1)
            coherence = min(1.0, (avg_sentence_length / 15) * unique_ratio)

            coherence_scores.append(coherence)
        print(f"   Generated {len(generated_text)} characters in {end_time - start_time:.2f}s")
        overall_coherence = sum(coherence_scores) / len(coherence_scores)
        print(f"   Generated {len(generated_text)} characters in {end_time - start_time:.2f}s")
        if overall_coherence > 0.6:
            print("   🌟 Excellent coherence - approaching Grok-level quality!")
        elif overall_coherence > 0.4:
            print("   👍 Good coherence - model is learning well")
        else:
            print("   📈 Improving coherence - more training needed")

        return overall_coherence

def main():
    """Main testing function"""
    print("🚀 Starting Model Inference Testing...")

    tester = ModelTester()

    if not tester.model:
        print("❌ Could not load model. Please ensure training completed and saved the model.")
        return

    # Run comprehensive tests
    tester.test_content_types()
    tester.benchmark_generation_speed()
    coherence_score = tester.evaluate_coherence()

    print("\n🎉 Inference Testing Complete!")

    if coherence_score > 0.6:
        print("🏆 Your model is performing at advanced levels!")
    elif coherence_score > 0.4:
        print("📈 Good progress - continue training for better results")
    else:
        print("🔄 Model needs more training - current performance is developing")

if __name__ == "__main__":
    main()
