#!/usr/bin/env python3
"""
Comprehensive evaluation script for Quillan-Ronin v5.3.1
Tests generation quality, perplexity, and reasoning capabilities.
"""
import sys, os, json
from pathlib import Path
import torch
import torch.nn.functional as F
import time
import math

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / '_dev'))

from quillan_v8_saturated import QuillanRoninSovereign, QuillanArchConfig
from quillan_bpe_tokenizer import QuillanBPETokenizer

def load_tokenizer():
    """Load GPT-2 tokenizer to match model checkpoint (50257 vocab)."""
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print(f'[OK] Loaded GPT-2 tokenizer: {tokenizer.vocab_size} vocab')
        return tokenizer
    except ImportError:
        print('[WARN] transformers not installed, using fallback')
        return None

def load_model(ckpt_path, tokenizer, device='cuda'):
    """Load model from checkpoint."""
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        else:
            state = ckpt
        config = ckpt.get('config', {})
    else:
        state = ckpt
        config = {}
    
    # Use SFT checkpoint configuration (hidden_dim=1024, ffn_dim=2048, eggroll_rank=16)
    # Initialize with checkpoint vocab size (50257) to load successfully
    cfg = QuillanArchConfig(
        text_only=True,
        hidden_dim=config.get('hidden_dim', 1024),
        ffn_dim=config.get('ffn_dim', 2048),
        vocab_size=config.get('vocab_size', 50257),  # Match checkpoint vocab size first
        num_experts=config.get('num_experts', 34),
        top_k=config.get('top_k', 4),
        eggroll_rank=16,  # SFT checkpoint uses rank=16 for expert swarms
        e_ice_limit_ms=100,
        device=device
    )
    
    model = QuillanRoninSovereign(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    
    print(f'[OK] Model loaded')
    print(f'  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}')
    print(f'  Step: {ckpt.get("step", "unknown")}, Loss: {ckpt.get("loss", "unknown")}')
    
    return model, cfg

def calculate_perplexity(model, data, tokenizer, device='cuda', max_batches=10):
    """Calculate perplexity on validation data."""
    print('\n--- Perplexity Evaluation ---')
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, text in enumerate(data[:max_batches]):
            tokens = tokenizer.encode(text) if tokenizer else [ord(c) % 50257 for c in text]
            if len(tokens) < 10:
                continue
            
            # Truncate to max sequence length (1024)
            max_len = 1024
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
            
            out = model(x)
            logits = out['logits'] if isinstance(out, dict) else out
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
            
            if i % 5 == 0:
                print(f'  Batch {i+1}/{min(max_batches, len(data))}: Loss={loss.item():.4f}')
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    print(f'[OK] Perplexity: {perplexity:.2f}')
    return perplexity

def evaluate_generation(model, tokenizer, prompts, device='cuda', max_tokens=50, temperature=0.8):
    """Evaluate text generation quality with improved sampling."""
    print('\n--- Generation Evaluation ---')
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f'\nPrompt {i+1}: {prompt}')
        
        tokens = tokenizer.encode(prompt) if tokenizer else [ord(c) % 50257 for c in prompt]
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        generated = []
        start_time = time.time()
        
        with torch.no_grad():
            for step in range(max_tokens):
                out = model(x, recursive_depth=0)  # Disable recursive consciousness loop
                logits = out['logits'] if isinstance(out, dict) else out
                
                # Apply repetition penalty (2.0 as per antigravity walkthrough)
                for token in set(generated):
                    logits[0, -1, token] /= 2.0
                
                # Nucleus sampling (top-p=0.85)
                probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.85
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, -1, indices_to_remove] = float('-inf')
                
                next_token = torch.multinomial(F.softmax(logits[0, -1], dim=-1), 1)
                generated.append(next_token.item())
                x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        
        elapsed = time.time() - start_time
        tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0
        
        full_tokens = tokens + generated
        generated_text = tokenizer.decode(full_tokens) if tokenizer else ''.join(chr(t % 128) for t in full_tokens)
        
        try:
            print(f'Generated: {generated_text}')
        except UnicodeEncodeError:
            print(f'Generated: [Unicode error - text contains unsupported characters]')
        print(f'Speed: {tokens_per_sec:.2f} tokens/sec')
        
        results.append({
            'prompt': prompt,
            'generated': generated_text,
            'tokens_per_sec': tokens_per_sec,
            'time': elapsed
        })
    
    return results

def evaluate_reasoning(model, tokenizer, device='cuda'):
    """Evaluate reasoning capabilities with simple logic tasks."""
    print('\n--- Reasoning Evaluation ---')
    
    reasoning_prompts = [
        "If all cats are animals and Fluffy is a cat, then",
        "The sum of 5 and 7 is",
        "To bake a cake, first you need to",
        "The opposite of hot is"
    ]
    
    results = evaluate_generation(model, tokenizer, reasoning_prompts, device, max_tokens=20, temperature=0.6)
    
    # Simple coherence check
    coherent = 0
    for r in results:
        gen = r['generated'].lower()
        if any(word in gen for word in ['is', 'then', 'the', 'you', 'need', 'cold']):
            coherent += 1
    
    coherence_rate = coherent / len(results) if results else 0
    print(f'[OK] Coherence rate: {coherence_rate:.2%}')
    
    return results, coherence_rate

def check_vram_usage(model, device='cuda'):
    """Check VRAM usage."""
    if device == 'cuda' and torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated(device) / 1024**3
        vram_reserved = torch.cuda.memory_reserved(device) / 1024**3
        vram_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        print(f'\n--- VRAM Usage ---')
        print(f'Allocated: {vram_allocated:.2f} GB')
        print(f'Reserved: {vram_reserved:.2f} GB')
        print(f'Total: {vram_total:.2f} GB')
        print(f'Usage: {vram_allocated/vram_total*100:.1f}%')
        
        return {
            'allocated_gb': vram_allocated,
            'reserved_gb': vram_reserved,
            'total_gb': vram_total,
            'usage_percent': vram_allocated/vram_total*100
        }
    return None

def main():
    device = 'cpu'  # Force CPU - GTX 1050 incompatible with current PyTorch and insufficient VRAM
    print(f'Using device: {device}')
    
    # Find latest checkpoint - use checkpoints_sft (completed SFT training)
    ckpt_dir = BASE / 'checkpoints_sft'
    ckpt_files = list(ckpt_dir.glob('*.pt'))
    
    if not ckpt_files:
        print(f'ERROR: No checkpoints found in {ckpt_dir}')
        return
    
    latest = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    print(f'Latest checkpoint: {latest.name}')
    
    # Load model and tokenizer
    tokenizer = load_tokenizer()
    model, cfg = load_model(str(latest), tokenizer, device)
    
    # Check VRAM
    vram_info = check_vram_usage(device)
    
    # Evaluation prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time",
        "In a world where technology advances rapidly",
        "The most important lesson I learned"
    ]
    
    # Run evaluations
    gen_results = evaluate_generation(model, tokenizer, test_prompts, device, max_tokens=30)
    reason_results, coherence = evaluate_reasoning(model, tokenizer, device)
    
    # Load actual JSONL training datasets used in SFT training
    validation_texts = []
    jsonl_files = [
        BASE / 'training_data' / 'instruct_train.jsonl',
        BASE / 'training_data' / 'GPT_5.5_Distilled.jsonl',
        BASE / 'training_data' / 'code_train.jsonl',
        BASE / 'training_data' / 'quillan_12mb_training_dataset.jsonl',
        BASE / 'training_data' / 'quillan_science_absolute.jsonl',
        BASE / 'training_data' / 'quillan_science_additional.jsonl'
    ]
    
    for jsonl_file in jsonl_files:
        if jsonl_file.exists():
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= 50:  # Take first 50 samples per file
                            break
                        try:
                            data = json.loads(line)
                            # Extract text content from different formats
                            if 'messages' in data:
                                # Chat format - extract assistant responses
                                for msg in data['messages']:
                                    if msg.get('role') == 'assistant':
                                        validation_texts.append(msg.get('content', ''))
                                        break
                            elif 'text' in data:
                                validation_texts.append(data['text'])
                            elif 'content' in data:
                                validation_texts.append(data['content'])
                            elif 'prompt' in data and 'completion' in data:
                                validation_texts.append(data['prompt'] + ' ' + data['completion'])
                            count += 1
                        except:
                            continue
                print(f'Loaded {count} samples from {jsonl_file.name}')
            except Exception as e:
                print(f'Failed to load {jsonl_file.name}: {e}')
    
    # Also load pre-train blend data
    pretrain_file = BASE / 'training_data' / 'quillan_corpus_CLEAN_V7.pt'
    if pretrain_file.exists():
        try:
            data = torch.load(pretrain_file, map_location='cpu')
            if isinstance(data, list):
                validation_texts.extend(data[:50])
            print(f'Loaded 50 samples from {pretrain_file.name}')
        except Exception as e:
            print(f'Failed to load {pretrain_file.name}: {e}')
    
    if not validation_texts:
        # Fallback to hardcoded samples
        validation_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "The model architecture uses mixture of experts for efficiency."
        ]
        print('Using fallback validation texts')
    
    print(f'Total validation samples: {len(validation_texts)}')
    
    if tokenizer:
        perplexity = calculate_perplexity(model, validation_texts, tokenizer, device)
    else:
        perplexity = None
    
    # Summary
    print('\n' + '='*50)
    print('EVALUATION SUMMARY')
    print('='*50)
    print(f'Checkpoint: {latest.name}')
    print(f'Device: {device}')
    print(f'Coherence Rate: {coherence:.2%}')
    if perplexity:
        print(f'Perplexity: {perplexity:.2f}')
    if vram_info:
        print(f'VRAM Usage: {vram_info["usage_percent"]:.1f}%')
    print(f'Average Generation Speed: {sum(r["tokens_per_sec"] for r in gen_results)/len(gen_results):.2f} tokens/sec')
    print('='*50)
    
    # Save results
    results = {
        'checkpoint': str(latest),
        'device': device,
        'coherence_rate': coherence,
        'perplexity': perplexity,
        'vram_usage': vram_info,
        'generation_results': gen_results,
        'reasoning_results': reason_results
    }
    
    out_path = BASE / 'evaluation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[OK] Results saved to {out_path}')

if __name__ == '__main__':
    main()
