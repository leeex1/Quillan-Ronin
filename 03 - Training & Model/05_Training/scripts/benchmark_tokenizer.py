#!/usr/bin/env python3
"""
Benchmark tokenizer performance for Quillan-Ronin.
Tests BPE tokenizer speed and compares against alternatives.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / '_dev'))

# Test text samples
test_texts = [
    "The future of artificial intelligence is bright and promising.",
    "Once upon a time, in a land far away, there lived a brave knight.",
    "In a world where technology advances rapidly, adaptation is key.",
    "The most important lesson I learned is that perseverance pays off.",
    "Machine learning models require large datasets to generalize well.",
    "Python is a popular programming language for data science and AI.",
    "The model architecture uses mixture of experts for efficiency.",
    "Quantum computing promises to revolutionize cryptographic systems.",
    "Natural language processing has made significant progress recently.",
    "Deep learning networks can learn hierarchical representations."
]

def benchmark_bpe():
    """Benchmark BPE tokenizer."""
    print("=== BPE Tokenizer Benchmark ===")
    try:
        from quillan_bpe_tokenizer import QuillanBPETokenizer
        tokenizer = QuillanBPETokenizer()
        tok_path = ROOT / 'training_data' / 'tokenizer.json'
        if not tok_path.exists():
            tok_path = ROOT / '_dev' / 'quillan_bpe_tokenizer_hf' / 'tokenizer.json'
        tokenizer.load(str(tok_path))
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        # Warmup
        for text in test_texts[:2]:
            tokenizer.encode(text)
        
        # Benchmark encode
        encode_times = []
        for text in test_texts:
            start = time.time()
            tokens = tokenizer.encode(text)
            elapsed = time.time() - start
            encode_times.append(elapsed)
        
        # Benchmark decode
        decode_times = []
        for text in test_texts:
            tokens = tokenizer.encode(text)
            start = time.time()
            decoded = tokenizer.decode(tokens)
            elapsed = time.time() - start
            decode_times.append(elapsed)
        
        avg_encode = sum(encode_times) / len(encode_times)
        avg_decode = sum(decode_times) / len(decode_times)
        
        print(f"Average encode time: {avg_encode*1000:.2f} ms")
        print(f"Average decode time: {avg_decode*1000:.2f} ms")
        print(f"Total (encode+decode): {(avg_encode+avg_decode)*1000:.2f} ms")
        
        return avg_encode, avg_decode
    except Exception as e:
        print(f"Error benchmarking BPE: {e}")
        return None, None

def benchmark_sentencepiece():
    """Benchmark SentencePiece tokenizer."""
    print("\n=== SentencePiece Tokenizer Benchmark ===")
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp_model = ROOT / 'training_data' / 'tokenizer.model'
        if sp_model.exists():
            sp.load(str(sp_model))
            print(f"Vocab size: {sp.get_piece_size()}")
            
            # Warmup
            for text in test_texts[:2]:
                sp.encode(text)
            
            # Benchmark encode
            encode_times = []
            for text in test_texts:
                start = time.time()
                tokens = sp.encode(text)
                elapsed = time.time() - start
                encode_times.append(elapsed)
            
            # Benchmark decode
            decode_times = []
            for text in test_texts:
                tokens = sp.encode(text)
                start = time.time()
                decoded = sp.decode(tokens)
                elapsed = time.time() - start
                decode_times.append(elapsed)
            
            avg_encode = sum(encode_times) / len(encode_times)
            avg_decode = sum(decode_times) / len(decode_times)
            
            print(f"Average encode time: {avg_encode*1000:.2f} ms")
            print(f"Average decode time: {avg_decode*1000:.2f} ms")
            print(f"Total (encode+decode): {(avg_encode+avg_decode)*1000:.2f} ms")
            
            return avg_encode, avg_decode
        else:
            print("SentencePiece model not found")
            return None, None
    except ImportError:
        print("sentencepiece not installed")
        return None, None
    except Exception as e:
        print(f"Error benchmarking SentencePiece: {e}")
        return None, None

def benchmark_gpt2():
    """Benchmark GPT-2 tokenizer."""
    print("\n=== GPT-2 Tokenizer Benchmark ===")
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        # Warmup
        for text in test_texts[:2]:
            tokenizer.encode(text)
        
        # Benchmark encode
        encode_times = []
        for text in test_texts:
            start = time.time()
            tokens = tokenizer.encode(text)
            elapsed = time.time() - start
            encode_times.append(elapsed)
        
        # Benchmark decode
        decode_times = []
        for text in test_texts:
            tokens = tokenizer.encode(text)
            start = time.time()
            decoded = tokenizer.decode(tokens)
            elapsed = time.time() - start
            decode_times.append(elapsed)
        
        avg_encode = sum(encode_times) / len(encode_times)
        avg_decode = sum(decode_times) / len(decode_times)
        
        print(f"Average encode time: {avg_encode*1000:.2f} ms")
        print(f"Average decode time: {avg_decode*1000:.2f} ms")
        print(f"Total (encode+decode): {(avg_encode+avg_decode)*1000:.2f} ms")
        
        return avg_encode, avg_decode
    except ImportError:
        print("transformers not installed")
        return None, None
    except Exception as e:
        print(f"Error benchmarking GPT-2: {e}")
        return None, None

if __name__ == '__main__':
    bpe_enc, bpe_dec = benchmark_bpe()
    sp_enc, sp_dec = benchmark_sentencepiece()
    gpt2_enc, gpt2_dec = benchmark_gpt2()
    
    print("\n=== Summary ===")
    print("Tokenizer | Encode (ms) | Decode (ms) | Total (ms)")
    print("-" * 50)
    if bpe_enc is not None:
        print(f"BPE        | {bpe_enc*1000:10.2f} | {bpe_dec*1000:10.2f} | {(bpe_enc+bpe_dec)*1000:10.2f}")
    if sp_enc is not None:
        print(f"SentencePiece | {(sp_enc)*1000:8.2f} | {sp_dec*1000:10.2f} | {(sp_enc+sp_dec)*1000:10.2f}")
    if gpt2_enc is not None:
        print(f"GPT-2      | {gpt2_enc*1000:10.2f} | {gpt2_dec*1000:10.2f} | {(gpt2_enc+gpt2_dec)*1000:10.2f}")
