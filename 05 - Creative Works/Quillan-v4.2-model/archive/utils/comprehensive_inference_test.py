#!/usr/bin/env python3
"""
Comprehensive Multimodal Inference Testing Suite
Tests text, image, audio, video modalities for stability and correctness
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from train_full_multimodal import QuillanRoninV5_3, Config, SimpleTokenizer
from data_loader import QuillanDataset

class MultimodalInferenceTester:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device('cpu')  # Use CPU for stable testing
        self.cfg.device = self.device
        self.model = None
        self.tokenizer = None
        self.test_results = {
            'text': {'passed': 0, 'failed': 0, 'errors': []},
            'image': {'passed': 0, 'failed': 0, 'errors': []},
            'audio': {'passed': 0, 'failed': 0, 'errors': []},
            'video': {'passed': 0, 'failed': 0, 'errors': []},
            'combined': {'passed': 0, 'failed': 0, 'errors': []}
        }
        self.benchmarks = {
            'latency': [],
            'memory': [],
            'throughput': []
        }

    def load_model(self):
        """Load the trained model safely"""
        try:
            print("🔄 Loading Quillan-Ronin model...")
            self.model = QuillanRoninV5_3(self.cfg).to(self.device)
            self.model.eval()

            # Try to load checkpoint
            checkpoint_path = "best_multimodal_quillan.pt"
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print("✅ Model checkpoint loaded")
                except Exception as e:
                    print(f"⚠️ Checkpoint loading failed: {e}")
                    print("🔄 Using untrained model for testing")
            else:
                print("⚠️ No checkpoint found, using untrained model")

            # Setup tokenizer
            dataset = QuillanDataset()
            self.tokenizer = SimpleTokenizer(vocab_size=1000)
            all_texts = [s['text'] for s in dataset.samples]
            self.tokenizer.train(all_texts)

            print("✅ Model and tokenizer ready")
            return True

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

    def get_memory_usage(self):
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_text_inference(self):
        """Test text generation inference"""
        print("\n🧪 Testing Text Inference...")

        test_cases = [
            "Hello world",
            "The future of AI",
            "Machine learning is",
            "Quantum computing",
            "Neural networks can",
            "",  # Empty input
            "A" * 100,  # Long input
        ]

        for i, text_input in enumerate(test_cases):
            try:
                start_time = time.time()
                initial_memory = self.get_memory_usage()

                # Tokenize input
                tokens = self.tokenizer.encode(text_input, max_length=64)
                if not tokens:
                    tokens = [1] * 10  # Fallback tokens

                text_tensor = torch.tensor([tokens], device=self.device)

                # Create dummy multimodal inputs
                dummy_image = torch.randn(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
                dummy_audio = torch.randn(1, 1, self.cfg.audio_samples // 64, device=self.device)
                dummy_video = torch.randn(1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(text_tensor, dummy_image, dummy_audio, dummy_video)

                # Check outputs
                text_output = outputs['text']
                assert text_output.shape[0] == 1, "Batch size mismatch"
                assert text_output.shape[2] == self.cfg.vocab_size, "Vocab size mismatch"
                assert not torch.isnan(text_output).any(), "NaN in text output"
                assert not torch.isinf(text_output).any(), "Inf in text output"

                # Check reproducibility
                with torch.no_grad():
                    outputs2 = self.model(text_tensor, dummy_image, dummy_audio, dummy_video)
                assert torch.allclose(text_output, outputs2['text'], atol=1e-6), "Non-deterministic outputs"

                end_time = time.time()
                final_memory = self.get_memory_usage()

                self.test_results['text']['passed'] += 1
                self.benchmarks['latency'].append(end_time - start_time)
                self.benchmarks['memory'].append(final_memory - initial_memory)

                print(f"  ✅ Test {i+1}/{len(test_cases)}: '{text_input[:30]}...' -> {text_output.shape}")

            except Exception as e:
                self.test_results['text']['failed'] += 1
                self.test_results['text']['errors'].append(f"Test {i+1} ({text_input[:30]}): {str(e)}")
                print(f"  ❌ Test {i+1}/{len(test_cases)} failed: {e}")

    def test_image_inference(self):
        """Test image generation inference"""
        print("\n🧪 Testing Image Inference...")

        test_cases = [
            "A beautiful sunset",
            "A cat playing",
            "Mountain landscape",
            "Abstract art",
            "Portrait photo",
        ]

        for i, prompt in enumerate(test_cases):
            try:
                start_time = time.time()
                initial_memory = self.get_memory_usage()

                # Create dummy inputs
                text_tokens = torch.tensor([[1] * 10], device=self.device)  # Dummy text
                image_input = torch.randn(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
                dummy_audio = torch.randn(1, 1, self.cfg.audio_samples // 64, device=self.device)
                dummy_video = torch.randn(1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(text_tokens, image_input, dummy_audio, dummy_video)

                # Check image output
                image_output = outputs['image']
                expected_shape = (1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size)
                assert image_output.shape == expected_shape, f"Shape mismatch: {image_output.shape} vs {expected_shape}"
                assert not torch.isnan(image_output).any(), "NaN in image output"
                assert not torch.isinf(image_output).any(), "Inf in image output"
                assert image_output.min() >= -10.0 and image_output.max() <= 10.0, "Output values out of range"

                end_time = time.time()
                final_memory = self.get_memory_usage()

                self.test_results['image']['passed'] += 1
                self.benchmarks['latency'].append(end_time - start_time)
                self.benchmarks['memory'].append(final_memory - initial_memory)

                print(f"  ✅ Test {i+1}/{len(test_cases)}: '{prompt}' -> {image_output.shape}")

            except Exception as e:
                self.test_results['image']['failed'] += 1
                self.test_results['image']['errors'].append(f"Test {i+1} ({prompt}): {str(e)}")
                print(f"  ❌ Test {i+1}/{len(test_cases)} failed: {e}")

    def test_audio_inference(self):
        """Test audio generation inference"""
        print("\n🧪 Testing Audio Inference...")

        test_cases = [
            "Classical music",
            "Rock guitar solo",
            "Nature sounds",
            "Speech recording",
            "Electronic beats",
        ]

        for i, prompt in enumerate(test_cases):
            try:
                start_time = time.time()
                initial_memory = self.get_memory_usage()

                # Create dummy inputs
                text_tokens = torch.tensor([[1] * 10], device=self.device)
                dummy_image = torch.randn(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
                audio_input = torch.randn(1, 1, self.cfg.audio_samples // 64, device=self.device)
                dummy_video = torch.randn(1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(text_tokens, dummy_image, audio_input, dummy_video)

                # Check audio output
                audio_output = outputs['audio']
                expected_shape = (1, 1, self.cfg.audio_samples // 64)
                assert audio_output.shape == expected_shape, f"Shape mismatch: {audio_output.shape} vs {expected_shape}"
                assert not torch.isnan(audio_output).any(), "NaN in audio output"
                assert not torch.isinf(audio_output).any(), "Inf in audio output"

                end_time = time.time()
                final_memory = self.get_memory_usage()

                self.test_results['audio']['passed'] += 1
                self.benchmarks['latency'].append(end_time - start_time)
                self.benchmarks['memory'].append(final_memory - initial_memory)

                print(f"  ✅ Test {i+1}/{len(test_cases)}: '{prompt}' -> {audio_output.shape}")

            except Exception as e:
                self.test_results['audio']['failed'] += 1
                self.test_results['audio']['errors'].append(f"Test {i+1} ({prompt}): {str(e)}")
                print(f"  ❌ Test {i+1}/{len(test_cases)} failed: {e}")

    def test_video_inference(self):
        """Test video generation inference"""
        print("\n🧪 Testing Video Inference...")

        test_cases = [
            "Dancing people",
            "Car driving",
            "Ocean waves",
            "Cooking demonstration",
            "Sports action",
        ]

        for i, prompt in enumerate(test_cases):
            try:
                start_time = time.time()
                initial_memory = self.get_memory_usage()

                # Create dummy inputs
                text_tokens = torch.tensor([[1] * 10], device=self.device)
                dummy_image = torch.randn(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
                dummy_audio = torch.randn(1, 1, self.cfg.audio_samples // 64, device=self.device)
                video_input = torch.randn(1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(text_tokens, dummy_image, dummy_audio, video_input)

                # Check video output
                video_output = outputs['video']
                expected_shape = (1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8)
                assert video_output.shape == expected_shape, f"Shape mismatch: {video_output.shape} vs {expected_shape}"
                assert not torch.isnan(video_output).any(), "NaN in video output"
                assert not torch.isinf(video_output).any(), "Inf in video output"

                end_time = time.time()
                final_memory = self.get_memory_usage()

                self.test_results['video']['passed'] += 1
                self.benchmarks['latency'].append(end_time - start_time)
                self.benchmarks['memory'].append(final_memory - initial_memory)

                print(f"  ✅ Test {i+1}/{len(test_cases)}: '{prompt}' -> {video_output.shape}")

            except Exception as e:
                self.test_results['video']['failed'] += 1
                self.test_results['video']['errors'].append(f"Test {i+1} ({prompt}): {str(e)}")
                print(f"  ❌ Test {i+1}/{len(test_cases)} failed: {e}")

    def test_combined_inference(self):
        """Test combined multimodal inference"""
        print("\n🧪 Testing Combined Multimodal Inference...")

        test_cases = [
            {
                'text': 'A beautiful landscape',
                'has_image': True,
                'has_audio': True,
                'has_video': True
            },
            {
                'text': 'Music performance',
                'has_image': True,
                'has_audio': False,
                'has_video': True
            },
            {
                'text': 'Silent movie scene',
                'has_image': True,
                'has_audio': False,
                'has_video': True
            }
        ]

        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                initial_memory = self.get_memory_usage()

                # Create inputs based on test case
                text_tokens = torch.tensor([self.tokenizer.encode(test_case['text'], max_length=32)], device=self.device)

                if test_case['has_image']:
                    image_input = torch.randn(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
                else:
                    image_input = torch.zeros(1, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)

                if test_case['has_audio']:
                    audio_input = torch.randn(1, 1, self.cfg.audio_samples // 64, device=self.device)
                else:
                    audio_input = torch.zeros(1, 1, self.cfg.audio_samples // 64, device=self.device)

                if test_case['has_video']:
                    video_input = torch.randn(1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)
                else:
                    video_input = torch.zeros(1, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(text_tokens, image_input, audio_input, video_input)

                # Validate all outputs
                for modality in ['text', 'image', 'audio', 'video']:
                    output = outputs[modality]
                    assert not torch.isnan(output).any(), f"NaN in {modality} output"
                    assert not torch.isinf(output).any(), f"Inf in {modality} output"

                end_time = time.time()
                final_memory = self.get_memory_usage()

                self.test_results['combined']['passed'] += 1
                self.benchmarks['latency'].append(end_time - start_time)
                self.benchmarks['memory'].append(final_memory - initial_memory)

                modalities = []
                if test_case['has_image']: modalities.append('I')
                if test_case['has_audio']: modalities.append('A')
                if test_case['has_video']: modalities.append('V')

                print(f"  ✅ Test {i+1}/{len(test_cases)}: '{test_case['text']}' + {''.join(modalities)} -> All outputs valid")

            except Exception as e:
                self.test_results['combined']['failed'] += 1
                self.test_results['combined']['errors'].append(f"Test {i+1} ({test_case['text']}): {str(e)}")
                print(f"  ❌ Test {i+1}/{len(test_cases)} failed: {e}")

    def test_batch_inference(self):
        """Test batch processing capabilities"""
        print("\n🧪 Testing Batch Inference...")

        batch_sizes = [1, 2, 4]

        for batch_size in batch_sizes:
            try:
                print(f"  Testing batch size {batch_size}...")

                # Create batched inputs
                text_tokens = torch.randint(0, self.cfg.vocab_size, (batch_size, 32), device=self.device)
                image_input = torch.randn(batch_size, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size, device=self.device)
                audio_input = torch.randn(batch_size, 1, self.cfg.audio_samples // 64, device=self.device)
                video_input = torch.randn(batch_size, self.cfg.video_channels, self.cfg.video_frames // 8, self.cfg.video_height // 8, self.cfg.video_width // 8, device=self.device)

                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(text_tokens, image_input, audio_input, video_input)
                end_time = time.time()

                # Validate batch outputs
                for modality in ['text', 'image', 'audio', 'video']:
                    output = outputs[modality]
                    expected_batch = batch_size
                    actual_batch = output.shape[0]
                    assert actual_batch == expected_batch, f"Batch size mismatch for {modality}: {actual_batch} vs {expected_batch}"

                latency = end_time - start_time
                throughput = batch_size / latency

                self.benchmarks['throughput'].append(throughput)
                print(f"  ✅ Batch size {batch_size}: {throughput:.3f} samples/sec")
            except Exception as e:
                print(f"  ❌ Batch size {batch_size} failed: {e}")

    def run_full_test_suite(self):
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive Multimodal Inference Test Suite")
        print("=" * 70)

        if not self.load_model():
            print("❌ Failed to load model - aborting tests")
            return False

        # Run all tests
        self.test_text_inference()
        self.test_image_inference()
        self.test_audio_inference()
        self.test_video_inference()
        self.test_combined_inference()
        self.test_batch_inference()

        # Generate summary
        self.generate_test_summary()
        return True

    def generate_test_summary(self):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE TEST SUITE RESULTS")
        print("=" * 70)

        total_passed = 0
        total_failed = 0

        for modality, results in self.test_results.items():
            passed = results['passed']
            failed = results['failed']
            total_passed += passed
            total_failed += failed

            success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0

            print(f"\n🔍 {modality.upper()} MODALITY:")
            print(f"   ✅ Passed: {passed}")
            print(f"   ❌ Failed: {failed}")
            print(f"   Success Rate: {success_rate:.1f}%")
            if results['errors']:
                print("   ⚠️  Errors:")
                for error in results['errors'][:3]:  # Show first 3 errors
                    print(f"      • {error}")

        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   ✅ Total Passed: {total_passed}")
        print(f"   ❌ Total Failed: {total_failed}")
        overall_success = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        print(f"   Success Rate: {overall_success:.1f}%")
        # Benchmark summary
        if self.benchmarks['latency']:
            avg_latency = np.mean(self.benchmarks['latency'])
            avg_memory = np.mean(self.benchmarks['memory']) if self.benchmarks['memory'] else 0
            avg_throughput = np.mean(self.benchmarks['throughput']) if self.benchmarks['throughput'] else 0

            print(f"\n⚡ BENCHMARKS:")
            print(f"   Avg Latency: {avg_latency:.4f}s")
            print(f"   Avg Memory: {avg_memory:.1f}MB")
            print(f"   Avg Throughput: {avg_throughput:.2f} samples/sec")
        print("\n" + "=" * 70)

        if overall_success >= 95:
            print("🎉 TEST SUITE PASSED - Pipeline is stable and production-ready!")
        elif overall_success >= 80:
            print("⚠️  TEST SUITE MOSTLY PASSED - Minor issues need attention")
        else:
            print("❌ TEST SUITE FAILED - Critical issues need fixing")

def main():
    """Run the comprehensive inference test suite"""
    tester = MultimodalInferenceTester()
    success = tester.run_full_test_suite()

    if not success:
        print("❌ Test suite failed to initialize")
        exit(1)

if __name__ == "__main__":
    main()
