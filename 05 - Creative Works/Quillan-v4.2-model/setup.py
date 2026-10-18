#!/usr/bin/env python3
"""
Setup script for Quillan Model
Run this after resolving antivirus/pip issues
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print("❌ FAILED")
            print("Error:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("🚀 Quillan Model Setup Script")
    print("This script will set up the environment for your Quillan model")
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Install PyTorch first
    pytorch_cmd = 'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA support"):
        print("⚠️  PyTorch installation failed, trying CPU-only version...")
        pytorch_cpu_cmd = 'pip install torch torchvision torchaudio'
        run_command(pytorch_cpu_cmd, "Installing PyTorch CPU-only")
    
    # Install other dependencies
    deps = [
        'pip install fastapi uvicorn',
        'pip install numpy pydantic',
        'pip install scikit-learn matplotlib',
        'pip install transformers'
    ]
    
    for dep in deps:
        run_command(dep, f"Installing {dep.split()[-1]}")
    
    # Test the installation
    print("\n" + "="*50)
    print("Testing installation...")
    print("="*50)
    
    test_cmd = 'python -c "import torch; print(f\'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}\')"'
    run_command(test_cmd, "Testing PyTorch import")
    
    # Test model import
    test_model_cmd = 'python test_model.py'
    run_command(test_model_cmd, "Testing Quillan model import")
    
    print("\n" + "="*50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run: python train.py")
    print("2. Or run: python inference.py")
    print("3. Check the README.md for more details")
    print("="*50)

if __name__ == "__main__":
    main()
