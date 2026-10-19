#!/usr/bin/env python3
"""
Quillan-Ronin v5.3.0 Training Script for Google Colab
Optimized for GPU acceleration with visual progress tracking
"""

# Install dependencies first
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install matplotlib numpy scipy

# Check GPU availability
import torch
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🧠 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Mount Google Drive for data access
from google.colab import drive
drive.mount('/content/drive')

# Upload your training data to Google Drive first
# Then update these paths accordingly
DATA_PATH = "/content/drive/MyDrive/QuillanData/"

# Copy the training script and run
# !cp /content/drive/MyDrive/Quillan/train_full_multimodal.py .
# !python train_full_multimodal.py

# After training, download the results
from google.colab import files
# !files.download("best_multimodal_quillan.pt")
# !files.download("training_progress_step_*.png")
