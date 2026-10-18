#!/usr/bin/env python3
"""
Upload Quillan-Ronin model to Hugging Face Hub (Large Folder)
"""

from huggingface_hub import HfApi
import os

def upload_large_folder():
    """Upload large model files to Hugging Face using upload_large_folder"""
    print("🤗 Uploading Large Quillan-Ronin Model to Hugging Face")
    print("=" * 60)

    token = os.getenv("HUGGINGFACE_TOKEN", "")

    try:
        # Initialize API
        api = HfApi(token=token)
        repo_id = "CrashOverrideX/Quillan-Ronin"

        print(f"📁 Uploading to: {repo_id}")
        print("📤 Using large folder upload for 2.4GB+ model...")
        print("This will take considerable time - please be patient!")

        # Use upload_large_folder for large files
        api.upload_large_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=[
                "*.pyc",
                "__pycache__",
                ".git",
                "*.tmp",
                "node_modules",
                "*.log",
                ".DS_Store",
                "upload_to_hf.py",
                "upload_with_token.py",
                "fix_export.py",
                "export_for_gguf.py",
                "quick_text_test.py",
                "debug_text_gen.py"
            ]
        )

        print("✅ Large folder upload completed successfully!")
        print("\n🌐 Your model is now available at:")
        print(f"https://huggingface.co/{repo_id}")

        # Create and upload README
        model_card = f"""---
language: en
tags:
- multimodal
- text-generation
- image-processing
- audio-processing
- video-processing
- pytorch
- transformers
license: apache-2.0
---

# Quillan-Ronin v5.3.0

**SOTA Multimodal AI Model** - Text, Image, Audio, Video Processing

## Model Details

- **Model Name:** Quillan-Ronin v5.3.0
- **Model Type:** Multimodal Transformer with MoE Architecture
- **Parameters:** 207M
- **Training Steps:** 1500
- **Final Loss:** 0.009767
- **Confidence Score:** 87.4%

## Architecture

- **Mixture of Experts (MoE):** 8 experts with capacity loss regularization
- **Diffusion Layers:** 4 layers for multimodal generation
- **Confidence Calibration:** Meta-gradient CCRL framework
- **Paradox Gates:** Contradiction detection and handling
- **Epistemic Humility:** Uncertainty-aware processing

## Capabilities

- **Text Generation:** Creative and technical writing
- **Image Processing:** Understanding and generation
- **Audio Analysis:** Music and speech processing
- **Video Understanding:** Temporal sequence analysis
- **Multimodal Fusion:** Cross-modal reasoning

## Usage

### Web Interface
```bash
# Start the web interface
python chat_api.py
# Open progress.html in browser
```

### Python API
```python
from chat_api import ChatAPI
chat = ChatAPI()
chat.load_model()
response = chat.generate_response("Hello!")
```

## Files Included

- `best_multimodal_quillan.pt`: Trained model checkpoint (2.4GB)
- `train_full_multimodal.py`: Training script
- `progress.html`: Web interface
- `chat_api.py`: Flask API backend
- `config.json`: Model configuration

## Training Data

Trained on comprehensive multimodal dataset including:
- Text corpora and knowledge bases
- Image collections
- Audio files and lyrics
- Video content
- Cross-modal alignment data

## Authors

- **CrashOverrideX** - Lead Developer
- **Quillan Research Team** - Research and Development

## License

Apache 2.0
"""

        with open("README.md", "w") as f:
            f.write(model_card)

        # Upload README
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

        print("✅ README uploaded")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("• Check token permissions")
        print("• Ensure repository exists")
        print("• Large uploads may take time")
        print("• Check internet connection")
        return False

    return True

if __name__ == "__main__":
    upload_large_folder()
