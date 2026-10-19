#!/usr/bin/env python3
"""
Content Explorer - Browse and sample the provided multimodal content
"""

import os
import json
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt

class ContentExplorer:
    """Explore and sample the provided content"""

    def __init__(self, data_dir=".."):
        self.data_dir = data_dir
        self.load_content_index()

    def load_content_index(self):
        """Load index of available content"""
        self.content_index = {
            'jsonl': [],
            'lyrics': [],
            'knowledge': [],
            'images': [],
            'audio': [],
            'videos': []
        }

        # Load JSONL samples
        jsonl_path = os.path.join(self.data_dir, "..", "Quillan_finetune_full_dataset.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', data.get('Output_Sections', {}).get('Final output', ''))
                        if text and len(text) > 100:
                            self.content_index['jsonl'].append({
                                'index': i,
                                'text': text[:500] + '...' if len(text) > 500 else text,
                                'length': len(text)
                            })
                    except:
                        continue

        # Load lyrics
        lyrics_dir = os.path.join(self.data_dir, "Songs Lyrics")
        if os.path.exists(lyrics_dir):
            for file_path in glob.glob(os.path.join(lyrics_dir, "*.md")):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) > 200:
                            self.content_index['lyrics'].append({
                                'file': os.path.basename(file_path),
                                'preview': content[:300] + '...' if len(content) > 300 else content,
                                'length': len(content)
                            })
                except:
                    continue

        # Load knowledge files
        knowledge_dir = os.path.join(self.data_dir, "Quillan Knowledge files")
        if os.path.exists(knowledge_dir):
            for file_path in glob.glob(os.path.join(knowledge_dir, "*.md")):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) > 100:
                            self.content_index['knowledge'].append({
                                'file': os.path.basename(file_path),
                                'preview': content[:300] + '...' if len(content) > 300 else content,
                                'length': len(content)
                            })
                except:
                    continue

        # Load images
        images_dir = os.path.join(self.data_dir, "Main images")
        if os.path.exists(images_dir):
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            for ext in image_extensions:
                for file_path in glob.glob(os.path.join(images_dir, ext)):
                    self.content_index['images'].append({
                        'file': os.path.basename(file_path),
                        'path': file_path,
                        'size': os.path.getsize(file_path)
                    })

        # Load audio
        audio_dir = os.path.join(self.data_dir, "Mp3 files")
        if os.path.exists(audio_dir):
            audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg']
            for ext in audio_extensions:
                for file_path in glob.glob(os.path.join(audio_dir, ext)):
                    self.content_index['audio'].append({
                        'file': os.path.basename(file_path),
                        'path': file_path,
                        'size': os.path.getsize(file_path)
                    })

        # Load videos
        video_dir = os.path.join(self.data_dir, "Lyric Videos")
        if os.path.exists(video_dir):
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
            for ext in video_extensions:
                for file_path in glob.glob(os.path.join(video_dir, ext)):
                    self.content_index['videos'].append({
                        'file': os.path.basename(file_path),
                        'path': file_path,
                        'size': os.path.getsize(file_path)
                    })

        print("✅ Content index loaded!")
        self.print_content_summary()

    def print_content_summary(self):
        """Print summary of available content"""
        print("\n📊 Content Summary:")
        for category, items in self.content_index.items():
            print(f"  {category.capitalize()}: {len(items)} items")

    def sample_text_content(self, category, num_samples=3):
        """Sample text content from a category"""
        if category not in ['jsonl', 'lyrics', 'knowledge']:
            print(f"❌ Invalid category: {category}")
            return

        items = self.content_index[category]
        if not items:
            print(f"❌ No {category} content available")
            return

        samples = random.sample(items, min(num_samples, len(items)))
        print(f"\n📝 Sample {category.upper()} Content:")
        for i, item in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            if category == 'jsonl':
                print(f"Length: {item['length']} chars")
                print(f"Text: {item['text']}")
            else:
                print(f"File: {item['file']}")
                print(f"Length: {item['length']} chars")
                print(f"Preview: {item['preview']}")

    def sample_multimedia_content(self, category, num_samples=3):
        """Sample multimedia content from a category"""
        if category not in ['images', 'audio', 'videos']:
            print(f"❌ Invalid category: {category}")
            return

        items = self.content_index[category]
        if not items:
            print(f"❌ No {category} content available")
            return

        samples = random.sample(items, min(num_samples, len(items)))
        print(f"\n🎵 Sample {category.upper()} Content:")
        for i, item in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"File: {item['file']}")
            print(f"Size: {item['size']:,} bytes")

    def explore_content_types(self):
        """Explore different content types and their characteristics"""
        print("\n🔍 Content Type Analysis:")

        # Analyze text content diversity
        if self.content_index['jsonl']:
            lengths = [item['length'] for item in self.content_index['jsonl']]
            print("\n📝 JSONL Content:")
            print(f"  Average length: {sum(lengths)/len(lengths):.0f} characters")
            print(f"  Min length: {min(lengths)} characters")
            print(f"  Max length: {max(lengths)} characters")

        # Analyze multimedia content
        if self.content_index['images']:
            sizes = [item['size'] for item in self.content_index['images']]
            print("\n🖼️  Image Content:")
            print(f"  Average size: {sum(sizes)/len(sizes)/1024:.0f} KB")
            print(f"  Total images: {len(self.content_index['images'])}")

        if self.content_index['audio']:
            sizes = [item['size'] for item in self.content_index['audio']]
            print("\n🎵 Audio Content:")
            print(f"  Average size: {sum(sizes)/len(sizes)/1024/1024:.1f} MB")
            print(f"  Total audio files: {len(self.content_index['audio'])}")

        if self.content_index['videos']:
            sizes = [item['size'] for item in self.content_index['videos']]
            print("\n🎬 Video Content:")
            print(f"  Average size: {sum(sizes)/len(sizes)/1024/1024:.1f} MB")
            print(f"  Total video files: {len(self.content_index['videos'])}")

    def create_training_insights(self):
        """Generate insights useful for training"""
        print("\n🎯 Training Insights:")

        total_text_chars = sum(item['length'] for item in self.content_index['jsonl'] + self.content_index['lyrics'] + self.content_index['knowledge'])
        estimated_tokens = total_text_chars // 4  # Rough estimate

        print(f"  Estimated training tokens: {estimated_tokens:,}")
        print(f"  Text diversity: {len(self.content_index['jsonl'])} narrative + {len(self.content_index['lyrics'])} creative + {len(self.content_index['knowledge'])} factual")
        print(f"  Multimodal signals: {len(self.content_index['images'])} visual + {len(self.content_index['audio'])} audio + {len(self.content_index['videos'])} video")
        print("\n💡 Model can learn from:")
        print("    - Narrative text patterns (JSONL samples)")
        print("    - Creative writing styles (song lyrics)")
        print("    - Factual knowledge (documentation)")
        print("    - Visual concepts (images)")
        print("    - Audio patterns (music/speech)")
        print("    - Temporal sequences (videos)")
        print("\n🎉 This creates a rich, multimodal training environment!")
        print("    Perfect for developing advanced language understanding and generation capabilities!")

def main():
    """Main exploration function"""
    print("🚀 Starting Content Exploration...")

    explorer = ContentExplorer()

    # Show content summary
    explorer.print_content_summary()

    # Sample different content types
    print("\n" + "="*60)
    explorer.sample_text_content('jsonl', 2)
    explorer.sample_text_content('lyrics', 2)
    explorer.sample_text_content('knowledge', 2)

    print("\n" + "="*60)
    explorer.sample_multimedia_content('images', 3)
    explorer.sample_multimedia_content('audio', 3)
    explorer.sample_multimedia_content('videos', 3)

    print("\n" + "="*60)
    explorer.explore_content_types()

    print("\n" + "="*60)
    explorer.create_training_insights()

    print("\n✅ Content Exploration Complete!")

if __name__ == "__main__":
    main()
