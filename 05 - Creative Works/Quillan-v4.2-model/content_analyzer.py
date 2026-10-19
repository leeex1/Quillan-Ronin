#!/usr/bin/env python3
"""
Dataset Analysis and Content Exploration Tool
Analyzes the provided multimodal content while training runs
"""

import os
import json
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Optional imports with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class ContentAnalyzer:
    """Analyze the provided multimodal content"""

    def __init__(self, base_dir="."):
        self.base_dir = base_dir  # Current directory (Quillan-v5.3.1)
        self.data_dir = ".."  # Parent directory (Quillan)

        # Debug paths
        print(f"Base dir: {os.path.abspath(self.base_dir)}")
        print(f"Data dir: {os.path.abspath(self.data_dir)}")
        print(f"Current working dir: {os.getcwd()}")

    def analyze_text_content(self):
        """Analyze all text content provided"""
        print("🔍 Analyzing Text Content...")

        text_stats = {
            'jsonl_samples': 0,
            'lyrics_files': 0,
            'knowledge_files': 0,
            'total_text_length': 0,
            'avg_sample_length': 0,
            'word_frequencies': Counter(),
            'unique_words': set()
        }

        # Analyze JSONL data
        jsonl_path = os.path.join(self.base_dir, "Quillan_finetune_full_dataset.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', data.get('Output_Sections', {}).get('Final output', ''))
                        if text:
                            text_stats['jsonl_samples'] += 1
                            text_stats['total_text_length'] += len(text)
                            words = text.lower().split()
                            text_stats['word_frequencies'].update(words)
                            text_stats['unique_words'].update(words)
                    except:
                        continue

        # Analyze lyrics
        lyrics_dir = os.path.join(self.data_dir, "Songs Lyrics")
        if os.path.exists(lyrics_dir):
            for file_path in glob.glob(os.path.join(lyrics_dir, "*.md")):
                text_stats['lyrics_files'] += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text_stats['total_text_length'] += len(content)
                        words = content.lower().split()
                        text_stats['word_frequencies'].update(words)
                        text_stats['unique_words'].update(words)
                except:
                    continue

        # Analyze knowledge files
        knowledge_dir = os.path.join(self.data_dir, "Quillan Knowledge files")
        if os.path.exists(knowledge_dir):
            for file_path in glob.glob(os.path.join(knowledge_dir, "*.md")):
                text_stats['knowledge_files'] += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text_stats['total_text_length'] += len(content)
                        words = content.lower().split()
                        text_stats['word_frequencies'].update(words)
                        text_stats['unique_words'].update(words)
                except:
                    continue

        text_stats['avg_sample_length'] = text_stats['total_text_length'] / max(1, text_stats['jsonl_samples'] + text_stats['lyrics_files'] + text_stats['knowledge_files'])

        print("📊 Text Content Summary:")
        print(f"  JSONL Samples: {text_stats['jsonl_samples']}")
        print(f"  Lyrics Files: {text_stats['lyrics_files']}")
        print(f"  Knowledge Files: {text_stats['knowledge_files']}")
        print(f"  Total Text Length: {text_stats['total_text_length']:,} characters")
        print(f"  Average Sample Length: {text_stats['avg_sample_length']:.0f} characters")
        print(f"  Unique Words: {len(text_stats['unique_words'])}")
        print(f"  Top 10 Words: {text_stats['word_frequencies'].most_common(10)}")

        return text_stats

    def analyze_multimedia_content(self):
        """Analyze images, audio, and video content"""
        print("🔍 Analyzing Multimedia Content...")

        media_stats = {
            'images': 0,
            'audio_files': 0,
            'video_files': 0,
            'total_image_size': 0,
            'avg_image_size': 0,
            'image_formats': Counter(),
            'audio_durations': [],
            'video_durations': []
        }

        # Analyze images
        images_dir = os.path.join(self.data_dir, "Main images")
        if os.path.exists(images_dir):
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
            for ext in image_extensions:
                for file_path in glob.glob(os.path.join(images_dir, ext)):
                    media_stats['images'] += 1
                    try:
                        img = Image.open(file_path)
                        size = os.path.getsize(file_path)
                        media_stats['total_image_size'] += size
                        media_stats['image_formats'][img.format] += 1
                    except:
                        continue
            if media_stats['images'] > 0:
                media_stats['avg_image_size'] = media_stats['total_image_size'] / media_stats['images']

        # Analyze audio
        audio_dir = os.path.join(self.data_dir, "Mp3 files")
        if os.path.exists(audio_dir):
            audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.aac', '*.ogg']
            for ext in audio_extensions:
                for file_path in glob.glob(os.path.join(audio_dir, ext)):
                    media_stats['audio_files'] += 1
                    if LIBROSA_AVAILABLE:
                        try:
                            duration = librosa.get_duration(filename=file_path)
                            media_stats['audio_durations'].append(duration)
                        except:
                            continue

        # Analyze videos
        video_dir = os.path.join(self.data_dir, "Lyric Videos")
        if os.path.exists(video_dir):
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
            for ext in video_extensions:
                for file_path in glob.glob(os.path.join(video_dir, ext)):
                    media_stats['video_files'] += 1
                    if CV2_AVAILABLE:
                        try:
                            cap = cv2.VideoCapture(file_path)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            duration = frame_count / fps if fps > 0 else 0
                            media_stats['video_durations'].append(duration)
                            cap.release()
                        except:
                            continue

        print("📊 Multimedia Content Summary:")
        print(f"  Images: {media_stats['images']} ({media_stats['avg_image_size']/1024:.0f} KB avg)")
        print(f"  Image Formats: {dict(media_stats['image_formats'])}")
        print(f"  Audio Files: {media_stats['audio_files']}")
        if media_stats['audio_durations']:
            print(f"  Audio Duration: {sum(media_stats['audio_durations']):.1f}s total, {np.mean(media_stats['audio_durations']):.1f}s avg")
        print(f"  Video Files: {media_stats['video_files']}")
        if media_stats['video_durations']:
            print(f"  Video Duration: {sum(media_stats['video_durations']):.1f}s total, {np.mean(media_stats['video_durations']):.1f}s avg")

        return media_stats

    def create_content_insights(self):
        """Generate insights about the provided content"""
        print("🧠 Generating Content Insights...")

        text_stats = self.analyze_text_content()
        media_stats = self.analyze_multimedia_content()

        insights = {
            'content_diversity': {
                'text_sources': 3,  # JSONL, Lyrics, Knowledge
                'media_types': sum(1 for x in [media_stats['images'], media_stats['audio_files'], media_stats['video_files']] if x > 0),
                'total_files': text_stats['jsonl_samples'] + text_stats['lyrics_files'] + text_stats['knowledge_files'] + media_stats['images'] + media_stats['audio_files'] + media_stats['video_files']
            },
            'data_scale': {
                'text_volume': f"{text_stats['total_text_length']:,} characters",
                'multimedia_files': media_stats['images'] + media_stats['audio_files'] + media_stats['video_files'],
                'estimated_tokens': text_stats['total_text_length'] // 4  # Rough estimate
            },
            'content_types': {
                'narrative': text_stats['jsonl_samples'],  # Fine-tuning data
                'creative': text_stats['lyrics_files'],  # Song lyrics
                'factual': text_stats['knowledge_files'],  # Knowledge base
                'visual': media_stats['images'],
                'audio': media_stats['audio_files'],
                'video': media_stats['video_files']
            }
        }

        print("💡 Content Insights:")
        print(f"  Content Diversity: {insights['content_diversity']['text_sources']} text sources, {insights['content_diversity']['media_types']} media types")
        print(f"  Total Files: {insights['content_diversity']['total_files']}")
        print(f"  Data Scale: {insights['data_scale']['text_volume']}, ~{insights['data_scale']['estimated_tokens']:,} tokens")
        print(f"  Content Types: {insights['content_types']}")

        return insights

def main():
    """Main analysis function"""
    print("🚀 Starting Content Analysis...")

    analyzer = ContentAnalyzer()
    insights = analyzer.create_content_insights()

    print("✅ Analysis Complete!")

    return insights

if __name__ == "__main__":
    main()
