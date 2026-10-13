import asyncio
import edge_tts
import os
import re
from pathlib import Path

def clean_markdown(text):
    """Remove markdown formatting to get clean text for TTS"""
    # Remove headers (#)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove bold/italic (*, _, **, __)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Remove links [text](url) -> keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()

def detect_content_type(text):
    """Detect the type of content for appropriate TTS styling"""
    text_lower = text.lower()
    
    # Dialogue detection
    if '"' in text or "'" in text:
        return "dialogue"
    
    # Action/excitement detection
    action_words = ['shouted', 'screamed', 'ran', 'fought', 'attacked', 'exploded', 
                   'crashed', 'smashed', 'leaped', 'dashed', 'thundered', 'roared']
    if any(word in text_lower for word in action_words):
        return "action"
    
    # Emotional/dramatic detection
    emotional_words = ['whispered', 'cried', 'sobbed', 'trembled', 'feared', 
                      'horrified', 'despair', 'grief', 'sorrow', 'love', 'passion']
    if any(word in text_lower for word in emotional_words):
        return "emotional"
    
    # Descriptive/narrative
    return "narrative"

def get_tts_params(content_type):
    """Get TTS parameters based on content type"""
    if content_type == "dialogue":
        # Natural conversation pace with emphasis
        return {"rate": "+0%", "pitch": "+10Hz"}
    elif content_type == "action":
        # Faster pace for action scenes
        return {"rate": "+10%", "pitch": "+20Hz"}
    elif content_type == "emotional":
        # Slower, deeper for emotional moments
        return {"rate": "-15%", "pitch": "-10Hz"}
    else:
        # Natural narrative pace
        return {"rate": "+0%", "pitch": "+0Hz"}

def split_text_chunks(text, max_chars=2500):
    """Split text into chunks for TTS processing with awareness of content"""
    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if len(current_chunk) + len(paragraph) < max_chars:
            current_chunk += paragraph + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

async def text_to_speech(text, output_file, voice="en-US-ChristopherNeural"):
    """Convert text to speech using edge-tts with rate/pitch parameters"""
    content_type = detect_content_type(text)
    params = get_tts_params(content_type)
    
    communicate = edge_tts.Communicate(text, voice, rate=params["rate"], pitch=params["pitch"])
    await communicate.save(output_file)

async def convert_book_to_audiobook(input_file, output_file, voice="en-US-ChristopherNeural"):
    """Convert a markdown book to audiobook"""
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_text = f.read()
    
    print("Cleaning markdown formatting...")
    clean_text = clean_markdown(markdown_text)
    
    print(f"Text length: {len(clean_text)} characters")
    
    print("Splitting into chunks...")
    chunks = split_text_chunks(clean_text)
    print(f"Created {len(chunks)} chunks")
    
    # Create temporary directory for chunk files
    temp_dir = Path("temp_audio_chunks")
    temp_dir.mkdir(exist_ok=True)
    
    # Convert each chunk to audio
    audio_files = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        chunk_file = temp_dir / f"chunk_{i:04d}.mp3"
        print(f"Converting chunk {i+1}/{len(chunks)}...")
        await text_to_speech(chunk, str(chunk_file), voice)
        audio_files.append(chunk_file)
    
    # Combine all chunks into a single file
    print("Combining audio chunks...")
    with open(output_file, 'wb') as outfile:
        for audio_file in audio_files:
            if audio_file.exists():
                with open(audio_file, 'rb') as infile:
                    outfile.write(infile.read())
            else:
                print(f"Warning: Missing chunk {audio_file.name}, skipping...")
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for audio_file in audio_files:
        audio_file.unlink()
    temp_dir.rmdir()
    
    print(f"Audiobook saved to: {output_file}")

async def main():
    """Main function to convert all books"""
    books = [
        "Book 1 - Twisted Destiny.md",
        "Book 2 - Rise of Ascension.md",
        "Book 3 - Battle Grandeur.md",
        "Book 4 - Fall of Empires.md",
        "Book 5 - Shadows That Speak.md"
    ]
    
    # Premium neural voices with natural storytelling quality
    voice = "en-US-ChristopherNeural"  # Warm, natural male narrator
    # Alternative voices: 
    # "en-US-EricNeural" (calm, professional male)
    # "en-US-GuyNeural" (friendly male)
    # "en-US-JennyNeural" (natural female)
    # "en-US-MichelleNeural" (warm female)
    
    for book in books:
        if os.path.exists(book):
            output_file = book.replace(".md", ".mp3")
            print(f"\n{'='*60}")
            print(f"Converting: {book}")
            print(f"Output: {output_file}")
            print(f"{'='*60}")
            
            await convert_book_to_audiobook(book, output_file, voice)
            
            print(f"[OK] Completed: {output_file}")
        else:
            print(f"[X] File not found: {book}")

if __name__ == "__main__":
    asyncio.run(main())
