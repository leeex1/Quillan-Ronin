import sys
import os

libs = ['pypdf', 'fitz', 'pdfplumber', 'pypdf2', 'whisper', 'torchaudio', 'librosa', 'moviepy', 'cv2']
print("=== INSTALLED LIBRARIES ===")
for mod in libs:
    try:
        __import__(mod)
        print(f"  [+] {mod}: INSTALLED")
    except Exception as e:
        print(f"  [-] {mod}: NOT INSTALLED ({e})")

print("\n=== SEARCHING MEDIA & PDF FILES IN C:\\02_QUILLAN and USER DIRECTORIES ===")
extensions = ('.pdf', '.mp3', '.wav', '.m4a', '.mp4', '.mkv', '.avi')
found_files = {ext: [] for ext in extensions}

search_dirs = [r"C:\02_QUILLAN", r"C:\Users\Admin\Downloads", r"C:\Users\Admin\Documents", r"C:\Users\Admin\Desktop"]
for sdir in search_dirs:
    if os.path.exists(sdir):
        for root, dirs, files in os.walk(sdir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in found_files:
                    found_files[ext].append(os.path.join(root, file))

for ext, file_list in found_files.items():
    print(f"  Found {len(file_list)} files with extension '{ext}'")
    for f in file_list[:5]:
        print(f"    - {f}")
