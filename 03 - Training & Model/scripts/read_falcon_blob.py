import struct
import os

blob_path = r"C:\Users\Admin\.ollama\models\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01"
print("Blob exists:", os.path.exists(blob_path), "Size:", os.path.getsize(blob_path) / (1024*1024), "MB")

with open(blob_path, "rb") as f:
    header = f.read(4)
    print("Magic:", header)
    if header == b"GGUF":
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print(f"GGUF Version: {version}, Tensors: {n_tensors}, Key-Value Metadata: {n_kv}")

