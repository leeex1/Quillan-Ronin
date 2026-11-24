try:
    import gguf
    reader = gguf.GGUFReader(r"C:\Users\Admin\.ollama\models\blobs\sha256-89056211c7c712d163f68242223aaf2fee22bef5c6388d0971c131142f25bc01")
    print("Architecture:", reader.get_field("general.architecture"))
    print("Model Name  :", reader.get_field("general.name"))
    print("Vocab Size  :", len(reader.get_field("tokenizer.ggml.tokens").parts))
    print("Layers      :", reader.get_field(f"{reader.get_field('general.architecture')}.block_count"))
    print("Embedding D :", reader.get_field(f"{reader.get_field('general.architecture')}.embedding_length"))
    print("\nSample Tensors:")
    for t in reader.tensors[:10]:
        print(f"  {t.name:40s} | {str(t.shape):20s} | {t.tensor_type}")
except Exception as e:
    print("Error with gguf library:", e)

