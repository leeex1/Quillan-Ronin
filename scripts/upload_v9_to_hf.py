import os
from huggingface_hub import HfApi

def upload_v9():
    api = HfApi()
    repo_id = "CrashOverrideX/Quillan-Ronin"
    m_path = r"C:\02_QUILLAN\exports\quillan-1b-sovereign-wsp-v3-q4_k_m.gguf"

    print(f"Uploading winning Model A Flagship Polymath v9 weights ({os.path.getsize(m_path)/(1024*1024):.2f} MB) to {repo_id}...")
    api.upload_file(
        path_or_fileobj=m_path,
        path_in_repo="quillan-1b-sovereign-wsp-v3-q4_k_m.gguf",
        repo_id=repo_id,
        commit_message="feat(model): update Model A 1.2B Flagship to Polymath v9 (88.24% accuracy, 7.92 PPL)"
    )
    print("Upload complete!")

if __name__ == "__main__":
    upload_v9()
