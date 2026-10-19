from huggingface_hub import list_repo_commits
commits = list_repo_commits("CrashOverrideX/Quillan-Ronin", repo_type="model")
for c in commits:
    print(f"{c.created_at} :: {c.commit_id[:8]} :: {c.title}", flush=True)
