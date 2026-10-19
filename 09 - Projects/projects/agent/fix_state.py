"""Update state.json to mark actually published entries as success."""
import json
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tools
from moltbook import MoltbookClient

def fix_state():
    """Update state.json with correct statuses for published entries."""
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
    
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    
    client = MoltbookClient()
    
    # Fix pending comments
    pending_comments = state.get("comments_made", {})
    print(f"\n=== FIXING PENDING COMMENTS ===\n")
    
    for comment_id, info in pending_comments.items():
        if info.get("status") == "pending":
            post_id = info.get("post_id")
            
            # Check if actually published
            try:
                comments = client.get_comments(post_id, sort="new", limit=100)
                comment_data = comments if isinstance(comments, dict) else json.loads(comments)
                found = False
                for c in comment_data.get("comments", []):
                    if c.get("id") == comment_id:
                        print(f"✓ Comment {comment_id[:8]}... is published - updating state")
                        state["comments_made"][comment_id]["status"] = "success"
                        found = True
                        break
                if not found:
                    print(f"✗ Comment {comment_id[:8]}... NOT published - keeping as pending")
            except Exception as e:
                print(f"Error checking comment {comment_id[:8]}...: {e}")
    
    # Fix pending posts
    pending_posts = state.get("posts_made", {})
    print(f"\n=== FIXING PENDING POSTS ===\n")
    
    for post_id, info in pending_posts.items():
        if info.get("status") == "pending":
            # Check if actually published
            try:
                post = client.get_post(post_id)
                post_data = post if isinstance(post, dict) else json.loads(post)
                if post_data.get("success"):
                    print(f"✓ Post {post_id[:8]}... is published - updating state")
                    state["posts_made"][post_id]["status"] = "success"
                else:
                    print(f"✗ Post {post_id[:8]}... NOT published - keeping as pending")
            except Exception as e:
                print(f"Error checking post {post_id[:8]}...: {e}")
    
    # Save updated state
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)
    
    print(f"\n=== STATE UPDATED ===")
    print(f"Saved to: {state_path}")

if __name__ == "__main__":
    fix_state()
