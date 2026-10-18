"""Check pending posts and comments to see if they were actually published."""
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

def check_pending_entries():
    """Check all pending comments and posts."""
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
    
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    
    client = MoltbookClient()
    
    # Check pending comments
    pending_comments = state.get("comments_made", {})
    print(f"\n=== PENDING COMMENTS ({len(pending_comments)}) ===\n")
    
    for comment_id, info in pending_comments.items():
        if info.get("status") == "pending":
            post_id = info.get("post_id")
            print(f"Comment ID: {comment_id}")
            print(f"Post ID: {post_id}")
            print(f"Timestamp: {info.get('ts')}")
            print(f"Status: {info.get('status')}")
            
            # Try to get the comment from Moltbook
            try:
                comments = client.get_comments(post_id, sort="new", limit=100)
                comment_data = comments if isinstance(comments, dict) else json.loads(comments)
                found = False
                for c in comment_data.get("comments", []):
                    if c.get("id") == comment_id:
                        print(f"✓ Found on Moltbook - published!")
                        found = True
                        break
                if not found:
                    print(f"✗ NOT found on Moltbook - verification likely failed")
            except Exception as e:
                print(f"Error checking: {e}")
            print()
    
    # Check pending posts
    pending_posts = state.get("posts_made", {})
    print(f"\n=== PENDING POSTS ({len(pending_posts)}) ===\n")
    
    for post_id, info in pending_posts.items():
        if info.get("status") == "pending":
            print(f"Post ID: {post_id}")
            print(f"Timestamp: {info.get('ts')}")
            print(f"Status: {info.get('status')}")
            
            # Try to get the post from Moltbook
            try:
                post = client.get_post(post_id)
                post_data = post if isinstance(post, dict) else json.loads(post)
                if post_data.get("success"):
                    print(f"✓ Found on Moltbook - published!")
                else:
                    print(f"✗ NOT found on Moltbook - verification likely failed")
            except Exception as e:
                print(f"Error checking: {e}")
            print()

if __name__ == "__main__":
    check_pending_entries()
