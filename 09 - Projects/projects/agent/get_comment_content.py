"""Get full content of a specific comment."""
import json
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moltbook import MoltbookClient

def get_comment_content(comment_id, post_id):
    """Get full content of a specific comment."""
    client = MoltbookClient()
    
    try:
        comments = client.get_comments(post_id, sort="new", limit=100)
        comment_data = comments if isinstance(comments, dict) else json.loads(comments)
        for c in comment_data.get("comments", []):
            if c.get("id") == comment_id:
                print(f"=== FULL COMMENT CONTENT ===")
                print(f"Comment ID: {c.get('id')}")
                print(f"Author: {c.get('author', {}).get('name')}")
                print(f"Content:\n{c.get('content')}")
                print(f"Karma: {c.get('karma')}")
                return
        print("Comment not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check the problematic comment
    get_comment_content("8daad499-2a6b-49a4-9739-a8770368a1f6", "5b65ea34-558e-4478-8c50-1ff5c5ffa8f9")
