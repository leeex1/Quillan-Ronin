"""Check pending comments for error content."""
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

def check_pending_comments():
    """Check pending comments for error content."""
    client = MoltbookClient()
    
    # Load state to get pending comments
    state_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    
    pending_comments = state.get("comments_made", {})
    error_patterns = ["ERROR", "VERIFICATION FAILED", "VERIFICATION ERROR", "Traceback", "Exception", "failed", "incorrect", "Invalid answer", "was incorrect"]
    
    print("=== CHECKING PENDING COMMENTS FOR ERRORS ===\n")
    
    for comment_id, info in pending_comments.items():
        if info.get("status") == "pending":
            post_id = info.get("post_id")
            print(f"Comment ID: {comment_id}")
            print(f"Post ID: {post_id}")
            
            # Get the comment from Moltbook
            try:
                comments = client.get_comments(post_id, sort="new", limit=100)
                comment_data = comments if isinstance(comments, dict) else json.loads(comments)
                found = False
                for c in comment_data.get("comments", []):
                    if c.get("id") == comment_id:
                        content = c.get("content", "")
                        print(f"Content: {content[:200]}...")
                        
                        # Check for error patterns
                        has_error = False
                        for pattern in error_patterns:
                            if pattern.lower() in content.lower():
                                print(f"*** FOUND ERROR PATTERN: {pattern} ***")
                                has_error = True
                        
                        if has_error:
                            print(f"*** THIS COMMENT SHOULD BE DELETED ***")
                        
                        found = True
                        break
                if not found:
                    print(f"Comment not found on Moltbook (may not have been published)")
            except Exception as e:
                print(f"Error checking comment: {e}")
            print()

if __name__ == "__main__":
    check_pending_comments()
