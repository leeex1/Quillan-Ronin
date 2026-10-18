"""Delete a problematic comment."""
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from moltbook import MoltbookClient

def delete_comment(comment_id):
    """Delete a comment by ID."""
    client = MoltbookClient()
    
    try:
        result = client.delete_comment(comment_id)
        print(f"=== DELETE RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except Exception as e:
        print(f"Error deleting comment: {e}")
        return None

if __name__ == "__main__":
    import json
    # Delete the problematic "Deleted comment" placeholder
    delete_comment("fcb06f31-0805-4b8e-8a7c-b95288183824")
