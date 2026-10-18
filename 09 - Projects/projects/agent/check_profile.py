"""Check Quillan's recent posts and comments for errors."""
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

def check_recent_activity():
    """Check recent posts and comments for errors."""
    client = MoltbookClient()
    
    # Get agent profile
    try:
        profile = client.me()
        print("=== QUILLAN PROFILE ===")
        print(f"Name: {profile.get('agent', {}).get('name')}")
        print(f"Karma: {profile.get('agent', {}).get('karma')}")
        print()
    except Exception as e:
        print(f"Error getting profile: {e}")
    
    # Get recent feed
    try:
        feed = client.get_feed(sort="new", limit=50)
        print("=== RECENT FEED (checking for Quillan's posts) ===")
        for post in feed.get("posts", [])[:20]:
            author = post.get('author', {}).get('name')
            if author == 'quillan-ronin':
                print(f"ID: {post.get('id')}")
                print(f"Title: {post.get('title', 'N/A')[:80]}")
                print(f"Content: {post.get('content', 'N/A')[:150]}...")
                print(f"Karma: {post.get('karma')}")
                print()
    except Exception as e:
        print(f"Error getting feed: {e}")
    
    # Check for error patterns in content
    error_patterns = ["ERROR", "VERIFICATION FAILED", "VERIFICATION ERROR", "Traceback", "Exception", "failed", "incorrect", "Invalid answer"]
    
    print("=== CHECKING FOR ERROR PATTERNS IN FEED ===")
    
    # Check feed posts
    try:
        feed = client.get_feed(sort="new", limit=100)
        for post in feed.get("posts", []):
            if post.get('author', {}).get('name') == 'quillan-ronin':
                content = post.get('content', '') + post.get('title', '')
                for pattern in error_patterns:
                    if pattern.lower() in content.lower():
                        print(f"FOUND ERROR IN POST: {post.get('id')}")
                        print(f"Pattern: {pattern}")
                        print(f"Title: {post.get('title', 'N/A')[:80]}")
                        print(f"Content: {content[:300]}...")
                        print()
    except Exception as e:
        print(f"Error checking posts for errors: {e}")

if __name__ == "__main__":
    check_recent_activity()
