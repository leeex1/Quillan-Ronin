import tools
import json

# Try to get posts from search first
try:
    search = json.loads(tools.molt_search('quillan-ronin', 100))
    posts = [item for item in search.get('items', []) if item.get('type') == 'post']
    print(f'Found {len(posts)} posts from search\n')
    
    for p in posts:
        post_id = p.get('post_id')
        title = p.get('title', 'No title')
        print(f"{post_id}: {title}")
        
        # Get full post details to check spam/verification status
        try:
            full_post = json.loads(tools.molt_get_post(post_id))
            is_spam = full_post.get('is_spam', False)
            verification = full_post.get('verification_status', 'unknown')
            print(f"  Spam: {is_spam}, Verified: {verification}")
        except Exception as e:
            print(f"  Error getting details: {e}")
        print()
except Exception as e:
    print(f"Error: {e}")
