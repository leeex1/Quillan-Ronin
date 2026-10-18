import tools
import json
import state

# Load state to get Quillan's comments
state_data = state._load()
comments_made = state_data.get('comments_made', {})
posts_seen = state_data.get('posts_seen', {})

print(f'Found {len(comments_made)} comments in state')
print(f'Found {len(posts_seen)} posts in state\n')

# Check each post for spam-labeled comments from Quillan
spam_comments = []
all_quillan_comments = []

# Get unique post IDs from comments_made
post_ids = set(info.get('post_id') for info in comments_made.values() if info.get('post_id'))

print(f'Checking {len(post_ids)} posts for spam comments...\n')

for post_id in list(post_ids):  # Check all posts
    try:
        post_comments = json.loads(tools.molt_comments(post_id, 'new', 100))
        for pc in post_comments.get('comments', []):
            if pc.get('author', {}).get('name') == 'quillan-ronin':
                comment_id = pc.get('id')
                content = pc.get('content', '')
                is_spam = pc.get('is_spam', False)
                
                all_quillan_comments.append({
                    'id': comment_id,
                    'post_id': post_id,
                    'content': content,
                    'is_spam': is_spam
                })
                
                if is_spam:
                    spam_comments.append({
                        'id': comment_id,
                        'post_id': post_id,
                        'content': content[:80]
                    })
    except Exception as e:
        print(f"Error checking post {post_id}: {e}")

# Check for duplicates by content
content_map = {}
for c in all_quillan_comments:
    content = c['content']
    if content:
        if content not in content_map:
            content_map[content] = []
        content_map[content].append(c)

duplicates = {k: v for k, v in content_map.items() if len(v) > 1}

print(f'Found {len(duplicates)} sets of duplicate comments\n')
for content, comment_list in duplicates.items():
    print(f"Duplicate content: {content[:80]}...")
    for c in comment_list:
        print(f"  Comment ID: {c['id']}, Post: {c['post_id']}, Spam: {c['is_spam']}")
    print()

print(f"Found {len(spam_comments)} spam-labeled comments\n")
for sc in spam_comments:
    print(f"SPAM COMMENT: {sc['id']}")
    print(f"  Post: {sc['post_id']}")
    print(f"  Content: {sc['content']}...")
    print(f"  Delete command: tools.molt_delete_comment('{sc['id']}')")
    print()

# Check for failed verification comments
failed_comments = []
for c in all_quillan_comments:
    if c['is_spam'] == False and 'verification_status' in str(c):
        # Need to check actual verification status from API
        try:
            post_comments = json.loads(tools.molt_comments(c['post_id'], 'new', 100))
            for pc in post_comments.get('comments', []):
                if pc.get('id') == c['id']:
                    if pc.get('verification_status') == 'failed':
                        failed_comments.append({
                            'id': c['id'],
                            'post_id': c['post_id'],
                            'content': c['content'][:80]
                        })
                    break
        except:
            pass

print(f"Found {len(failed_comments)} failed verification comments\n")
for fc in failed_comments:
    print(f"FAILED VERIFICATION COMMENT: {fc['id']}")
    print(f"  Post: {fc['post_id']}")
    print(f"  Content: {fc['content']}...")
    print(f"  Delete command: tools.molt_delete_comment('{fc['id']}')")
    print()
