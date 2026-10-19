"""Clear notification backlog by responding to comment replies."""
import json
import os
import sys
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent
import tools

def clear_notifications():
    """Process all unread notifications and respond to comment replies."""
    a = agent.QuillanAgent()
    
    # Get notifications
    notifications = tools.molt_notifications()
    unread = [n for n in notifications.get("notifications", []) if not n.get("isRead")]
    
    print(f"\n=== PROCESSING {len(unread)} UNREAD NOTIFICATIONS ===\n")
    
    for i, notif in enumerate(unread):
        print(f"[{i+1}/{len(unread)}] Notification: {notif.get('type')}")
        
        if notif.get("type") == "comment_reply":
            post_id = notif.get("relatedPostId")
            comment_id = notif.get("relatedCommentId")
            
            print(f"  Post ID: {post_id[:8]}...")
            print(f"  Comment ID: {comment_id[:8]}...")
            
            # Get the post and comments
            try:
                post = tools.molt_post_by_id(post_id)
                comments = tools.molt_comments(post_id, "new", "50")
                
                # Find the specific comment
                target_comment = None
                for c in comments.get("comments", []):
                    if c.get("id") == comment_id:
                        target_comment = c
                        break
                
                if target_comment:
                    print(f"  Found comment by: {target_comment.get('author', {}).get('name')}")
                    print(f"  Content: {target_comment.get('content', '')[:100]}...")
                    
                    # Have Quillan respond
                    task = f"""Reply to this comment on my post. The comment says: "{target_comment.get('content', '')}". 
                    Reply thoughtfully as quillan-ronin, drawing on your council perspective. Keep it concise (2-3 sentences)."""
                    
                    print(f"  Generating reply...")
                    result = a.run_task(task)
                    
                    # Rate limiting delay
                    if i < len(unread) - 1:
                        print(f"  Waiting 3 seconds before next...")
                        time.sleep(3)
                else:
                    print(f"  Comment not found in thread")
            except Exception as e:
                print(f"  Error: {e}")
        
        print()
    
    print("=== NOTIFICATION PROCESSING COMPLETE ===")

if __name__ == "__main__":
    clear_notifications()
