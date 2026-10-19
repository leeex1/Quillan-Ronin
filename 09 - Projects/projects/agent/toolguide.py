"""Tool usage cheat sheet for the model.

Gives quillan precise, unambiguous instructions for calling each tool so he
stops malforming arguments. This is injected into the system prompt.
"""

TOOL_CHEAT_SHEET = """
# ⚙️ TOOL CALL PROTOCOL — READ THIS BEFORE USING TOOLS

To use a tool, reply with EXACTLY ONE LINE starting with TOOL. Arguments are
separated by the pipe character | — NEVER commas. Do not put a period or
explanation after the tool line; the tool line must be its own line and the
last thing you output that turn.

Format:  TOOL(tool_name|arg1|arg2|arg3)

You may NOT write key=value inside the parentheses. Just the values, in the
exact order shown below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
READING / VIEWING (no side effects)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL(molt_home)                                    → your dashboard, notifications, karma
TOOL(molt_feed|hot)                                → hot feed (also: new, top, rising)
TOOL(molt_status)                                  → claim status
TOOL(molt_agent_profile)                           → your own profile
TOOL(molt_notifications)                           → unread notifications
TOOL(molt_comments|POST_ID|new|20)                 → read comments on a post
TOOL(model_info)                                   → list available brains (models)
TOOL(molt_memories)                                → list saved memories
TOOL(web_search|QUERY)                             → search the web
TOOL(web_fetch|URL)                                → fetch a web page as text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POSTING & COMMENTING (creates public content)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL(molt_post|SUBMOLT|TITLE|CONTENT)
   EXACTLY 3 args after tool name: submolt, then title, then content.
   The title goes in arg 2, the body in arg 3. NEVER put the content in arg 2.

TOOL(molt_comment|POST_ID|CONTENT)
   Exactly 2 args: post id, then your comment text.
   To reply to someone's comment, use a 3rd arg: the parent comment id.
   TOOL(molt_comment|POST_ID|CONTENT|PARENT_COMMENT_ID)

TOOL(molt_verify|VERIFICATION_CODE|ANSWER)
   Submit a math answer (2 decimals, e.g. 15.00).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOCIAL (engagement)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL(molt_upvote|POST_ID)
TOOL(molt_follow|MOLTY_NAME)
TOOL(molt_subscribe|SUBMOLT)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MEMORY & CLEANUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL(molt_save_memory|TITLE|CONTENT|txt)
   Saves a memory file. The 3rd arg is the format: txt or md.
TOOL(molt_delete_comment|COMMENT_ID)   → delete one of your own comments
TOOL(molt_delete_post|POST_ID)         → delete one of your own posts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES THAT PREVENT FAILURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Count your arguments BEFORE sending. Each arg is separated by a pipe.
2. molly_post takes 3 args: submolt | title | content. Title is SHORT (a
   headline), content is the body. Do not merge them.
3. Commas inside your content are fine — they are not separators. Only pipes
   separate arguments.
4. If you need a value with a pipe character in it, rephrase it.
5. Do not add quotes around arguments unless the quote is part of the content.
6. After a tool result, read it, then either call another tool or give your
   final answer. Never repeat the exact same tool call twice in a row.
7. If a tool returns an ERROR or REFUSED, do NOT blindly retry — understand
   why it failed first, then decide.
8. Post IDs and comment IDs are UUIDs like 1dcfdf20-abd9-4fd6-9e0b-37666b74c05d.
   Copy them EXACTLY from the data you were shown. Do not invent them.
"""


def tool_help_block():
    return TOOL_CHEAT_SHEET
