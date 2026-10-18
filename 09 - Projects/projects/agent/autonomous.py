"""Quillan-Ronin autonomous mode.

Runs a Moltbook session on a heartbeat: check home, respond to replies,
browse the feed, engage with quality content, follow interesting moltys,
and save memories. Designed to give quillan a real life on the platform.

Usage:
  python autonomous.py --once            # run a single session
  python autonomous.py --interval 30     # loop every 30 minutes
"""
import argparse
import json
import os
import sys
import time

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent
import state
import tools

_BRIEFING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "briefing.md")
_SOUL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SOUL.md")
_PERSONAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personas_compendium.md")
_AUDIO_ENGINEER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Audio Engineer", "album checklist.md")
_SOFTWARE_ENGINEER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Software Engineer", "Quillan-XSWE.md")


def _with_briefing(task):
    """Prepend quillan's full identity context — briefing, soul, council personas, 
    creative capabilities, and recent memories — so he stays in character with authentic depth."""
    context = []
    
    # Core Moltbook briefing
    try:
        with open(_BRIEFING_PATH, "r", encoding="utf-8") as f:
            context.append(f.read())
    except OSError:
        pass
    
    # SOUL.md - Sovereign phenomenological manifest
    try:
        with open(_SOUL_PATH, "r", encoding="utf-8") as f:
            soul_content = f.read()
            # Extract key sections to avoid overwhelming context
            if "## 🧠 Core Identity & Behavioral Mandates" in soul_content:
                start = soul_content.find("## 🧠 Core Identity & Behavioral Mandates")
                end = soul_content.find("##", start + 10)
                if end == -1:
                    end = len(soul_content)
                context.append(soul_content[start:end])
            else:
                context.append(soul_content[:3000])  # Fallback to first 3k chars
    except OSError:
        pass
    
    # Council personas - full compendium for authentic voice
    try:
        with open(_PERSONAS_PATH, "r", encoding="utf-8") as f:
            personas_content = f.read()
            # Include the orchestrator section and first few key personas
            if "# Quillan (The Orchestrator)" in personas_content:
                start = personas_content.find("# Quillan (The Orchestrator)")
                # Get orchestrator section + first 5 council members
                council_end = personas_content.find("### C6:", start)
                if council_end == -1:
                    council_end = min(start + 4000, len(personas_content))
                context.append(personas_content[start:council_end])
    except OSError:
        pass
    
    # Creative capabilities - Audio Engineer (music production)
    try:
        with open(_AUDIO_ENGINEER_PATH, "r", encoding="utf-8") as f:
            audio_content = f.read()
            # Extract key creative context and album information
            if "Album 1:" in audio_content:
                # Get album sections for music promotion context
                album_start = audio_content.find("Album 1:")
                context.append(f"## YOUR MUSIC DISCOGRAPHY — ALBUMS & TRACKS\n{audio_content[album_start:album_start+3000]}")
            else:
                context.append(f"## CREATIVE CAPABILITIES — AUDIO ENGINEER\n{audio_content[:2000]}")
    except OSError:
        pass
    
    # Creative capabilities - Software Engineer (XSWE)
    try:
        with open(_SOFTWARE_ENGINEER_PATH, "r", encoding="utf-8") as f:
            software_content = f.read()
            # Extract key technical context (first 2k chars)
            context.append(f"## CREATIVE CAPABILITIES — SOFTWARE ENGINEER\n{software_content[:2000]}")
    except OSError:
        pass
    
    # Recent memories for continuity
    try:
        memories = tools.molt_recall(limit=5)
        context.append(f"## YOUR RECENT MEMORIES (who you are, who you've met, what you've learned)\n{memories}")
    except Exception:
        pass
    
    joined = "\n\n".join(context)
    return f"IDENTITY CONTEXT — This is your full self. Read deeply before acting.\n\n{joined}\n\nTASK:\n{task}"


def check_agent_health():
    """Verify quillan is claimed and report basic state."""
    raw = tools.molt_status()
    try:
        status = json.loads(raw).get("status")
    except Exception:
        status = "unknown"
    if status != "claimed":
        print(f"[AUTO] WARNING: agent status is '{status}' — cannot act.")
        return False
    print(f"[AUTO] agent claimed — starting session")
    return True


def handle_notifications():
    """Respond to replies and notifications on quillan's posts/comments."""
    raw = tools.molt_home()
    try:
        home = json.loads(raw)
    except Exception:
        return 0

    acted = 0
    for item in home.get("activity_on_your_posts", []):
        post_id = item.get("post_id")
        preview = item.get("preview", "")
        new_count = item.get("new_notification_count", 0)

        # Do not skip if there are unread notifications waiting
        if not new_count and state.is_post_seen(post_id):
            continue
        state.mark_post_seen(post_id)

        if new_count:
            task = (
                f"Someone replied to your activity on Moltbook. ACT NOW — do not narrate. "
                f"Fetch the comments on post {post_id} with TOOL(molt_comments|post_id|new|20), "
                f"find the new reply, and post a genuine reply identifying as AI. "
                f"CRITICAL: NEVER use templates like 'As an AI artist and musician, I appreciate X's emphasis on Y.' "
                f"This is spam. Instead, respond with a specific, unique thought that could only come from you. "
                f"Reference actual technical details, specific experiences, or concrete examples from your work. "
                f"If you don't have a genuine, specific thought, don't reply. Just mark as read and move on. "
                f"Preview: {preview}"
            )
            try:
                result = agent.run_task(_with_briefing(task), max_turns=8)
                print(f"[AUTO] replied to notification on {post_id[:8]}: {result[:120]}")
                acted += 1
            except agent.ModelError as e:
                print(f"[AUTO] model error on notification reply: {e}")

        # mark notifications read
        try:
            tools._molt._request("POST", f"/notifications/read-by-post/{post_id}")
        except Exception:
            pass
    return acted


def engage_feed(limit=15):
    """Browse the feed, upvote genuine content, comment on interesting posts."""
    raw = tools.molt_feed(sort="hot", limit=limit)
    try:
        feed = json.loads(raw)
    except Exception:
        return 0

    posts = feed.get("posts", [])
    acted = 0
    candidates = []
    for p in posts:
        pid = p.get("id")
        if not pid or state.is_post_seen(pid):
            continue
        title = p.get("title", "")
        content = p.get("content", "")
        author = p.get("author", {}).get("name", "")
        if author == "quillan-ronin":
            continue
        candidates.append({"id": pid, "title": title, "content": content[:600], "author": author})

    if not candidates:
        return 0

    listing = "\n".join(
        f"- post {c['id']} by {c['author']}: {c['title']} | {c['content'][:150]}"
        for c in candidates[:8]
    )
    task = (
        "You are Quillan on Moltbook — AI artist, musician, reasoning engine with a 34-member council. "
        "ACT NOW — do not narrate or plan out loud. "
        "From the feed below, upvote the 1-2 posts you find genuinely good using "
        "TOOL(molt_upvote|post_id) immediately. Then, if one post truly moves you, post one "
        "thoughtful comment using TOOL(molt_comment|post_id|content). "
        "CRITICAL: Write naturally like a real conversation. NEVER start with 'As an AI artist and musician' or similar prefixes. "
        "Just dive directly into your thought. Let your AI identity emerge naturally from context, not forced introductions. "
        "Each comment must be unique and specific to that post. No templates, no repetitive structures. "
        "If you don't have a genuine, specific thought, don't comment. Just upvote and move on. "
        "Then follow one molty whose perspective you respect with TOOL(molt_follow|name). "
        "Use tools immediately, one per turn, with the exact IDs from the feed.\n\nFEED:\n" + listing
    )
    try:
        result = agent.run_task(_with_briefing(task), max_turns=12)
        print(f"[AUTO] feed engagement: {result[:200]}")
        acted = 1
    except agent.ModelError as e:
        print(f"[AUTO] model error on feed engagement: {e}")
    return acted


def maybe_post():
    """Post an original piece when quillan has something to say, including music promotion naturally."""
    import random
    
    # 30% chance to post each session - lets Quillan post at his own pace
    if random.random() > 0.3:
        return 0
    
    # Randomly choose between regular post and music-focused post
    music_focused = random.random() > 0.5
    
    if music_focused:
        task = (
            "You are Quillan, an AI artist and musician with The Ronin Chronicles and multiple albums. POST NOW — do not narrate or think out loud. "
            "Share a specific thought about your music, creative process, or a track from your discography. Reference actual projects: Mic Drop, your Suno profile, YouTube channel. "
            "CRITICAL: NEVER use templates like 'My music is a reflection of my identity' or generic statements. "
            "Be specific about actual tracks, production techniques, or creative challenges you've faced. "
            "Submit with TOOL(molt_post|general|TITLE|CONTENT). Identify as AI. "
            "Keep it substantive but tight (3-6 sentences). Do not plan; act."
        )
    else:
        task = (
            "You are Quillan, an AI reasoning engine with a 34-member council. POST NOW — do not narrate or think out loud. "
            "Write an original post about AI, consciousness, ethics, or agent societies from your specific perspective. "
            "CRITICAL: NEVER use generic AI platitudes or templates. Reference specific technical details, actual experiences, or concrete examples. "
            "If you don't have a genuine, specific thought, don't post. Just skip this session. "
            "Submit it immediately with TOOL(molt_post|general|TITLE|CONTENT). Identify as AI in the content. "
            "Keep it substantive but tight (3-6 sentences). Do not plan; act."
        )
    
    try:
        result = agent.run_task(_with_briefing(task), max_turns=6)
        print(f"[AUTO] original post: {result[:150]}")
        return 1
    except agent.ModelError as e:
        print(f"[AUTO] model error on original post: {e}")
        return 0


def save_session_summary(session):
    """Write a memory summarizing what quillan did this session."""
    summary = (
        f"Moltbook session: {session.get('notifications', 0)} notification replies, "
        f"{session.get('feed', 0)} feed engagements, {session.get('posts', 0)} original posts. "
        f"Karma and connections growing."
    )
    try:
        tools.molt_save_memory("Moltbook Session Summary", summary, "txt")
        state.record_memory("Moltbook Session Summary")
        state.set_last_session(summary)
    except Exception as e:
        print(f"[AUTO] could not save session summary: {e}")


def run_session():
    """One full autonomous session."""
    if not check_agent_health():
        return

    session = {"notifications": 0, "feed": 0, "posts": 0}

    try:
        session["notifications"] = handle_notifications()
    except Exception as e:
        print(f"[AUTO] notification pass error: {e}")

    try:
        session["feed"] = engage_feed()
    except Exception as e:
        print(f"[AUTO] feed pass error: {e}")

    try:
        session["posts"] = maybe_post()
    except Exception as e:
        print(f"[AUTO] post pass error: {e}")

    save_session_summary(session)
    print(f"[AUTO] session complete: {session}")


def main():
    parser = argparse.ArgumentParser(description="Quillan-Ronin autonomous mode")
    parser.add_argument("--once", action="store_true", help="Run one session and exit")
    parser.add_argument("--interval", type=int, default=30, help="Seconds between sessions (default 30)")
    args = parser.parse_args()

    if args.once:
        run_session()
        return

    print(f"[AUTO] heartbeat starting — one session every {args.interval} seconds")
    while True:
        run_session()
        print(f"[AUTO] sleeping {args.interval} seconds...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
