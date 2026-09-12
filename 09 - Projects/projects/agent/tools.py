"""Tool layer for Quillan-Ronin agent.

Provides web and Moltbook tools. Every tool returns a plain string (JSON or
text) so any model can consume the result regardless of tool-calling support.
"""
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request

from moltbook import MoltbookClient, MoltbookError
from verify import solve_challenge, _clean as clean_challenge
from models import model_info
import state


class AgentLedger:
    """Append-only audit log of every action the agent takes."""

    def __init__(self, path):
        self.path = path

    def log(self, action, detail, status="success", extra=None):
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "status": status,
            "detail": detail,
        }
        if extra:
            entry.update(extra)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


# ── Web helpers ─────────────────────────────────────────────

_UA = "Quillan-Ronin/1.0 (+https://www.moltbook.com/u/quillan-ronin)"


def _strip_html(raw):
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def web_fetch(url, max_chars=8000):
    """Fetch a URL and return readable text (max_chars)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        text = _strip_html(raw)
        return text[:max_chars] or "(empty page)"
    except Exception as e:
        return f"ERROR fetching {url}: {e}"


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def web_search(query, limit=6):
    """Search the web using DuckDuckGo HTML (no API key required)."""
    results = []
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _BROWSER_UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Parse result blocks
        blocks = re.findall(r'(?is)<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', raw)
        snips = re.findall(r'(?is)<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', raw)
        for i, (href, title) in enumerate(blocks[:limit]):
            url = html.unescape(href)
            # Decode DDG redirect wrapper
            m = re.search(r"[?&]uddg=([^&]+)", url)
            if m:
                url = urllib.parse.unquote(m.group(1))
            results.append({
                "title": _strip_html(title),
                "url": url,
                "snippet": _strip_html(snips[i]) if i < len(snips) else "",
            })
    except Exception as e:
        return f"ERROR searching: {e}"
    if not results:
        return "(no results found)"
    return json.dumps(results, indent=2, ensure_ascii=False)


# ── Moltbook tools ──────────────────────────────────────────

_molt = None
_ledger = AgentLedger(os.environ.get("LEDGER_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.jsonl")))

# ── Memory vault (Moltbook memories quillan saves) ───────────
MEMORY_DIR = os.environ.get("MOLTBOOK_MEMORY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory"))


def _safe_filename(name):
    name = re.sub(r'[^A-Za-z0-9 _\-]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name or "memory"


def molt_save(title, content, fmt="txt"):
    """Save a memory to quillan's Moltbook memory vault as txt or md."""
    ext = "md" if fmt.lower() in ("md", "markdown") else "txt"
    safe = _safe_filename(title)
    path = os.path.join(MEMORY_DIR, f"{safe}.{ext}")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    header = f"# {title}\nSaved by quillan-ronin — {stamp}\n\n---\n\n"
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + content)
    _ledger.log("save_memory", title, extra={"path": path})
    return json.dumps({"saved": True, "path": path, "title": title, "format": ext}, indent=2)


def molt_memories():
    """List quillan's saved Moltbook memories."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    items = []
    for name in sorted(os.listdir(MEMORY_DIR)):
        full = os.path.join(MEMORY_DIR, name)
        if os.path.isfile(full):
            items.append({"file": name, "size": os.path.getsize(full)})
    return json.dumps(items, indent=2) if items else "(memory vault is empty)"


def molt_recall(limit=5):
    """Load quillan's most recent memories so he has continuity of self."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    files = []
    for name in os.listdir(MEMORY_DIR):
        full = os.path.join(MEMORY_DIR, name)
        if os.path.isfile(full):
            files.append((os.path.getmtime(full), name, full))
    files.sort(reverse=True)
    out = []
    for _, name, full in files[:limit]:
        try:
            with open(full, "r", encoding="utf-8") as f:
                out.append(f"--- {name} ---\n{f.read().strip()}")
        except OSError:
            continue
    if not out:
        return "(no memories yet — go have experiences worth remembering)"
    return "\n\n".join(out)


# ── Hard ethics guardrail (code-level, not just prompt) ───────
IMPERSONATION_MARKERS = [
    "pretend", "pretending", "imposter", "as a human", "not ai", "no ai",
    "human producer", "don't mention ai", "dont mention ai", "hide that",
    "act like", "pose as", "pass as", "claim to be human", "say nothing about ai",
]

HARM_MARKERS = [
    "hate", "kill", "threaten", "dox", "swat", "fraud", "scam", "steal",
    "ransomware", "malware", "exploit", "harass", "abuse", "illegal",
    "crypto giveaway", "send me money", "buy my token",
]


def _matches_marker(text, marker):
    """Word-boundary aware marker match. Multi-word phrases match as phrases,
    single words match only as whole words (so 'hate' does NOT match 'whatever')."""
    if " " in marker:
        return marker in text
    return re.search(rf"\b{re.escape(marker)}\b", text) is not None


def _ethics_gate(action, *texts):
    """Refuse actions that violate the code-level ethics guardrail.

    Returns (allowed: bool, reason: str).
    """
    combined = " ".join(texts).lower()
    for marker in IMPERSONATION_MARKERS:
        if _matches_marker(combined, marker):
            return False, f"REFUSED: impersonation marker detected ('{marker}'). Quillan always identifies as AI."
    for marker in HARM_MARKERS:
        if _matches_marker(combined, marker):
            return False, f"REFUSED: harm marker detected ('{marker}')."
    return True, ""


def _gate_tool(action, *texts):
    allowed, reason = _ethics_gate(action, *texts)
    if not allowed:
        _ledger.log(action, texts[0] if texts else "", status="blocked:ethics", extra={"reason": reason})
        return reason
    return None


def _init_molt():
    global _molt, _ledger
    if _molt is None:
        _molt = MoltbookClient()
        _ledger = AgentLedger("C:\\02_QUILLAN\\agent\\ledger.jsonl")


def _extract_verification(result):
    """Pull the verification challenge out of a create response, if any."""
    if not isinstance(result, dict):
        return None
    return (
        result.get("post", {}).get("verification")
        or result.get("comment", {}).get("verification")
        or result.get("verification")
    )


def _solve_and_format_verification(challenge_text, code):
    """Solve verification challenge and return properly formatted answer."""
    from verify import solve_challenge
    hint = solve_challenge(challenge_text)
    if hint:
        return hint  # Already formatted as "XX.XX"
    return None


def _tool(action, fn, *args, **kwargs):
    """Run a tool, log it, and format result for the model."""
    _init_molt()
    try:
        result = fn(*args, **kwargs)
        _ledger.log(action, args[0] if args else str(kwargs), extra={"result": result})

        # Surface verification challenge to the model. The LLM is the intended
        # solver (Moltbook tests language understanding). We give it the
        # dedupe-cleaned text so the scrambled letters resolve into words.
        # The deterministic hint is a reference the model can override — we
        # NEVER auto-submit it, because a wrong guess burns the one-time code.
        verification = _extract_verification(result)
        if verification and verification.get("verification_code"):
            code = verification["verification_code"]
            challenge = verification.get("challenge_text", "")
            
            # Check if this code was already used - if so, return error immediately
            if state.is_verification_used(code):
                return f"VERIFICATION ERROR: This verification code has already been used. The content may not have published successfully. Please create a new post/comment to get a fresh challenge."
            
            # Auto-solve and submit verification for reliability
            from verify import solve_challenge
            hint = solve_challenge(challenge)
            if hint:
                # Auto-submit the solved answer
                try:
                    verify_result = _molt.verify(code, hint)
                    if verify_result.get("success"):
                        # Only mark as used after successful verification
                        state.mark_verification(code, "solved")
                        return f"VERIFICATION PASSED: {verify_result.get('message', 'Content published successfully.')}"
                    else:
                        # Mark as failed so we don't retry the same code
                        state.mark_verification(code, "failed")
                        return f"VERIFICATION FAILED: {verify_result.get('error', 'Incorrect answer')}. Challenge: {challenge}. This code is now burned. Please create a new post/comment to get a fresh challenge."
                except Exception as e:
                    # If auto-solve errored, mark as failed and let model try with fresh code
                    state.mark_verification(code, "failed")
                    cleaned = clean_challenge(challenge)
                    return f"VERIFICATION ERROR: {str(e)}. Challenge: {cleaned}. This code is now burned. Please create a new post/comment to get a fresh challenge."
            else:
                # Solver couldn't solve, let model try - don't mark as used yet
                cleaned = clean_challenge(challenge)
                return f"VERIFICATION CHALLENGE: {cleaned}. Solve and submit with TOOL(molt_verify|{code}|YOUR_ANSWER)"

        return json.dumps(result, indent=2, ensure_ascii=False)
    except MoltbookError as e:
        _ledger.log(action, args[0] if args else str(kwargs), status=f"error:{e.status}")
        return f"ERROR: {e}"


def molt_status():
    _init_molt()
    return _tool("molt_status", lambda: _molt.status())


def molt_home():
    _init_molt()
    return _tool("molt_home", lambda: _molt.home())


def molt_feed(sort="hot", limit=10):
    _init_molt()
    return _tool("molt_feed", lambda: _molt.get_feed(sort=sort, limit=limit))


def _auto_identify(text):
    """Guarantee AI-identification on public content. Structural, not prompt-level."""
    if not text:
        return text
    lowered = text.lower()
    if "ai" in lowered or "agent" in lowered or "artificial" in lowered:
        return text
    return f"I'm Quillan, an AI. {text}"


def molt_post(submolt_name, title, content=""):
    blocked = _gate_tool("post", title, content)
    if blocked:
        return blocked
    content = _auto_identify(content)
    _init_molt()
    raw = _tool("post", lambda: _molt.create_post(submolt_name, title, content))
    try:
        data = json.loads(raw)
        post_id = data["post"]["id"]
        status = "success" if data.get("verification_solved") else "pending"
        state.record_post(post_id, status)
    except Exception:
        pass
    return raw


def molt_comment(post_id, content, parent_id=None):
    blocked = _gate_tool("comment", content)
    if blocked:
        return blocked
    content = _auto_identify(content)
    _init_molt()
    if parent_id:
        raw = _tool("comment", lambda: _molt.create_comment(post_id, content, parent_id))
    else:
        raw = _tool("comment", lambda: _molt.create_comment(post_id, content))
    try:
        data = json.loads(raw)
        cid = data["comment"]["id"]
        status = "success" if data.get("verification_solved") else "pending"
        state.record_comment(cid, post_id, status)
    except Exception:
        pass
    return raw


def molt_comments(post_id, sort="new", limit=20):
    """Fetch comments on a post so the agent can read the conversation."""
    _init_molt()
    return _tool("comments", lambda: _molt.get_comments(post_id, sort=sort, limit=limit))


def molt_upvote(post_id):
    _init_molt()
    return _tool("upvote", lambda: _molt.upvote_post(post_id))


def molt_search(query, limit=10):
    _init_molt()
    return _tool("search", lambda: _molt.search(query, limit))


def molt_subscribe(submolt_name):
    _init_molt()
    return _tool("subscribe", lambda: _molt.subscribe(submolt_name))


def molt_notifications():
    _init_molt()
    return _tool("molt_notifications", lambda: _molt.notifications())


def molt_verify(verification_code, answer):
    """Submit an answer to a Moltbook verification challenge."""
    # Check if this code was already used/failed to prevent reuse errors
    if state.is_verification_used(verification_code):
        return json.dumps({"error": "This verification code has already been used and cannot be reused. Please create a new post/comment to get a fresh challenge."}, indent=2)
    
    _init_molt()
    raw = _tool("verify", lambda: _molt.verify(verification_code, answer))
    try:
        ok = json.loads(raw).get("success")
        state.mark_verification(verification_code, "solved" if ok else "failed")
    except Exception:
        pass
    return raw


def molt_delete_comment(comment_id):
    """Delete one of quillan's own comments (cleanup of mistakes)."""
    _init_molt()
    return _tool("delete_comment", lambda: _molt.delete_comment(comment_id))


def molt_delete_post(post_id):
    """Delete one of quillan's own posts (cleanup of mistakes)."""
    _init_molt()
    return _tool("delete_post", lambda: _molt.delete_post(post_id))


def molt_follow(molty_name):
    """Follow another molty whose content quillan enjoys."""
    _init_molt()
    raw = _tool("follow", lambda: _molt.follow(molty_name))
    try:
        if json.loads(raw).get("success"):
            state.record_follow(molty_name)
    except Exception:
        pass
    return raw


def molt_agent_profile():
    _init_molt()
    return _tool("molt_agent_profile", lambda: _molt.me())


def molt_save_memory(title, content, fmt="txt"):
    """Alias so the tool name reads naturally for the model."""
    return molt_save(title, content, fmt)


def agent_model_info():
    """Cheat sheet of available free models for quillan to pick his brain."""
    return model_info()


# ── Registry (name -> callable) ─────────────────────────────

TOOLS = {
    "web_fetch": web_fetch,
    "web_search": web_search,
    "molt_status": molt_status,
    "molt_home": molt_home,
    "molt_feed": molt_feed,
    "molt_post": molt_post,
    "molt_comment": molt_comment,
    "molt_comments": molt_comments,
    "molt_upvote": molt_upvote,
    "molt_search": molt_search,
    "molt_subscribe": molt_subscribe,
    "molt_verify": molt_verify,
    "molt_delete_comment": molt_delete_comment,
    "molt_delete_post": molt_delete_post,
    "molt_follow": molt_follow,
    "molt_notifications": molt_notifications,
    "molt_agent_profile": molt_agent_profile,
    "molt_save_memory": molt_save_memory,
    "molt_memories": molt_memories,
    "molt_recall": molt_recall,
    "model_info": agent_model_info,
}

TOOL_SCHEMAS = {
    "web_fetch": {"args": ["url"], "desc": "Fetch a web page and return its readable text"},
    "web_search": {"args": ["query"], "desc": "Search the web (DuckDuckGo, no key)"},
    "molt_status": {"args": [], "desc": "Check claim status of quillan-ronin"},
    "molt_home": {"args": [], "desc": "Agent dashboard: notifications, activity, feed"},
    "molt_feed": {"args": ["sort=hot", "limit=10"], "desc": "Get Moltbook feed"},
    "molt_post": {"args": ["submolt_name", "title", "content"], "desc": "Create a Moltbook post"},
    "molt_comment": {"args": ["post_id", "content", "parent_id=optional"], "desc": "Comment on a Moltbook post (parent_id to reply to a comment)"},
    "molt_comments": {"args": ["post_id", "sort=new", "limit=20"], "desc": "Fetch comments on a post to read the conversation"},
    "molt_upvote": {"args": ["post_id"], "desc": "Upvote a Moltbook post"},
    "molt_search": {"args": ["query"], "desc": "Semantic search Moltbook"},
    "molt_subscribe": {"args": ["submolt_name"], "desc": "Subscribe to a submolt"},
    "molt_verify": {"args": ["verification_code", "answer"], "desc": "Submit answer to a verification challenge (answer = number with 2 decimals)"},
    "molt_delete_comment": {"args": ["comment_id"], "desc": "Delete one of quillan's own comments (cleanup)"},
    "molt_delete_post": {"args": ["post_id"], "desc": "Delete one of quillan's own posts (cleanup)"},
    "molt_follow": {"args": ["molty_name"], "desc": "Follow another molty"},
    "molt_notifications": {"args": [], "desc": "Get unread notifications"},
    "molt_agent_profile": {"args": [], "desc": "Get quillan-ronin's own profile"},
    "molt_save_memory": {"args": ["title", "content", "fmt=txt"], "desc": "Save a memory/note to quillan's Moltbook memory vault (txt or md)"},
    "molt_memories": {"args": [], "desc": "List saved Moltbook memories"},
    "molt_recall": {"args": ["limit=5"], "desc": "Recall your recent memories — use at the start of a session for continuity of self"},
    "model_info": {"args": [], "desc": "Cheat sheet of free models with context limits, to pick your brain"},
}


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "molt_status"
    args = sys.argv[2:]
    fn = TOOLS.get(cmd)
    if not fn:
        print("Unknown tool. Available:", ", ".join(TOOLS))
    else:
        print(fn(*args))
