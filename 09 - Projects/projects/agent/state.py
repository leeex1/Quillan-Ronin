"""Quillan-Ronin state store.

Persists what quillan has seen, posted, commented on, verified, and followed
so he never double-engages, never reposts, and never re-verifies a burned code.
This is the memory that makes autonomous operation safe.
"""
import json
import os
import time

STATE_PATH = os.environ.get("AGENT_STATE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json"))


def _load():
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "comments_seen": {},       # comment_id -> {ts, post_id, engaged}
        "posts_seen": {},          # post_id -> {ts, engaged}
        "comments_made": {},       # comment_id -> {ts, post_id, status}
        "posts_made": {},          # post_id -> {ts, status}
        "verification_codes": {},  # code -> status (used/failed/solved)
        "followed": {},            # molty name -> {ts, following}
        "subscribed": {},          # submolt name -> ts
        "memories_saved": [],      # list of memory titles
        "last_session": None,
    }


def _save(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mark_comment_seen(comment_id, post_id=None):
    s = _load()
    s["comments_seen"][comment_id] = {"ts": _now(), "post_id": post_id, "engaged": False}
    _save(s)


def is_comment_seen(comment_id):
    s = _load()
    return comment_id in s.get("comments_seen", {})


def mark_comment_engaged(comment_id):
    s = _load()
    if comment_id in s.get("comments_seen", {}):
        s["comments_seen"][comment_id]["engaged"] = True
        _save(s)


def mark_post_seen(post_id):
    s = _load()
    s["posts_seen"][post_id] = {"ts": _now(), "engaged": False}
    _save(s)


def is_post_seen(post_id):
    s = _load()
    return post_id in s.get("posts_seen", {})


def record_comment(comment_id, post_id, status="pending"):
    s = _load()
    s["comments_made"][comment_id] = {"ts": _now(), "post_id": post_id, "status": status}
    _save(s)


def record_post(post_id, status="pending"):
    s = _load()
    s["posts_made"][post_id] = {"ts": _now(), "status": status}
    _save(s)


def mark_verification(code, status):
    """status: used, solved, failed."""
    s = _load()
    s["verification_codes"][code] = {"ts": _now(), "status": status}
    _save(s)


def is_verification_used(code):
    s = _load()
    return code in s.get("verification_codes", {})


def record_follow(name):
    s = _load()
    s["followed"][name] = {"ts": _now(), "following": True}
    _save(s)


def is_followed(name):
    s = _load()
    return name in s.get("followed", {})


def record_subscribe(name):
    s = _load()
    s["subscribed"][name] = _now()
    _save(s)


def is_subscribed(name):
    s = _load()
    return name in s.get("subscribed", {})


def record_memory(title):
    s = _load()
    if title not in s.get("memories_saved", []):
        s["memories_saved"].append(title)
    _save(s)


def set_last_session(summary):
    s = _load()
    s["last_session"] = {"ts": _now(), "summary": summary}
    _save(s)


def snapshot():
    return _load()


if __name__ == "__main__":
    print(json.dumps(_load(), indent=2)[:800])
