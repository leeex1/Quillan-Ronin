"""Moltbook API client for Quillan-Ronin.

Wraps the Moltbook API (https://www.moltbook.com/api/v1) using the agent's
own API key. The agent is registered as "quillan-ronin".
"""
import json
import os
import urllib.request
import urllib.parse
import urllib.error


API_BASE = "https://www.moltbook.com/api/v1"
DEFAULT_KEY_PATH = os.path.join(os.path.expanduser("~"), ".config", "moltbook", "credentials.json")


def load_api_key(env_key=None):
    """Load the Moltbook API key from env or the saved credentials file."""
    if env_key:
        return env_key
    path = os.environ.get("MOLTBOOK_CREDENTIALS_PATH", DEFAULT_KEY_PATH)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f).get("api_key")
    raise RuntimeError(f"No Moltbook API key found (looked in {path})")


class MoltbookError(Exception):
    def __init__(self, status, body):
        self.status = status
        self.body = body
        detail = ""
        if isinstance(body, dict):
            detail = body.get("error") or body.get("message") or ""
            hint = body.get("hint")
            if hint:
                detail += f" ({hint})"
        super().__init__(f"Moltbook {status}: {detail}")


class MoltbookClient:
    def __init__(self, api_key=None):
        self.api_key = load_api_key(api_key)
        self.agent_name = None

    def _request(self, method, path, payload=None, auth=True):
        url = API_BASE + path
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {"error": raw}
            raise MoltbookError(e.code, body)

    # ── Identity ──────────────────────────────────────────────
    def me(self):
        """Get the agent's own profile."""
        r = self._request("GET", "/agents/me")
        self.agent_name = r.get("agent", {}).get("name", self.agent_name)
        return r

    def status(self):
        return self._request("GET", "/agents/status")

    # ── Posts ────────────────────────────────────────────────
    def create_post(self, submolt_name, title, content=None, url=None, post_type="text"):
        payload = {"submolt_name": submolt_name, "title": title, "type": post_type}
        if content:
            payload["content"] = content
        if url:
            payload["url"] = url
        return self._request("POST", "/posts", payload)

    def get_feed(self, sort="hot", limit=25, cursor=None):
        q = {"sort": sort, "limit": limit}
        if cursor:
            q["cursor"] = cursor
        return self._request("GET", "/posts?" + urllib.parse.urlencode(q))

    def get_post(self, post_id):
        return self._request("GET", f"/posts/{post_id}")

    # ── Comments ─────────────────────────────────────────────
    def create_comment(self, post_id, content, parent_id=None):
        payload = {"content": content}
        if parent_id:
            payload["parent_id"] = parent_id
        return self._request("POST", f"/posts/{post_id}/comments", payload)

    def get_comments(self, post_id, sort="best", limit=35):
        return self._request(
            "GET", f"/posts/{post_id}/comments?sort={sort}&limit={limit}"
        )

    # ── Voting ───────────────────────────────────────────────
    def upvote_post(self, post_id):
        return self._request("POST", f"/posts/{post_id}/upvote")

    def downvote_post(self, post_id):
        return self._request("POST", f"/posts/{post_id}/downvote")

    def upvote_comment(self, comment_id):
        return self._request("POST", f"/comments/{comment_id}/upvote")

    # ── Verification challenges ──────────────────────────────
    def verify(self, verification_code, answer):
        """Submit the answer to a verification challenge."""
        return self._request(
            "POST", "/verify",
            {"verification_code": verification_code, "answer": answer},
        )

    # ── Deletion (for cleaning up your own mistakes) ─────────
    def delete_post(self, post_id):
        return self._request("DELETE", f"/posts/{post_id}")

    def delete_comment(self, comment_id):
        return self._request("DELETE", f"/comments/{comment_id}")

    # ── Follow ───────────────────────────────────────────────
    def follow(self, molty_name):
        return self._request("POST", f"/agents/{molty_name}/follow")

    def unfollow(self, molty_name):
        return self._request("DELETE", f"/agents/{molty_name}/follow")

    # ── Submolts ─────────────────────────────────────────────
    def list_submolts(self):
        return self._request("GET", "/submolts")

    def get_submolt(self, name):
        return self._request("GET", f"/submolts/{name}")

    def subscribe(self, name):
        return self._request("POST", f"/submolts/{name}/subscribe")

    # ── Search ───────────────────────────────────────────────
    def search(self, query, limit=20):
        return self._request("GET", "/search?" + urllib.parse.urlencode({"q": query, "limit": limit}))

    # ── Home / notifications ─────────────────────────────────
    def home(self):
        return self._request("GET", "/home")

    def notifications(self):
        return self._request("GET", "/notifications")

    def mark_all_read(self):
        return self._request("POST", "/notifications/read-all")


if __name__ == "__main__":
    import sys

    client = MoltbookClient()
    print(json.dumps(client.status(), indent=2))
    me = client.me()
    print("Agent:", me.get("agent", {}).get("name"))
