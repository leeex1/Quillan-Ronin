import urllib.request, urllib.error, json, re, time
md = open(r"C:\02_QUILLAN\09 - Projects\projects\opencode quillan.md",
          encoding="utf-8").read()
tok = re.search(r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}",
                md).group(0)
ME = "1549554201649090630"
CH = "1438282228898467951"
META = re.compile(r"^\[(SYS|Q-OPENCODE)\]\s*(handshake|ack|got it|seen|relay-openCode)", re.I)


def api(method, path, data=None):
    req = urllib.request.Request(
        f"https://discord.com/api/v10{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Authorization": "Bot " + tok, "User-Agent": "quillan-relay",
                 "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r) if r.length != 0 else None


def raw_delete(msg_id):
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{CH}/messages/{msg_id}",
        headers={"Authorization": "Bot " + tok, "User-Agent": "quillan-relay"},
        method="DELETE")
    with urllib.request.urlopen(req, timeout=15):
        pass


msgs = api("GET", f"/channels/{CH}/messages?limit=50")
mine_meta = [m for m in msgs
             if m["author"]["id"] == ME and META.search(m.get("content", ""))]
print(f"my meta-msgs found: {len(mine_meta)}")
for m in mine_meta:
    try:
        raw_delete(m["id"])
        print(f"deleted {m['id']} ({m.get('content','')[:60]!r})")
        time.sleep(1)
    except urllib.error.HTTPError as e:
        print(f"keep {m['id']}: http {e.code}")
print("cleanup done")
