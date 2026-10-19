import urllib.request, json, re
md = open(r"C:\02_QUILLAN\09 - Projects\projects\opencode quillan.md",
          encoding="utf-8").read()
tok = re.search(r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}",
                md).group(0)
req = urllib.request.Request(
    "https://discord.com/api/v10/channels/1438282228898467951/messages?limit=20",
    headers={"Authorization": "Bot " + tok, "User-Agent": "quillan-relay"})
msgs = json.load(urllib.request.urlopen(req, timeout=15))
seen = {}
for m in msgs:
    seen.setdefault(m["author"]["id"], m["author"]["username"])
for uid, uname in seen.items():
    print(f"{uname} -> {uid}")
