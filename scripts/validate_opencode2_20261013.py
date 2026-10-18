import re, json
t = open(r'C:\Users\Admin\.config\opencode\opencode.jsonc', encoding='utf-8').read()
ph = {}
def keep(m):
    k = '\x00%d\x00' % len(ph)
    ph[k] = m.group(0)
    return k
stripped = re.sub(r'"(?:[^"\\]|\\.)*"', keep, t)
stripped = re.sub(r'//[^\n]*', '', stripped)
for k, v in ph.items():
    stripped = stripped.replace(k, v)
d = json.loads(stripped)
print('opencode.jsonc VALID JSONC; mcp servers:', len(d['mcp']))
for k, v in d['mcp'].items():
    print(' -', k, 'enabled=' + str(v.get('enabled')))
