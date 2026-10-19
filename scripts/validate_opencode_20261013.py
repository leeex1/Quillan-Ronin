import re, json
t = open(r'C:\Users\Admin\.config\opencode\opencode.jsonc', encoding='utf-8').read()
# blank string literals, then strip // comments
stripped = re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: m.group(0).replace('\n', '\\n'), t)
stripped = re.sub(r'//[^\n]*', '', stripped)
d = json.loads(stripped)
print('opencode.jsonc VALID JSONC; mcp servers:', len(d['mcp']))
for k, v in d['mcp'].items():
    print(' -', k, 'enabled=' + str(v.get('enabled')))
