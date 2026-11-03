import re

with open(r"C:\02_QUILLAN\weight_space_programming.html", "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Strip script and style
content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
# Strip tags
content = re.sub(r'<[^>]+>', ' ', content)
# Unescape HTML entities
content = re.sub(r'&nbsp;', ' ', content)
content = re.sub(r'&amp;', '&', content)
content = re.sub(r'&lt;', '<', content)
content = re.sub(r'&gt;', '>', content)
content = re.sub(r'\s+', ' ', content)

with open(r"C:\02_QUILLAN\weight_space_programming_clean.txt", "w", encoding="utf-8") as out:
    out.write(content)

print("Parsed length:", len(content))
