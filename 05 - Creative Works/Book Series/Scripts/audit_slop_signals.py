import re
import os
import statistics

BASE = r'C:\02_QUILLAN\Book Series'
BOOKS = [
    'Book 1 - Twisted Destiny.md',
    'Book 2 - Rise of Ascension.md',
    'Book 3 - Battle Grandeur.md',
    'Book 4 - Fall of Empires.md',
    'Book 5 - Shadows That Speak.md',
]

SIGNALS = {
    'filter: could feel': r'\bcould (?:feel|sense)\b',
    'filter: could see/hear': r'\bcould (?:see|hear)\b',
    'filter: noticed/watched': r'\b(noticed|watched)\b',
    'filter: felt/saw/heard bare': r'\b(fe lt|felt|saw|heard|realized)\b',
    'hedge: seemed to': r'\bseemed to\b',
    'hedge: almost as if/as though': r'\b(almost as if|as though|as if)\b',
    'filler: began/started to': r'\b(?:began|started) to\b',
    'choreo: reached out': r'\breached out\b',
    'emotion wave: wave/surge/flood of': r'\b(?:wave|surge|flood|tidal wave|crash) of (?:grief|fear|anger|pain|joy|dread|panic|relief|guilt|shame|heat|cold|nausea|vertigo|exhaustion)',
    'washed over him/her': r'\bwashed over (?:him|her|them|me)\b',
    'adverb tag: -ly said/whispered': r'\b(?:said|whispered|asked|murmured|replied|shouted) \w+ly\b',
    'triple adjectives': r'\b\w+,\s+\w+,\s+and \w+ \w+(?:ing|ed)?\b',
}

print(f'{"signal":34}' + ''.join(f'{b[5]:>6}' for b in BOOKS))
for name, pat in SIGNALS.items():
    row = f'{name:34}'
    for fname in BOOKS:
        text = open(os.path.join(BASE, fname), encoding='utf-8-sig').read()
        n = len(re.findall(pat, text, re.IGNORECASE))
        row += f'{n:>6}'
    print(row)

# sentence-length variance per book (rhythm flatness) + flattest chapters
print('\nRHYTHM: mean sentence len / stdev (lower stdev = flatter = more robotic)')
for fname in BOOKS:
    text = open(os.path.join(BASE, fname), encoding='utf-8-sig').read()
    body = '\n'.join(p for p in text.split('\n\n') if not p.strip().startswith(('#', '- ', '|')))
    sents = [len(s.split()) for s in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', body)) if len(s.split()) >= 2]
    print(f'  {fname[:32]:34} mean:{statistics.mean(sents):5.1f}  stdev:{statistics.stdev(sents):5.1f}')

# flattest chapters overall (stdev below threshold)
print('\nFLATTEST CHAPTERS (stdev < 6.0):')
flat_total = 0
for fname in BOOKS:
    text = open(os.path.join(BASE, fname), encoding='utf-8-sig').read()
    parts = re.split(r'(?m)(^#{1,3}[ \t]*(?:Chapter|Epilogue|Coda)[^\n]*)$', text)
    for i in range(1, len(parts), 2):
        title = parts[i].strip('# ').strip()[:38]
        body = parts[i + 1]
        sents = [len(s.split()) for s in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', body)) if len(s.split()) >= 2]
        if len(sents) > 50:
            sd = statistics.stdev(sents)
            if sd < 6.0:
                flat_total += 1
                print(f'  {fname[:6]}  {title:40} stdev:{sd:.2f}')
print(f'  total flat chapters: {flat_total}')
