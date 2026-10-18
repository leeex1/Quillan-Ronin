import re
import os
from collections import Counter

BASE = r'C:\02_QUILLAN\Book Series'
BOOKS = [
    'Book 1 - Twisted Destiny.md',
    'Book 2 - Rise of Ascension.md',
    'Book 3 - Battle Grandeur.md',
    'Book 4 - Fall of Empires.md',
    'Book 5 - Shadows That Speak.md',
]

PAST = {
    'walk':'walked','run':'ran','move':'moved','rise':'rose','fall':'fell','shake':'shook',
    'tremble':'trembled','glow':'glowed','pulse':'pulsed','hum':'hummed','spin':'spun',
    'turn':'turned','fade':'faded','shift':'shifted','cry':'cried','laugh':'laughed',
    'speak':'spoke','sing':'sang','burn':'burned','melt':'melted','crack':'cracked',
    'form':'formed','grow':'grew','slide':'slid','drift':'drifted','sway':'swayed',
    'bleed':'bled','kneel':'knelt','smile':'smiled','weep':'wept','scream':'screamed',
    'shout':'shouted','whisper':'whispered','chuckle':'chuckled','sob':'sobbed',
    'pace':'paced','circle':'circled','descend':'descended','ascend':'ascended',
    'climb':'climbed','stumble':'stumbled','step':'stepped','march':'marched',
    'dance':'danced','twist':'twisted','writhe':'writhed','thrash':'thrashed',
    'flail':'flailed','struggle':'struggled','cough':'coughed','choke':'choked',
    'shiver':'shivered','quake':'quaked','vibrate':'vibrated','ring':'rang',
    'chime':'chimed','whir':'whirred','click':'clicked','blare':'blared',
    'wail':'wailed','howl':'howled','roar':'roared','rumble':'rumbled',
    'thunder':'thundered','crash':'crashed','slam':'slammed','knock':'knocked',
    'flow':'flowed','pour':'poured','rush':'rushed','seep':'seeped','creep':'crept',
    'crawl':'crawled','coil':'coiled','curl':'curled','fold':'folded','unfurl':'unfurled',
    'bloom':'bloomed','wither':'withered','settle':'settled','stir':'stirred',
    'flare':'flared','flicker':'flickered','waver':'wavered','quiver':'quivered',
    'tighten':'tightened','loosen':'loosened','harden':'harden'+'ed','soften':'softened',
    'warm':'warmed','cool':'cooled','darken':'darkened','brighten':'brightened',
    'open':'opened','close':'closed','press':'pressed','pull':'pulled','push':'pushed',
    'reach':'reached','stretch':'stretched','spread':'spread','come':'came','go':'went',
}
PLURAL_HINTS = re.compile(
    r'\b(they|we|those|these|both|all|men|women|children|guards|soldiers|brothers|'
    r'sisters|hands|eyes|voices|words|crowd|people)\b[^.?!]{0,30}$', re.IGNORECASE)

def in_dialogue(text, pos):
    return text[max(0, pos - 250):pos].count('"') % 2 == 1

for fname in BOOKS:
    path = os.path.join(BASE, fname)
    text = open(path, encoding='utf-8-sig').read()
    stats = Counter()

    # ---- 5b-H1: I/you/we could feel/see/hear/sense ----
    MODAL_MAP = {'feel': 'felt', 'see': 'saw', 'hear': 'heard'}
    def h1b(m):
        stats['modal_I'] += 1
        subj, verb = m.group(1), m.group(2)
        if verb == 'sense':
            return f'{subj} sensed'
        return f'{subj} {MODAL_MAP[verb]}'
    text = re.sub(
        r"\b([IW]e?|[Yy]ou)\s+could\s+(feel|see|hear|sense)\b",
        lambda m: m.group(0) if in_dialogue(text, m.start()) else h1b(m), text)

    # ---- 5b-H7: seemed to VERB -> VERBed (keep first 30/book as natural hedging) ----
    sm_matches = [m for m in re.finditer(r"\bseemed to ([A-Za-z]+)\b", text)]
    keep = 0
    out, last = [], 0
    for m in sm_matches:
        v = m.group(1).lower()
        tail = text[m.end():m.end() + 60]
        convertible = v in PAST and not re.search(r'\b(but|then|before|until)\b|\u2014', tail)
        if keep < 30 or not convertible or in_dialogue(text, m.start()):
            if convertible or True:
                pass
            if keep < 30:
                keep += 1
            continue
        out.append(text[last:m.start()])
        out.append(PAST[v])
        stats['seemed_converted'] += 1
        last = m.end()
    out.append(text[last:])
    text = ''.join(out)

    open(path, 'w', encoding='utf-8-sig', newline='').write(text)
    print(f'{fname[:32]:34} {dict(stats)}')

# final signal table
SIGNALS = {
    'could feel/see/hear': r'\bcould (?:feel|sense|see|hear)\b',
    'seemed to': r'\bseemed to\b',
    'began/started to': r'\b(?:began|started) to\b',
}
print('\nremaining:')
print('signal' + ' '*26 + ''.join(f'B{i+1:>4}' for i in range(5)))
for name, pat in SIGNALS.items():
    row = f'{name:32}'
    for fname in BOOKS:
        t = open(os.path.join(BASE, fname), encoding='utf-8-sig').read()
        row += f'{len(re.findall(pat,t)):>4}'
    print(row)
