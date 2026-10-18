import re
import os
from collections import Counter, defaultdict

BASE = r'C:\02_QUILLAN\Book Series'
BACKUP = os.path.join(BASE, 'BACKUP_20260821_PRE_PASS5')
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
    'bloom':'bloomed','wither':'withered','settle':'settled','shift':'shifted',
    'stir':'stirred','flare':'flared','flicker':'flickered','waver':'wavered',
    'quiver':'quivered','tighten':'tightened','loosen':'loosened','harden':'hardened',
    'soften':'softened','warm':'warmed','cool':'cooled','darken':'darkened',
    'brighten':'brightened','still':'stilled','open':'opened','close':'closed',
}
PLURAL_HINTS = re.compile(
    r'\b(they|we|those|these|both|all|men|women|children|guards|soldiers|brothers|'
    r'sisters|hands|eyes|voices|words|crowd|people)\b[^.?!]{0,30}$', re.IGNORECASE)

def in_dialogue(text, pos):
    return text[max(0, pos - 250):pos].count('"') % 2 == 1

def process(fname):
    path = os.path.join(BASE, fname)
    text = open(path, encoding='utf-8-sig').read()
    stats = Counter()
    samples = defaultdict(list)

    # ---- H1: modal filter drops ----
    MODAL_MAP = {'feel': 'felt', 'see': 'saw', 'hear': 'heard'}
    def h1(m):
        stats['H1_modal'] += 1
        if len(samples['H1']) < 4:
            samples['H1'].append(m.group(0))
        subj, verb = m.group(1), m.group(3)
        if verb == 'sense':
            return f'{subj} sensed'
        return f'{subj} {MODAL_MAP[verb]}'
    text = re.sub(r"\b([A-Z][a-z]+|[Hh]e|[Ss]he|[Tt]hey|[Ii]t)\s+(could)\s+(feel|see|hear|sense)\b",
                  lambda m: h1(m) if not in_dialogue(text, m.start()) else m.group(0), text)

    # ---- H2: began/started to VERB -> VERBed (with interruption guard) ----
    def h2(m):
        tail = text[m.end():m.end() + 80]
        v = m.group('v').lower()
        if v not in PAST or re.search(r'\b(but|then|before|until)\b|\u2014', tail):
            return m.group(0)
        stats['H2_began'] += 1
        if len(samples['H2']) < 4:
            samples['H2'].append(m.group(0))
        return PAST[v]
    text = re.sub(r"\b(?:began|started)\s+to\s+(?P<v>[A-Za-z]+)\b",
                  lambda m: m.group(0) if in_dialogue(text, m.start()) else h2(m), text)

    # ---- H3: He/She/They realized that X -> X ----
    def h3(m):
        rest = text[m.end():]
        first = re.match(r'[A-Za-z]', rest)
        repl = ''
        if first:
            ch = rest[0]
            repl += ch.upper() + rest[1:3]
            return repl.rstrip('')
        return repl
    # simpler deterministic approach below instead of clever closure
    def h3_sub(m):
        follow = text[m.end():m.end() + 1]
        stats['H3_realized'] += 1
        if len(samples['H3']) < 4:
            s = max(0, m.start() - 60)
            samples['H3'].append(re.sub(r'\s+', ' ', text[s:m.end() + 70]))
        if follow.isalpha():
            return follow.upper()
        return follow
    text = re.sub(r"(?:(?<=[.!?]\s)|(?<=\n\n))[HST][heiutm'y ]{1,7}\s(?:had\s)?realized\s+that\s",
                  h3_sub, text)

    # ---- H4: seemed to be -> was/were ----
    def h4(m):
        prev = text[max(0, m.start() - 40):m.start()]
        if re.search(r'\bthere\b[^.?!]{0,12}$', prev, re.IGNORECASE):
            return m.group(0)
        be = 'were' if PLURAL_HINTS.search(prev) else 'was'
        stats['H4_seemedbe'] += 1
        return be
    text = re.sub(r"\bseemed to be\b", h4, text)

    # ---- H5: triple adjectives -> drop middle (cap 25/book) ----
    tri_rx = re.compile(r"\b([a-z]+), ([a-z]+), and ([a-z]+) ([a-z]{3,})\b")
    tri_matches = [m for m in tri_rx.finditer(text)]
    over = max(0, len(tri_matches) - 25)
    drop_idx = set()
    if over:
        step = max(1, len(tri_matches) // over)
        i = len(tri_matches) - 1
        dropped = 0
        while dropped < over and i >= 0:
            if not in_dialogue(text, tri_matches[i].start()):
                drop_idx.add(i)
                dropped += 1
            i -= step
    out, last = [], 0
    for i, m in enumerate(tri_matches):
        if i in drop_idx:
            out.append(text[last:m.start()])
            out.append(f'{m.group(1)} and {m.group(3)} {m.group(4)}')
            stats['H5_triadj'] += 1
            if len(samples['H5']) < 4:
                samples['H5'].append(f'{m.group(0)} -> {m.group(1)} and {m.group(3)} {m.group(4)}')
            last = m.end()
    out.append(text[last:])
    text = ''.join(out)

    # ---- H6: adverb dialogue tags (cap 14/book) ----
    tag_rx = re.compile(r"\b(said|asked|whispered|murmured|replied|shouted|growled|hissed) (\w+)ly\b")
    tags = [m for m in tag_rx.finditer(text)]
    kept = 0
    out, last = [], 0
    for m in tags:
        if kept < 14 or in_dialogue(text, m.start()) is False and kept < 14:
            kept += 1
            continue
        out.append(text[last:m.start()])
        out.append(m.group(1))
        stats['H6_advtags'] += 1
        last = m.end()
    out.append(text[last:])
    text = ''.join(out)

    open(path, 'w', encoding='utf-8-sig', newline='').write(text)
    return dict(stats), dict(samples)

if __name__ == '__main__':
    os.makedirs(BACKUP, exist_ok=True)
    import shutil
    for fname in BOOKS:
        shutil.copy2(os.path.join(BASE, fname), os.path.join(BACKUP, fname))
        print(f'== {fname}')
        stats, samples = process(fname)
        for k, v in sorted(stats.items()):
            print(f'   {k}: {v}')
        for k, exs in samples.items():
            for e in exs[:2]:
                print(f'      e.g. [{k}] {e}')
        print()
