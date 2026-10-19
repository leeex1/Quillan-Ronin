"""Moltbook verification challenge solver.

New agents must solve an obfuscated math word problem before content
publishes. The challenge scatters symbols, jumbles case, and inserts noise
letters, but the true letters appear in order.

Strategy:
  1. De-duplicate consecutive letters -> reveals the words ("ThIrTy TwO" ->
     "thirtytwo"). Number words become contiguous substrings.
  2. Find number words as substrings, prefer longest match (fourten = 14 not
     four + ten), merge tens+unit compounds (thirty+two = 32).
  3. Determine the operation from keywords.
  4. Select the two operative numbers near "force/exert/lose/newtons".
  5. Compute. Return None if not confident (never burn a one-time code).
"""
import json
import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_TEENS = {
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

# After de-dup cleaning, teen words lose their double-e: fourteen -> fourten,
# sixteen -> sixten, three -> thre, etc.
_DEDUP_ALIASES = {
    "thirten": 13, "fourten": 14, "fiften": 15, "sixten": 16,
    "seventen": 17, "eighten": 18, "nineten": 19,
    "thre": 3, "eleven": 11, "twelve": 12, "twelven": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

_NUMBERS = {}
_NUMBERS.update(_UNITS)
_NUMBERS.update(_TEENS)
_NUMBERS.update(_TENS)
_NUMBERS.update(_DEDUP_ALIASES)

# Small noise values that appear inside larger words (antenna->ten, lobster->one)
_NOISE_VALUES = {1, 4, 6, 10}


def _clean(text):
    """Lowercase, drop non-letters, remove consecutive duplicate letters.

    This reveals the true words in order. Example:
      'ThIrTy TwO' -> 'thirtytwo'
    """
    t = "".join(ch for ch in text.lower() if ch.isalpha())
    out = []
    prev = ""
    for ch in t:
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def _find_spans(clean):
    """Find (start, end, value) for each number word as a substring."""
    spans = []
    # longest-first so fourteen beats four/ten
    for w, v in sorted(_NUMBERS.items(), key=lambda kv: -len(kv[0])):
        pos = 0
        while True:
            pos = clean.find(w, pos)
            if pos == -1:
                break
            spans.append((pos, pos + len(w), v, w))
            pos += 1
    return spans


def _dedupe_spans(spans):
    """Remove spans fully contained in a longer span (e.g. 'four' in 'fourten')."""
    spans.sort(key=lambda s: (s[1] - s[0]), reverse=True)
    kept = []
    for s in spans:
        contained = any(
            k[0] <= s[0] and s[1] <= k[1] and (k[1] - k[0]) > (s[1] - s[0])
            for k in kept
        )
        if not contained:
            kept.append(s)
    return sorted(kept, key=lambda s: s[0])


def _filter_noise(clean, spans):
    """Drop small noise values (1,4,6,10) not near 'newtons' or a compound."""
    filtered = []
    for s in spans:
        start, end, val, word = s
        if val in _NOISE_VALUES and len(word) <= 4:
            left = clean[max(0, start - 15):start]
            right = clean[end:end + 15]
            near_newtons = any(k in (left + right) for k in ("newton", "noton", "ooton", "tOnS"))
            near_compound = any(
                o is not s and (abs(o[0] - end) <= 2 or abs(start - o[1]) <= 2)
                for o in spans
            )
            if not (near_newtons or near_compound):
                continue
        filtered.append(s)
    return filtered


def _merge_compounds(spans):
    """Merge adjacent tens+unit into a compound (thirty+two -> 32).

    Handles both normal units (two, five) and dedup aliases (thre = three,
    twelven = twelve) that appear after dedupe-cleaning.
    Also handles hyphenated compounds like "thirty-two" directly.
    """
    spans = sorted(spans, key=lambda s: s[0])
    merged = []
    i = 0
    while i < len(spans):
        s = spans[i]
        # Check for hyphenated compound (thirty-two, forty-five, etc.)
        if s[3] in _TENS and i + 1 < len(spans):
            nxt = spans[i + 1]
            # Allow overlapping or adjacent (hyphen becomes adjacent after cleaning)
            if nxt[0] <= s[1] + 2:  # Slightly larger gap for hyphen
                unit_val = _UNITS.get(nxt[3]) or _DEDUP_ALIASES.get(nxt[3])
                if unit_val is not None and unit_val < 10:
                    merged.append((s[0], nxt[1], _TENS[s[3]] + unit_val, s[3] + nxt[3]))
                    i += 2
                    continue
        merged.append(s)
        i += 1
    return merged


def _extract_op(clean):
    # Check for multiplication first (most specific)
    if re.search(r"times|multiplied|product|doubl|tripl", clean):
        return "*"
    # Check for division
    if re.search(r"divid|half of|quarter|split", clean):
        return "/"
    # Check for subtraction (more specific patterns)
    if re.search(r"slow|minus|subtract|less|fewer|take away|remain|left over|decrease|drop|reduce|lower|lose|reduces|molting|but", clean):
        return "-"
    # Then check for addition (most common, check last)
    if re.search(r"total|combined|altogether|sum|plus|add|both|more|gain|increase|accelerate|overall|together|faster|intake|increases|and", clean):
        return "+"
    # Default to addition if no clear operation
    return "+"


def _numbers_near_keywords(clean, spans, keywords):
    """Keep spans touching a keyword region (force/exert/lose/newtons/etc)."""
    hits = []
    for kw in keywords:
        for m in re.finditer(kw, clean):
            hits.append((m.start(), m.end()))
    if not hits:
        return spans
    selected = []
    for s in spans:
        start, end, val, word = s
        near = any((start - 12) <= h[1] and end >= (h[0] - 4) for h in hits)
        if near:
            selected.append(s)
    return selected or spans


def solve_challenge(challenge_text):
    """Solve a Moltbook math challenge. Returns answer string or None."""
    if not challenge_text:
        return None
    clean = _clean(challenge_text)
    spans = _find_spans(clean)
    if not spans:
        return None
    spans = _dedupe_spans(spans)
    spans = _filter_noise(clean, spans)
    spans = _merge_compounds(spans)

    op = _extract_op(clean)
    if op is None:
        return None

    # For velocity/speed problems, use all numbers
    if re.search(r"swim|speed|velocity|fast|slow", clean):
        ctx = spans
    elif op == "+":
        ctx = _numbers_near_keywords(clean, spans, [r"force", r"exert", r"add", r"gain", r"total", r"push", r"newton", r"new", r"increase"])
    elif op == "-":
        ctx = _numbers_near_keywords(clean, spans, [r"force", r"exert", r"lose", r"remain", r"slow", r"reduc", r"molting", r"newton", r"new", r"decrease"])
    else:
        ctx = spans

    values = []
    seen = set()
    for s in ctx:
        if s[2] not in seen:
            seen.add(s[2])
            values.append(s[2])

    if len(values) < 2:
        return None
    if len(values) > 2:
        # For velocity problems, use the first two meaningful numbers
        if re.search(r"swim|speed|velocity", clean):
            values = values[:2]
        else:
            values = sorted(values)[-2:]

    a, b = values[0], values[1]
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    elif op == "*":
        ans = a * b
    elif op == "/":
        ans = a / b if b else None
    else:
        return None
    if ans is None:
        return None
    return f"{ans:.2f}"


def clean_text_for_model(challenge_text):
    """Return the de-dup cleaned text for the model to read."""
    return _clean(challenge_text)


if __name__ == "__main__":
    tests = [
        ("A] lO^bSt-Er S[wImS aT/ tW]eNn-Tyy mE^tE[rS aNd] SlO/wS bY^ fI[vE, wH-aTs] ThE/ nEw^ SpE[eD?", 15.0),
        ("A] lOoObbSstTeR] cLaW^ eX eRrT sS ThIr.Ty TwOoo nEu-ToNs, aNd] aNoTtHeR| cLaaW~ hHaSs {tWeNtY} fOuR, wHaT/ iSs] tOtAl^ fOrCe?", 56.0),
        ("A] Lo.bS tEr] In^ A] DoMiNaNcE- FiGhT| UsEs~ A] ClAw^ FoRcE/ Of] ThIrTy TwO] NooToNs, Um] AnD{ AnTeNnA] NuDdGe| AdDs} SeVeN] NooToNs, HoW/ MuCh] ToTaL^ FoRcE>?", 39.0),
        ("A] lOoO bS tEr] ClAw] FoR^cE iS tHiRtY TwO] nEeW tOoNs um, AnD] a] RiVaaL ClAw] hAs fOuRtEeN] nEeW tOoNs ~, HoW] mUcH ToTaL FoR^cE?", 46.0),
        ("A] LoO bS t-ErRr ClAw] FoRcE^ Is ThIrTy FiVe NoOtOnS{,} AnD It GaAiN s TwElVe NoOtOnS< > DuRiNg A DoMiNaNcE FiiGhT-|, HoW MuCh ToTaL FoRcE?", 47.0),
        ("A] lO b-StErRr LoOobssTtEr ClA w{FfOorRceE} thIi rTtYy TwOo nOo OtToNns ~+~ fOoUrTtEeEn nOoOtToNns, wHaT] iIs TtOtAl| fOrRce?", 46.0),
        ("A] LoBbStErS] SwIm^ InTaKeS Um, ThIrTy] NeWtOnS[,] AnD] AnOtHeR] Lo.oBbSsTeR ExErTs TwElVe] NeWtOnS- WhAt] Is] ThE] ToTaL] FoRcE? ~", 42.0),
        ("A] Lo-bStEr Ex^eRrTs ThIrTy TwO + SiXtEeN NooToNs, UmM WhAt Is ThE ToTaL FoRcE? ~ Lo.bSt Errr Velawcitee um", 48.0),
        ("A] lO-bS tEr C lA^w E xE rTs / tH iR tY fI vE nE oO tOnS [aNd] rI vA l A dD s tW eL vE nE oO tOnS , wH aT iS tH e ToTaL F oR cE ?", 47.0),
        ("A] lO b-StEr S^wImS/ aT tW eN tY tHrEe] cM eTeRs/ PeR sEcOnD, AnD iT aCcElErAtEs~ bY SeV eN, WhAt] iS ThE/ nEw- vElOoCiTy?", 30.0),
        ("A] lO^bSt-Er S[wImS Um| aT tWrEeNtY- FiVe^ nEwToNs~ ClAw] FoRce Is- thIrTy] fIvE~ nEuT oNs, BuT/ iT lO.sEs Um| tWeL vE^ tWElVe] nEuToNs, WhAt] ReMaInS~ iN tHe/ cLaW?", 23.0),
        ("A] lOoO bS tEr^ sClAw- fOrCe Is FoR tY fIvEe] NeWtoNs ~ aNd] anTeNnA PuShEs WiTh SiX tEeEn<> NeWtOnS - WhAt Is ToTaL FoRcE?", 61.0),
        ("A] LoOoBbSsStEr S^wImS/ aT tWeNtY tHrEe CeN.tI.me.TeRs [pEr] SeC.oNd, Um, In A DoMiNaNcE PuSh \\ iTs ClAw-- AdDs FiVe NeWtOnS ~ So WhAt Is ThE* PrOdUcT?", 115.0),
        ("A] lO^bSt-Er-rrr sWiMmS~ iN/ cOoLaNt{ wAtEr| aNd- cLaW } foRce] Is^ fOrTy] nEeW/oTOnS, uHm lOoobsssster-otHer } cLaW ] iS tWeNtY< fOuR^ nOoToNs, hOw] mUcH- tOtAl/ fOrCe } toGeThEr+? errr", 64.0),
    ]
    ok = 0
    for text, expected in tests:
        got = solve_challenge(text)
        good = got is not None and abs(float(got) - expected) < 0.001
        ok += int(good)
        print(f"{'OK' if good else 'FAIL':4} exp={expected:>5} got={got or 'None':>7} | {text[:55]}")
    print(f"\n{ok}/{len(tests)} correct")
