#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KDP paperback wraparound covers (6x9 white paper, with bleed).

Usage: python build_print_cover.py [--book N]
Reads: covers/cover_bookN.jpg (front art 1600x2560), blurbs.json (back),
       print/*(print interior).pdf (page count -> spine math).
Output: print/Book N - <Title> (print cover).pdf
Size @300dpi: W = .125+6+spine+6+.125 in, H = 9.25in. Barcode box left blank
for KDP's own ISBN barcode. Spine text: title + author, rotated.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DPI = 300
AUTHOR = "Joshua Lee"
SERIES = "Quillan-Ronin Saga"
FONTS = Path(r"C:\Windows\Fonts")

BOOKS = [
    ("Book 1 - Twisted Destiny", "Twisted Destiny", 1),
    ("Book 2 - Rise of Ascension", "Rise of Ascension", 2),
    ("Book 3 - Battle Grandeur", "Battle Grandeur", 3),
    ("Book 4 - Fall of Empires", "Fall of Empires", 4),
    ("Book 5 - The Howling Shadow", "The Howling Shadow", 5),
]

WHITE_PP_IN = 0.002252
BLEED = 0.125


def page_count(num: int) -> int:
    from pypdf import PdfReader
    for f in (HERE / "print").glob(f"Book {num} - *(print interior).pdf"):
        return len(PdfReader(str(f)).pages)
    raise FileNotFoundError(f"interior PDF for book {num}")


def font(sz: int, bold=False):
    return ImageFont.truetype(str(FONTS / ("georgiab.ttf" if bold else "georgia.ttf")), sz)


def wrap(draw, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build(num: int, stem: str, title: str, total_width_in: float = 0.0) -> Path:
    pages = page_count(num)
    spine_in = pages * WHITE_PP_IN
    if total_width_in > 0:
        # KDP previewer is the judge: when its expected total differs from our
        # spine math (e.g. 13.730 vs 13.670), widen the spine to match exactly.
        spine_in = total_width_in - 12 - 2 * BLEED
        print(f"Book {num}: spine overridden to {spine_in:.3f}in (KDP expected {total_width_in:.3f})")
    W = int((BLEED + 6 + spine_in + 6 + BLEED) * DPI)
    H = int((9 + 2 * BLEED) * DPI)
    spine_px = int(spine_in * DPI)
    back_w = int(6 * DPI)

    img = Image.new("RGB", (W, H), (12, 10, 14))
    d = ImageDraw.Draw(img)

    # FRONT: cover art full-bleed
    art = Image.open(HERE / "covers" / f"cover_book{num}.jpg").convert("RGB")
    fx = BLEED + 6 + spine_in
    front = art.resize((int(6.25 * DPI), H), Image.LANCZOS)
    img.paste(front, (int(fx * DPI), 0))

    # SPINE: title + author, centered
    spx = int((BLEED + 6) * DPI)
    fnt_s = font(64, bold=True)
    txt = f"{title}  \u2022  {AUTHOR}"
    # draw horizontal on wide canvas, rotate into the spine strip
    strip = Image.new("RGB", (H, spine_px), (12, 10, 14))
    ds = ImageDraw.Draw(strip)
    tw = ds.textlength(txt, font=fnt_s)
    ds.text(((H - tw) / 2, (spine_px - 64) / 2), txt, font=fnt_s, fill=(235, 225, 200))
    img.paste(strip.rotate(90, expand=True), (spx, 0))

    # BACK: series head, blurb, author, barcode box
    blurbs = json.loads((HERE / "blurbs.json").read_text(encoding="utf-8"))
    blurb = blurbs[str(num)].split("\n")
    bx = int((BLEED + 0.6) * DPI)
    y = int((BLEED + 0.7) * DPI)
    maxw = back_w - int(1.2 * DPI)
    fnt_h = font(54, bold=True)
    d.text((bx, y), SERIES, font=fnt_h, fill=(200, 170, 90))
    y += 110
    fnt_b = font(44)
    for para in blurb:
        if not para.strip():
            y += 30
            continue
        for line in wrap(d, para.strip(), fnt_b, maxw):
            d.text((bx, y), line, font=fnt_b, fill=(232, 228, 220))
            y += 62
        y += 22
    fnt_a = font(48, bold=True)
    d.text((bx, H - int(2.6 * DPI)), f"by {AUTHOR}", font=fnt_a, fill=(200, 170, 90))
    # barcode placeholder (KDP overlays its own)
    bw, bh = int(2.0 * DPI), int(1.2 * DPI)
    d.rectangle([bx + maxw - bw + int(0.6 * DPI), H - int(BLEED * DPI) - bh - 40,
                 bx + maxw + int(0.6 * DPI), H - int(BLEED * DPI) - 40],
                fill=(255, 255, 255))

    out = HERE / "print" / f"{stem} (print cover).pdf"
    img.save(out, "PDF", resolution=DPI)
    print(f"Book {num}: {out.name} ({W}x{H}px, spine {spine_in:.3f}in, {pages}pp)")
    return out


def main():
    only = None
    if "--book" in sys.argv:
        only = int(sys.argv[sys.argv.index("--book") + 1])
    width_override = {}
    if "--cover-width" in sys.argv:
        # --cover-width N:W pairs KDP's expected total, e.g. --cover-width 1:13.73
        for pair in sys.argv[sys.argv.index("--cover-width") + 1:]:
            if ":" not in pair:
                break
            n, w = pair.split(":")
            width_override[int(n)] = float(w)
    for stem, title, num in BOOKS:
        if only and num != only:
            continue
        build(num, stem, title, width_override.get(num, 0.0))


if __name__ == "__main__":
    main()
