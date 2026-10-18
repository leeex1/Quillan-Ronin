#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KDP Cover Generator & Rescaler (Quillan-Ronin Saga)
--------------------------------------------------
Rescales draft cover artwork to precisely fit official Amazon KDP downloaded templates:
1. Paperback Book 1 Exact KDP Preflight: 13.730" x 9.250" (988.56 x 666.0 pt) @ 300 DPI (4119 x 2775 px)
2. Case Laminate Hardcover 550-page BW Cream: 15.139" x 10.417" (1090.0 x 750.0 pt) @ 300 DPI (4542 x 3125 px)
3. Formatted Front Covers: 6.125" x 9.250" (441.0 x 666.0 pt) @ 300 DPI (1838 x 2775 px)

Handles bleed zones, spine typography rotation, back cover blurbs, and barcode exclusion safe zones.
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pypdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(r"C:\02_QUILLAN\09 - Projects\projects\Book Series").resolve()
DRAFTS_DIR = ROOT / "book covers drafts"
UPSCALED_DIR = Path(r"C:\02_QUILLAN\02 - Knowledge Foundation\knowledge\legacy\00 - Meta\toolkit\session\upscaled").resolve()
OUT_DIR = ROOT / "print"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLURBS_PATH = ROOT / "blurbs.json"
FONTS_DIR = Path(r"C:\Windows\Fonts")

DPI = 300
AUTHOR = "Joshua Lee"
SERIES = "Quillan-Ronin Saga"

BOOKS = [
    {
        "num": 1,
        "title": "Twisted Destiny",
        "file_stem": "Book 1 - Twisted Destiny",
        "draft_name": "twisted destiny.jpg",
        "upscaled_name": "cover_book1.png",
    },
    {
        "num": 2,
        "title": "Rise of Ascension",
        "file_stem": "Book 2 - Rise of Ascension",
        "draft_name": "rise of acenssion.jpg",
        "upscaled_name": "cover_book2.png",
    },
    {
        "num": 3,
        "title": "Battle Grandeur",
        "file_stem": "Book 3 - Battle Grandeur",
        "draft_name": "battle of grandur.jpg",
        "upscaled_name": "cover_book3.png",
    },
    {
        "num": 4,
        "title": "Fall of Empires",
        "file_stem": "Book 4 - Fall of Empires",
        "draft_name": "fall of empires.jpg",
        "upscaled_name": "cover_book4.png",
    },
    {
        "num": 5,
        "title": "The Howling Shadow",
        "file_stem": "Book 5 - The Howling Shadow",
        "draft_name": "The howling Shadow.jpg",
        "upscaled_name": "cover_book5.png",
    },
]

# Actual Interior Page Counts (from compiled print interiors)
BOOK_PAGE_COUNTS = {
    1: 631,  # Exact Amazon KDP preflight requires 13.730" x 9.250"
    2: 527,
    3: 575,
    4: 484,
    5: 531,
}

def get_paperback_specs(book_num: int) -> dict:
    """
    Returns exact template specs for KDP Paperback based on interior page count and paper type.
    - Book 1 (631 pages): Exact Amazon KDP preflight mandates 13.730" x 9.250" (1.480" spine).
    - Book 2 (527 pages): Exact Amazon KDP preflight mandates 13.568" x 9.250" (1.318" spine, cream stock).
    - Books 3-5: Dynamic calculation using KDP standard B&W Cream paper caliper (0.0025" / page).
    """
    if book_num == 1:
        return {
            "name": "Paperback Book 1 Exact KDP (13.730x9.250)",
            "width_in": 13.730,
            "height_in": 9.250,
            "spine_in": 1.480,
            "bleed_in": 0.125,
            "trim_w_in": 6.000,
            "trim_h_in": 9.000,
            "width_px": 4119,
            "height_px": 2775,
            "spine_px": 444,
            "front_x_px": 2282,
            "front_w_px": 1837,
        }

    if book_num == 2:
        return {
            "name": "Paperback Book 2 Exact KDP (13.568x9.250)",
            "width_in": 13.568,
            "height_in": 9.250,
            "spine_in": 1.318,
            "bleed_in": 0.125,
            "trim_w_in": 6.000,
            "trim_h_in": 9.000,
            "width_px": 4070,
            "height_px": 2775,
            "spine_px": 395,
            "front_x_px": 2233,
            "front_w_px": 1837,
        }

    # Calibrated cream paper stock calculation (0.0025 in / page) for other books
    pages = BOOK_PAGE_COUNTS.get(book_num, 550)
    spine_in = round(pages * 0.0025, 3)
    width_in = round(12.250 + spine_in, 3)
    height_in = 9.250
    width_px = int(round(width_in * DPI))
    height_px = int(round(height_in * DPI))
    spine_px = int(round(spine_in * DPI))
    front_w_px = int(round(6.125 * DPI))
    front_x_px = width_px - front_w_px

    return {
        "name": f"Paperback Book {book_num} ({pages}pp Cream, {width_in}x{height_in})",
        "width_in": width_in,
        "height_in": height_in,
        "spine_in": spine_in,
        "bleed_in": 0.125,
        "trim_w_in": 6.000,
        "trim_h_in": 9.000,
        "width_px": width_px,
        "height_px": height_px,
        "spine_px": spine_px,
        "front_x_px": front_x_px,
        "front_w_px": front_w_px,
    }

HARDCOVER_SPECS = {
    "name": "Case Laminate Hardcover 550pp BW Cream",
    "width_in": 15.139,
    "height_in": 10.417,
    "spine_in": 1.564,
    "bleed_in": 0.589,  # wrap allowance
    "trim_w_in": 6.000,
    "trim_h_in": 9.000,
    "width_px": 4542,
    "height_px": 3125,
    "spine_px": 469,
    "front_x_px": 2505,
    "front_w_px": 2037,
}


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    fname = "georgiab.ttf" if bold else "georgia.ttf"
    fpath = FONTS_DIR / fname
    if not fpath.exists():
        fname = "arialbd.ttf" if bold else "arial.ttf"
        fpath = FONTS_DIR / fname
    return ImageFont.truetype(str(fpath), size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list:
    words = text.split()
    lines = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def load_source_artwork(book: dict) -> Image.Image:
    """Loads highest quality available source draft art."""
    upscaled_path = UPSCALED_DIR / book["upscaled_name"]
    if upscaled_path.exists():
        logger.info(f"Book {book['num']}: Loading high-resolution master art ({upscaled_path.name})")
        return Image.open(upscaled_path).convert("RGB")

    draft_path = DRAFTS_DIR / book["draft_name"]
    if draft_path.exists():
        logger.info(f"Book {book['num']}: Loading draft art ({draft_path.name})")
        return Image.open(draft_path).convert("RGB")

    fallback = ROOT / "covers" / f"cover_book{book['num']}.jpg"
    logger.info(f"Book {book['num']}: Loading fallback art ({fallback.name})")
    return Image.open(fallback).convert("RGB")


def build_full_wrap_cover(book: dict, specs: dict, blurbs: dict, suffix: str) -> Path:
    """Builds a complete, print-ready full-bleed KDP wraparound PDF matching template specifications."""
    W = specs["width_px"]
    H = specs["height_px"]
    spine_px = specs["spine_px"]
    front_x = specs["front_x_px"]
    front_w = specs["front_w_px"]

    # Background canvas: Deep Void obsidian
    canvas = Image.new("RGB", (W, H), (14, 12, 16))
    draw = ImageDraw.Draw(canvas)

    # 1. Front Cover Artwork — Inset inside KDP live safe zone (>= 0.25" from trim) with seamless bleed extension
    source_art = load_source_artwork(book)
    front_canvas = Image.new("RGB", (front_w, H), (14, 12, 16))

    trim_w_px = int(specs["trim_w_in"] * DPI)
    art_w = min(int(trim_w_px - 160), 1640)
    art_h = int(art_w * source_art.height / source_art.width)
    if art_h > (H - 332):
        art_h = H - 332
        art_w = int(art_h * source_art.width / source_art.height)

    scaled_art = source_art.resize((art_w, art_h), Image.LANCZOS)
    art_x = (trim_w_px - art_w) // 2
    art_y = (H - art_h) // 2

    # Seamless background extension strips to fill bleed without repeating text
    top_strip = scaled_art.crop((0, 0, art_w, 1)).resize((art_w, art_y), Image.BILINEAR)
    front_canvas.paste(top_strip, (art_x, 0))
    bot_strip = scaled_art.crop((0, art_h - 1, art_w, art_h)).resize((art_w, H - (art_y + art_h)), Image.BILINEAR)
    front_canvas.paste(bot_strip, (art_x, art_y + art_h))
    left_strip = front_canvas.crop((art_x, 0, art_x + 1, H)).resize((art_x, H), Image.BILINEAR)
    front_canvas.paste(left_strip, (0, 0))
    right_strip = front_canvas.crop((art_x + art_w - 1, 0, art_x + art_w, H)).resize((front_w - (art_x + art_w), H), Image.BILINEAR)
    front_canvas.paste(right_strip, (art_x + art_w, 0))
    front_canvas.paste(scaled_art, (art_x, art_y))

    canvas.paste(front_canvas, (front_x, 0))

    # 2. Spine Text: Title & Author, centered with safe margin clearance from fold lines
    strip = Image.new("RGB", (H, spine_px), (14, 12, 16))
    d_strip = ImageDraw.Draw(strip)
    fnt_sz = max(24, int(spine_px * 0.12))
    fnt_spine = get_font(fnt_sz, bold=True)
    spine_title = book["title"].upper()
    spine_auth = AUTHOR.upper()
    spine_txt = f"{spine_title}   \u2022   {spine_auth}"
    tw = d_strip.textlength(spine_txt, font=fnt_spine)
    d_strip.text(
        ((H - tw) / 2, (spine_px - fnt_sz) / 2 - 4),
        spine_txt,
        font=fnt_spine,
        fill=(240, 230, 205),
    )
    # Rotate 90 degrees counter-clockwise (standard book spine reading top-to-bottom)
    spine_rotated = strip.rotate(90, expand=True)
    spx = front_x - spine_px
    canvas.paste(spine_rotated, (spx, 0))

    # 3. Back Cover Copy: Series header, manuscript blurb, author credit, barcode blank
    blurb_content = blurbs.get(str(book["num"]), "")
    bx = int(specs["bleed_in"] * DPI + 0.85 * DPI)
    by = int(1.15 * DPI)
    back_live_w = int(specs["trim_w_in"] * DPI - 1.4 * DPI)

    fnt_series = get_font(48, bold=True)
    draw.text((bx, by), SERIES.upper(), font=fnt_series, fill=(212, 175, 55))  # Sovereign Gold
    by += 95

    fnt_body = get_font(38)
    for para in blurb_content.split("\n"):
        para = para.strip()
        if not para:
            by += 24
            continue
        lines = wrap_text(draw, para, fnt_body, back_live_w)
        for line in lines:
            draw.text((bx, by), line, font=fnt_body, fill=(235, 230, 222))
            by += 54
        by += 18

    fnt_author = get_font(44, bold=True)
    draw.text((bx, H - int(specs["bleed_in"] * DPI + 2.5 * DPI)), f"By {AUTHOR}", font=fnt_author, fill=(212, 175, 55))

    # Barcode placeholder zone (2.0" x 1.2" blank white safe box for KDP ISBN generation)
    bw_px = int(2.0 * DPI)
    bh_px = int(1.2 * DPI)
    margin_b = int(specs["bleed_in"] * DPI + 0.25 * DPI)
    margin_r = spx - int(0.6 * DPI)
    draw.rectangle(
        [margin_r - bw_px, H - margin_b - bh_px, margin_r, H - margin_b],
        fill=(255, 255, 255),
    )

    out_pdf = OUT_DIR / f"{book['file_stem']} ({suffix}).pdf"
    canvas.save(str(out_pdf), "PDF", resolution=DPI)

    # Exact point precision calibration for Amazon KDP preflight
    target_w_pt = round(float(specs["width_in"]) * 72.0, 3)
    target_h_pt = round(float(specs["height_in"]) * 72.0, 3)
    try:
        reader = pypdf.PdfReader(str(out_pdf))
        writer = pypdf.PdfWriter()
        page = reader.pages[0]
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (target_w_pt, target_h_pt)
        writer.add_page(page)
        with open(out_pdf, "wb") as f:
            writer.write(f)
    except Exception as e:
        logger.warning(f"Could not calibrate MediaBox points for {out_pdf.name}: {e}")

    logger.info(f"Generated Full-Wrap PDF: {out_pdf.name} ({W}x{H} px @ {DPI} DPI, {float(specs['width_in']):.3f}x{float(specs['height_in']):.3f} in)")
    return out_pdf


def build_front_cover_pdf(book: dict) -> Path:
    """Builds a formatted standalone 6.125" x 9.250" front cover PDF with bleed and safe margin inset."""
    W = int(6.125 * DPI)
    H = int(9.250 * DPI)
    trim_w_px = int(6.0 * DPI)
    source_art = load_source_artwork(book)
    front_canvas = Image.new("RGB", (W, H), (14, 12, 16))

    art_w = min(int(trim_w_px - 160), 1640)
    art_h = int(art_w * source_art.height / source_art.width)
    if art_h > (H - 332):
        art_h = H - 332
        art_w = int(art_h * source_art.width / source_art.height)

    scaled_art = source_art.resize((art_w, art_h), Image.LANCZOS)
    art_x = (trim_w_px - art_w) // 2
    art_y = (H - art_h) // 2

    top_strip = scaled_art.crop((0, 0, art_w, 1)).resize((art_w, art_y), Image.BILINEAR)
    front_canvas.paste(top_strip, (art_x, 0))
    bot_strip = scaled_art.crop((0, art_h - 1, art_w, art_h)).resize((art_w, H - (art_y + art_h)), Image.BILINEAR)
    front_canvas.paste(bot_strip, (art_x, art_y + art_h))
    left_strip = front_canvas.crop((art_x, 0, art_x + 1, H)).resize((art_x, H), Image.BILINEAR)
    front_canvas.paste(left_strip, (0, 0))
    right_strip = front_canvas.crop((art_x + art_w - 1, 0, art_x + art_w, H)).resize((W - (art_x + art_w), H), Image.BILINEAR)
    front_canvas.paste(right_strip, (art_x + art_w, 0))
    front_canvas.paste(scaled_art, (art_x, art_y))

    out_pdf = OUT_DIR / f"{book['file_stem']} (Front Cover - Formatted).pdf"
    front_canvas.save(str(out_pdf), "PDF", resolution=DPI)

    try:
        reader = pypdf.PdfReader(str(out_pdf))
        writer = pypdf.PdfWriter()
        page = reader.pages[0]
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (6.125 * 72.0, 9.250 * 72.0)
        writer.add_page(page)
        with open(out_pdf, "wb") as f:
            writer.write(f)
    except Exception as e:
        logger.warning(f"Could not calibrate MediaBox points for {out_pdf.name}: {e}")

    logger.info(f"Generated Front Cover PDF: {out_pdf.name} ({W}x{H} px @ {DPI} DPI)")
    return out_pdf


def build_scaled_draft_template_pdf(book: dict, specs: dict) -> Path:
    """Rescales the draft image directly to fit the exact template overall dimensions."""
    W = specs["width_px"]
    H = specs["height_px"]
    source_art = load_source_artwork(book)
    scaled = source_art.resize((W, H), Image.LANCZOS)

    out_pdf = OUT_DIR / f"{book['file_stem']} (Draft Scaled to Template).pdf"
    scaled.save(str(out_pdf), "PDF", resolution=DPI)

    target_w_pt = round(float(specs["width_in"]) * 72.0, 3)
    target_h_pt = round(float(specs["height_in"]) * 72.0, 3)
    try:
        reader = pypdf.PdfReader(str(out_pdf))
        writer = pypdf.PdfWriter()
        page = reader.pages[0]
        page.mediabox.lower_left = (0, 0)
        page.mediabox.upper_right = (target_w_pt, target_h_pt)
        writer.add_page(page)
        with open(out_pdf, "wb") as f:
            writer.write(f)
    except Exception as e:
        logger.warning(f"Could not calibrate MediaBox points for {out_pdf.name}: {e}")

    logger.info(f"Generated Draft-Scaled Template PDF: {out_pdf.name} ({W}x{H} px @ {DPI} DPI, {float(specs['width_in']):.3f}x{float(specs['height_in']):.3f} in)")
    return out_pdf


def main():
    logger.info("Initializing KDP Cover Rescaling and PDF Generation...")
    blurbs = {}
    if BLURBS_PATH.exists():
        blurbs = json.loads(BLURBS_PATH.read_text(encoding="utf-8"))

    for book in BOOKS:
        logger.info(f"--- Processing Book {book['num']}: {book['title']} ---")

        pb_specs = get_paperback_specs(book["num"])
        logger.info(f"Using Paperback Specs: {pb_specs['name']} -> {pb_specs['width_in']}x{pb_specs['height_in']} in ({pb_specs['width_px']}x{pb_specs['height_px']} px)")

        # 1. Official Paperback calibrated template full wrap
        build_full_wrap_cover(book, pb_specs, blurbs, "Paperback 550pp Cover")

        # Also save to standard legacy name (print cover.pdf)
        build_full_wrap_cover(book, pb_specs, blurbs, "print cover")

        # 2. Official Case Laminate Hardcover 550pp template full wrap (15.139" x 10.417")
        build_full_wrap_cover(book, HARDCOVER_SPECS, blurbs, "Hardcover 550pp Cover")

        # 3. Standalone Front Cover PDF (6.125" x 9.250" with bleed)
        build_front_cover_pdf(book)

        # 4. Direct Draft Scaled to Template PDF
        build_scaled_draft_template_pdf(book, pb_specs)

    logger.info("All 5 Book Covers and Formatted PDFs Generated Successfully!")


if __name__ == "__main__":
    main()
