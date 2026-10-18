#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print interior PDFs for KDP paperback (6x9, B&W).

Usage: python build_print_pdf.py [--book N]
Output: print/Book N - <Title> (print interior).pdf

Spec: 6in x 9in, inside 0.75in / outside 0.6in / top+bottom 0.75in,
Georgia 11pt justified body, chapter per H2 on new page, folio bottom-center.
Fonts embedded (Georgia from Windows Fonts). No-bleed (no full-bleed art).
"""
import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, PageBreak, NextPageTemplate)

HERE = Path(__file__).resolve().parent
SRC = Path(r"C:\02_QUILLAN\05 - Creative Works\Book Series\Manuscripts")
OUT = Path(r"C:\02_QUILLAN\05 - Creative Works\Book Series\Builds\Print")
FONTS = Path(r"C:\Windows\Fonts")
AUTHOR = "Joshua Lee"

PAGE_W, PAGE_H = 6 * inch, 9 * inch
# Symmetric 0.75in sides satisfy KDP inside-margin minimums for every page
# count up to 800 (inside req tops out at 0.75in); outside min is 0.25in.
M_SIDE, M_TB = 0.75 * inch, 0.75 * inch

BOOKS = [
    ("Book 1 - Twisted Destiny.md", "Twisted Destiny", 1),
    ("Book 2 - Rise of Ascension.md", "Rise of Ascension", 2),
    ("Book 3 - Battle Grandeur.md", "Battle Grandeur", 3),
    ("Book 4 - Fall of Empires.md", "Fall of Empires", 4),
    ("Book 5 - The Howling Shadow.md", "The Howling Shadow", 5),
]

pdfmetrics.registerFont(TTFont("Georgia", str(FONTS / "georgia.ttf")))
pdfmetrics.registerFont(TTFont("Georgia-Bold", str(FONTS / "georgiab.ttf")))
pdfmetrics.registerFont(TTFont("Georgia-Italic", str(FONTS / "georgiai.ttf")))

body_style = ParagraphStyle("body", fontName="Georgia", fontSize=11, leading=15,
                            alignment=TA_JUSTIFY, spaceAfter=6,
                            firstLineIndent=18)
chap_style = ParagraphStyle("chap", fontName="Georgia-Bold", fontSize=18, leading=24,
                            alignment=TA_CENTER, spaceBefore=72, spaceAfter=24)
title_style = ParagraphStyle("title", fontName="Georgia-Bold", fontSize=28, leading=34,
                             alignment=TA_CENTER)
center_style = ParagraphStyle("center", fontName="Georgia", fontSize=11, leading=15,
                              alignment=TA_CENTER, spaceAfter=6)
copy_style = ParagraphStyle("copy", fontName="Georgia", fontSize=9, leading=12,
                            alignment=TA_JUSTIFY, spaceAfter=4)


def md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def split_chapters(text: str):
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                lines = lines[i + 1:]
                break
    chunks, cur_head, cur = [], None, []
    for line in lines:
        m = re.match(r"^##\s+(.*)$", line)
        if m:
            if cur_head is not None or cur:
                chunks.append((cur_head or "", cur))
            cur_head, cur = m.group(1).strip(), []
        else:
            cur.append(line)
    if cur_head is not None or cur:
        chunks.append((cur_head or "", cur))
    # drop pasted TOC chapter
    chunks = [(h, l) for h, l in chunks if "table of contents" not in h.lower()]
    return chunks


def folio(canvas, doc):
    canvas.saveState()
    canvas.setFont("Georgia", 9)
    canvas.drawCentredString(PAGE_W / 2, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def build(src_file: Path, title: str, num: int) -> Path:
    text = src_file.read_text(encoding="utf-8", errors="replace")
    chunks = split_chapters(text)

    story = []
    # title page
    story += [Spacer(1, 2.2 * inch),
              Paragraph(md_inline(title), title_style), Spacer(1, 0.3 * inch),
              Paragraph(f"Book {num} of the Hybrid Throne", center_style),
              Spacer(1, 1.5 * inch), Paragraph(AUTHOR, center_style), PageBreak()]
    # copyright page
    story += [Spacer(1, 2.5 * inch),
              Paragraph(f"{title}<br/>Book {num} of the Hybrid Throne<br/><br/>"
                        f"Copyright \u00a9 2026 by {AUTHOR}<br/><br/>"
                        "All rights reserved. No part of this publication may be reproduced, "
                        "distributed, or transmitted in any form or by any means without the "
                        "prior written permission of the author.<br/><br/>"
                        "This is a work of fiction. Names, characters, places, and incidents "
                        "are products of the author's imagination.<br/><br/>"
                        "First edition, 2026", copy_style),
              PageBreak()]
    # single auto TOC (one page, no dupes)
    story += [Paragraph("Contents", chap_style)]
    toc_style = ParagraphStyle("toc", parent=center_style, alignment=0, leftIndent=36, spaceAfter=3)
    CHAP_RE = __import__("re")
    for heading, lines in chunks:
        if heading and heading != "Front Matter":
            story.append(Paragraph(CHAP_RE.sub(r"^#{1,3}\s+", "", heading) if heading.startswith("#") else heading, toc_style))
    story.append(PageBreak())
    # chapters
    for heading, lines in chunks:
        if heading and heading != "Front Matter":
            story.append(Paragraph(md_inline(heading), chap_style))
        para = []
        def flush():
            if para:
                story.append(Paragraph(" ".join(para), body_style))
                para.clear()
        for raw in lines:
            line = raw.rstrip()
            if not line.strip():
                flush()
                continue
            if re.match(r"^#{1,3}\s+", line):
                flush()
                story.append(Paragraph(md_inline(re.sub(r"^#{1,3}\s+", "", line)), chap_style))
                continue
            if re.match(r"^---+\s*$", line):
                flush()
                story.append(Paragraph("* * *", center_style))
                continue
            if re.match(r"^\s*[-*]\s+", line):
                flush()
                story.append(Paragraph("\u2022 " + md_inline(re.sub(r"^\s*[-*]\s+", "", line)), body_style))
                continue
            if line.lstrip().startswith(">"):
                flush()
                story.append(Paragraph("<i>" + md_inline(line.lstrip()[1:].strip()) + "</i>", body_style))
                continue
            if re.match(r"!\[.*?\]\(.*?\)", line):
                continue
            para.append(line.strip())
        flush()
        story.append(PageBreak())

    OUT.mkdir(exist_ok=True)
    dest = OUT / f"{src_file.stem} (print interior).pdf"

    doc = BaseDocTemplate(str(dest), pagesize=(PAGE_W, PAGE_H),
                          leftMargin=M_SIDE, rightMargin=M_SIDE,
                          topMargin=M_TB, bottomMargin=M_TB,
                          title=title, author=AUTHOR)
    frame = Frame(M_SIDE, M_TB, PAGE_W - 2 * M_SIDE, PAGE_H - 2 * M_TB, id="body")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=folio)])
    doc.build(story)
    return dest


def main():
    only = None
    if "--book" in sys.argv:
        only = int(sys.argv[sys.argv.index("--book") + 1])
    for fname, title, num in BOOKS:
        if only and num != only:
            continue
        dest = build(SRC / fname, title, num)
        print(f"Book {num}: {dest.name} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()


