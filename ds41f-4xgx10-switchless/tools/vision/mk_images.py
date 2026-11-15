#!/usr/bin/env python3
"""mk_images.py OUTDIR: draw the ten PNGs cases.json names, plus 03b_count_red5.png, and copy cases.json
into OUTDIR. The images imgtest.py sends exist nowhere, so they are drawn here.

Usage: python3 mk_images.py OUTDIR        (Pillow only)

Every position is fixed, so the output is byte-stable; no GPU and no network are used.
"""
import json
import os
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

RED = (200, 30, 30)
BLUE = (30, 70, 200)
GREEN = (20, 150, 60)
YELLOW = (240, 200, 20)
PURPLE = (130, 40, 160)
WHITE = (255, 255, 255)
BLACK = (10, 10, 10)
CODE_BG = (30, 30, 34)


def font(size):
    """DejaVuSans-Bold, or Pillow's built-in font when that file is missing."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def canvas(size, bg=WHITE):
    im = Image.new("RGB", size, bg)
    return im, ImageDraw.Draw(im)


def center(d, xy, text, f, fill):
    """Text centred on xy (textbbox works with any font, anchor does not); integer coordinates."""
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text((xy[0] - (l + r) // 2, xy[1] - (t + b) // 2), text, font=f, fill=fill)


def grid(n_cols, n_rows, x0, y0, dx, dy):
    """Slot centres, left to right then top to bottom."""
    return [(x0 + c * dx, y0 + r * dy) for r in range(n_rows) for c in range(n_cols)]


def sign(path):
    """GATE 47B / LISBON."""
    im, d = canvas((640, 480), (25, 90, 60))
    d.rectangle([16, 16, 623, 463], outline=WHITE, width=8)
    center(d, (320, 170), "GATE 47B", font(72), WHITE)
    center(d, (320, 310), "LISBON", font(76), WHITE)
    im.save(path)


def receipt(path):
    """CORNER CAFE, three items, TOTAL 10.50 (3.50 + 3.00 + 4.00)."""
    im, d = canvas((480, 640))
    f = font(34)
    lines = ["CORNER CAFE", "---------------", "Coffee    3.50", "Cake      3.00", "Tea       4.00",
             "---------------", "TOTAL    10.50"]
    for i, s in enumerate(lines):
        d.text((40, 50 + i * 62), s, font=f, fill=BLACK)
    im.save(path)


def circles(path, colors):
    """One circle per colour in colors, on the fixed 5x2 grid; the circles never touch."""
    im, d = canvas((640, 480))
    for (x, y), c in zip(grid(5, 2, 90, 150, 115, 170), colors):
        d.ellipse([x - 45, y - 45, x + 45, y + 45], fill=c)
    im.save(path)


def shapes(path, kinds):
    """One blue square ('s') or green triangle ('t') per letter of kinds, on the fixed 3x3 grid."""
    im, d = canvas((640, 480))
    for (x, y), k in zip(grid(3, 3, 130, 110, 190, 135), kinds):
        if k == "s":
            d.rectangle([x - 45, y - 45, x + 45, y + 45], fill=BLUE)
        else:
            d.polygon([(x, y - 55), (x - 55, y + 45), (x + 55, y + 45)], fill=GREEN)
    im.save(path)


def bar_chart(path):
    """Mon-Fri orders, Thu = 45 the highest, every bar labelled with its value."""
    im, d = canvas((640, 480))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    vals = [20, 35, 28, 45, 40]
    d.line([70, 60, 70, 400], fill=BLACK, width=3)
    d.line([70, 400, 610, 400], fill=BLACK, width=3)
    d.text((80, 16), "Orders per day", font=font(30), fill=BLACK)
    for i, (day, v) in enumerate(zip(days, vals)):
        x = 100 + i * 100
        d.rectangle([x, 400 - v * 6, x + 60, 400], fill=BLUE)
        center(d, (x + 30, 400 - v * 6 - 24), str(v), font(28), BLACK)
        center(d, (x + 30, 424), day, font(28), BLACK)
    im.save(path)


def table(path):
    """City / population, Osaka (19.2 M) the largest."""
    im, d = canvas((640, 480))
    rows = [("City", "Population"), ("Tokyo", "13.9 M"), ("Osaka", "19.2 M"), ("Nagoya", "9.5 M"),
            ("Sapporo", "2.6 M")]
    f = font(32)
    for i, (a, b) in enumerate(rows):
        y = 70 + i * 70
        d.line([60, y + 50, 580, y + 50], fill=BLACK, width=2)
        d.text((80, y), a, font=f, fill=BLACK)
        d.text((360, y), b, font=f, fill=BLACK)
    d.rectangle([60, 50, 580, 70 + len(rows) * 70 - 20], outline=BLACK, width=3)
    im.save(path)


def code(path):
    """A Python screenshot whose program prints 18."""
    im, d = canvas((640, 400), CODE_BG)
    for i, s in enumerate(["a = 3", "b = 6", "print(a * b)"]):
        d.text((50, 60 + i * 80), s, font=font(40), fill=(220, 220, 220))
    im.save(path)


def flag(path):
    """Vertical green / white / red tricolour."""
    im, d = canvas((600, 400))
    for i, c in enumerate((GREEN, WHITE, RED)):
        d.rectangle([i * 200, 0, (i + 1) * 200 - 1, 399], fill=c)
    d.rectangle([0, 0, 599, 399], outline=(120, 120, 120), width=2)
    im.save(path)


def math(path):
    """17 x 23 + 4 = ?"""
    im, d = canvas((640, 320))
    center(d, (320, 160), "17 \u00d7 23 + 4 = ?", font(64), BLACK)
    im.save(path)


def position(path):
    """A yellow circle top-right and a purple square bottom-left."""
    im, d = canvas((640, 480))
    d.ellipse([440, 50, 560, 170], fill=YELLOW)
    d.rectangle([60, 340, 180, 460], fill=PURPLE)
    im.save(path)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: mk_images.py OUTDIR")
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    sign(os.path.join(out, "01_ocr_sign.png"))
    receipt(os.path.join(out, "02_ocr_receipt.png"))
    circles(os.path.join(out, "03_count_red.png"), [RED, RED, BLUE, RED, RED, RED, BLUE, RED, BLUE, RED])
    shapes(os.path.join(out, "04_count_shapes.png"), "stsststtt")
    bar_chart(os.path.join(out, "05_bar_chart.png"))
    table(os.path.join(out, "06_table.png"))
    code(os.path.join(out, "07_code.png"))
    flag(os.path.join(out, "08_flag.png"))
    math(os.path.join(out, "09_math.png"))
    position(os.path.join(out, "10_position.png"))
    circles(os.path.join(out, "03b_count_red5.png"), [RED] * 5)  # 03's size, five red circles, no blue
    shutil.copyfile(os.path.join(HERE, "cases.json"), os.path.join(out, "cases.json"))
    print("wrote 11 PNGs and cases.json to %s" % out)


if __name__ == "__main__":
    main()
