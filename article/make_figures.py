"""Draw the figures for the mesh Sudoku article. Writes one SVG per figure next to this file.

Run: .venv/bin/python article/make_figures.py
Then turn each SVG into a PNG with: rsvg-convert -z 2 fig.svg -o fig.png
"""
import os
import re
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "..", "data", "logs")

W, H = 1200, 675
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One color per relation. The same three are used in every figure.
ROW, COL, BOX = "#2a78d6", "#eb6834", "#1baf7a"
ROW_T, COL_T, BOX_T = "#cde2fb", "#fbdccf", "#cdeee1"
FONT = "font-family='Helvetica Neue, Helvetica, Arial, sans-serif'"


def text(x, y, s, size=16, fill=INK, weight="normal", anchor="start"):
    return f"<text x='{x}' y='{y}' font-size='{size}' fill='{fill}' font-weight='{weight}' text-anchor='{anchor}'>{s}</text>"


def page(name, title, subtitle, body):
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' height='{H}' {FONT}>"
           "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
           f"<path d='M0,0 L10,5 L0,10 z' fill='{INK2}'/></marker></defs>"
           f"<rect width='{W}' height='{H}' fill='{SURFACE}'/>"
           + text(60, 68, title, 30, weight="bold") + text(60, 100, subtitle, 18, INK2)
           + "".join(body) + "</svg>")
    with open(os.path.join(HERE, name), "w") as f:
        f.write(svg)


def relation(i, j):
    """Return the color of the rule that links cells i and j, or None when no rule links them."""
    if i == j:
        return None
    if i // 9 == j // 9:
        return ROW
    if i % 9 == j % 9:
        return COL
    if (i // 27, i % 9 // 3) == (j // 27, j % 9 // 3):
        return BOX
    return None


def fig_mesh():
    s, x0, y0 = 52, 60, 150
    focus = 4 * 9 + 1
    tint = {ROW: ROW_T, COL: COL_T, BOX: BOX_T}
    b = []
    for i in range(81):
        r, c = divmod(i, 9)
        x, y = x0 + c * s, y0 + r * s
        fill = INK if i == focus else tint.get(relation(focus, i), SURFACE)
        b.append(f"<rect x='{x}' y='{y}' width='{s}' height='{s}' fill='{fill}' stroke='{GRID}'/>")
        # A valid solved board, with most cells hidden so it reads as a puzzle.
        digit = (3 * (r % 3) + r // 3 + c) % 9 + 1
        if i == focus:
            b.append(text(x + s / 2, y + 35, "?", 26, SURFACE, "bold", "middle"))
        elif (i * 7 + r) % 3 == 0:
            b.append(text(x + s / 2, y + 34, digit, 22, INK, anchor="middle"))
    for k in range(4):
        b.append(f"<line x1='{x0 + k * 3 * s}' y1='{y0}' x2='{x0 + k * 3 * s}' y2='{y0 + 9 * s}' stroke='{INK}' stroke-width='2'/>")
        b.append(f"<line x1='{x0}' y1='{y0 + k * 3 * s}' x2='{x0 + 9 * s}' y2='{y0 + k * 3 * s}' stroke='{INK}' stroke-width='2'/>")

    x = 610
    rows = [(ROW, ROW_T, "Its row", "8 cells"), (COL, COL_T, "Its column", "8 cells"),
            (BOX, BOX_T, "Its box", "8 cells, 4 of them already counted")]
    for k, (edge, fill, name, note) in enumerate(rows):
        y = 180 + k * 62
        b.append(f"<rect x='{x}' y='{y}' width='36' height='36' rx='4' fill='{fill}' stroke='{edge}' stroke-width='2'/>")
        b.append(text(x + 54, y + 17, name, 20, weight="bold") + text(x + 54, y + 40, note, 16, INK2))
    b.append(text(x, 440, "20", 84, weight="bold") + text(x + 112, 412, "neighbors.", 24, weight="bold"))
    b.append(text(x + 112, 440, "The cell hears from these in one pass.", 16, INK2))
    b.append(text(x, 540, "60", 84, INK2, "bold") + text(x + 112, 512, "cells it never hears from directly.", 24, INK2, "bold"))
    b.append(text(x + 112, 540, "Their news arrives second hand, over more passes.", 16, INK2))
    page("fig1_mesh.svg", "A Sudoku board is a mesh",
         "Each cell is wired only to the cells it shares a rule with.", b)


def fig_mask():
    px, y0 = 5, 190
    b = []
    for x0, name, note, mesh in ((60, "A standard transformer", "Every cell hears all 80 others.", False),
                                 (520, "The mesh model", "Every cell hears its 20 neighbors.", True)):
        b.append(text(x0, 150, name, 20, weight="bold") + text(x0, 174, note, 16, INK2))
        b.append(f"<rect x='{x0}' y='{y0}' width='{81 * px}' height='{81 * px}' fill='{SURFACE}' stroke='{GRID}'/>")
        for i in range(81):
            for j in range(81):
                color = relation(i, j) if mesh else ("#b4b3ad" if i != j else None)
                if color:
                    b.append(f"<rect x='{x0 + j * px}' y='{y0 + i * px}' width='{px}' height='{px}' fill='{color}'/>")
    for k, (color, name) in enumerate(((ROW, "Same row"), (COL, "Same column"), (BOX, "Same box"))):
        y = 210 + k * 40
        b.append(f"<rect x='970' y='{y}' width='20' height='20' rx='3' fill='{color}'/>" + text(1002, y + 16, name, 17))
    b.append(text(970, 372, "6,480", 40, INK2, "bold") + text(970, 396, "links on the left", 16, INK2))
    b.append(text(970, 462, "1,620", 40, weight="bold") + text(970, 486, "links on the right", 16, INK2))
    b.append(text(60, 635, "Each dot is one pair of cells that can talk. Rows of the picture are listeners. Columns are speakers.", 16, INK2))
    page("fig2_links.svg", "Who can talk to whom", "The mesh removes three quarters of the links, and keeps the ones the rules use.", b)


def box(x, y, w, h, title, note="", fill="#ffffff"):
    out = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{INK}' stroke-width='1.5'/>"
    out += text(x + w / 2, y + (h / 2 + 6 if not note else h / 2 - 4), title, 18, weight="bold", anchor="middle")
    return out + (text(x + w / 2, y + h / 2 + 18, note, 14, INK2, anchor="middle") if note else "")


def arrow(d):
    return f"<path d='{d}' fill='none' stroke='{INK2}' stroke-width='2' marker-end='url(#arrow)'/>"


def fig_architecture():
    b = []
    b.append(box(50, 290, 140, 80, "Puzzle", "81 cells"))
    b.append(box(230, 290, 150, 80, "Embed", "96 numbers a cell"))
    b.append(arrow("M190,330 L228,330"))
    # The loop container.
    b.append(f"<rect x='430' y='170' width='540' height='330' rx='12' fill='#f3f2ee' stroke='{INK2}' stroke-width='1.5' stroke-dasharray='6 5'/>")
    b.append(text(450, 200, "One block. The same weights on every pass.", 17, weight="bold"))
    b.append(box(455, 290, 120, 80, "Add clues", "back in"))
    b.append(box(605, 265, 200, 130, "", ""))
    b.append(text(705, 292, "Mesh attention", 18, weight="bold", anchor="middle"))
    for k, (color, name) in enumerate(((ROW, "2 heads: row"), (COL, "2 heads: column"), (BOX, "2 heads: box"))):
        y = 310 + k * 26
        b.append(f"<rect x='635' y='{y}' width='14' height='14' rx='3' fill='{color}'/>" + text(658, y + 12, name, 15))
    b.append(box(835, 290, 110, 80, "MLP", "per cell"))
    b.append(arrow("M380,330 L453,330") + arrow("M575,330 L603,330") + arrow("M805,330 L833,330"))
    # The loop back, over the top of the three boxes.
    b.append(arrow("M890,290 L890,232 L515,232 L515,288"))
    b.append(text(702, 224, "32 passes", 16, INK2, anchor="middle"))
    # The clues are fed in again on every pass.
    b.append(arrow("M305,370 L305,450 L515,450 L515,372"))
    b.append(text(420, 472, "the clues, on every pass", 14, INK2, anchor="end"))
    b.append(box(1010, 290, 140, 80, "Readout", "9 scores a cell"))
    b.append(arrow("M945,330 L1008,330"))
    b.append(text(1080, 396, "Scored after", 14, INK2, anchor="middle") + text(1080, 414, "every pass", 14, INK2, anchor="middle"))
    for k, (big, small) in enumerate((("113,673", "weights in the whole model"), ("32", "passes through one block"),
                                      ("20", "neighbors a cell can hear"))):
        x = 60 + k * 390
        b.append(text(x, 600, big, 44, weight="bold") + text(x, 628, small, 16, INK2))
    page("fig3_architecture.svg", "One small block, run again and again",
         "The model has one layer. It runs that layer 32 times, and the board settles a little on each pass.", b)


def chart(xmax, ymax, xticks, yticks, xlabel):
    """Return the axes and two functions that map data to pixels."""
    L, R, T, B = 90, 1130, 150, 580
    fx = lambda v: L + (R - L) * v / xmax
    fy = lambda v: B - (B - T) * v / ymax
    b = []
    for t in yticks:
        b.append(f"<line x1='{L}' y1='{fy(t)}' x2='{R}' y2='{fy(t)}' stroke='{GRID}'/>" + text(L - 12, fy(t) + 5, f"{t}%", 15, INK2, anchor="end"))
    for t in xticks:
        b.append(text(fx(t), B + 26, f"{t:,}", 15, INK2, anchor="middle"))
    b.append(f"<line x1='{L}' y1='{B}' x2='{R}' y2='{B}' stroke='{INK2}'/>" + text((L + R) / 2, B + 62, xlabel, 16, INK2, anchor="middle"))
    return b, fx, fy


def line(points, fx, fy, color):
    d = " ".join(f"{fx(x):.1f},{fy(y):.1f}" for x, y in points)
    return f"<polyline points='{d}' fill='none' stroke='{color}' stroke-width='2.5' stroke-linejoin='round' stroke-linecap='round'/>"


def dot(x, y, color):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{color}' stroke='{SURFACE}' stroke-width='2'/>"


def read_log(name):
    with open(os.path.join(LOGS, name)) as f:
        log = f.read()
    steps = [(int(s), 100 * float(v)) for s, v in re.findall(r"^eval step (\d+) .* solved ([\d.]+)", log, re.M)]
    by_pass = re.search(r"^solved by pass: (.*)$", log, re.M)
    full = re.search(r"solved at pass 32: ([\d.]+), at pass 64: ([\d.]+)", log)
    return steps, [100 * float(v) for v in by_pass.group(1).split()] if by_pass else [], full.groups() if full else ()


def pct(s):
    """Turn a log value such as 0.4135 into 41.4%. The log line by pass has three digits only, and plain float rounding gives 41.3."""
    return f"{(Decimal(s) * 100).quantize(Decimal('0.1'), ROUND_HALF_UP)}%"


def fig_passes(by_pass, full):
    b, fx, fy = chart(64, 50, [1, 8, 16, 24, 32, 40, 48, 56, 64], [0, 10, 20, 30, 40, 50], "Passes through the block")
    pts = list(enumerate(by_pass, 1))
    b.append(f"<line x1='{fx(32)}' y1='{fy(0)}' x2='{fx(32)}' y2='{fy(50)}' stroke='{INK2}' stroke-dasharray='5 5'/>")
    b.append(text(fx(32) - 10, fy(50) + 16, "trained with 32 passes", 15, INK2, anchor="end"))
    b.append(text(fx(32) + 10, fy(50) + 16, "extra passes, never trained", 15, INK2))
    b.append(line(pts, fx, fy, ROW))
    b.append(dot(fx(32), fy(by_pass[31]), ROW) + text(fx(32) - 12, fy(by_pass[31]) - 12, pct(full[0]), 20, weight="bold", anchor="end"))
    b.append(dot(fx(64), fy(by_pass[63]), ROW) + text(fx(64), fy(by_pass[63]) - 14, pct(full[1]), 20, weight="bold", anchor="end"))
    b.append(text(fx(6), fy(8), "Almost nothing is solved", 15, INK2, anchor="middle") + text(fx(6), fy(8) + 19, "before pass 12", 15, INK2, anchor="middle"))
    page("fig4_passes.svg", "The answer settles over the passes",
         "Share of 422,786 test puzzles fully solved after each pass.", b)


def fig_training(attn):
    b, fx, fy = chart(20000, 40, [0, 5000, 10000, 15000, 20000], [0, 10, 20, 30, 40], "Training steps")
    b.append(line(attn, fx, fy, ROW))
    x, y = attn[-1]
    b.append(dot(fx(x), fy(y), ROW) + text(fx(x), fy(y) + 30, f"{y:.1f}%", 20, weight="bold", anchor="end"))
    # The stall, marked with a bracket above the flat stretch.
    b.append(f"<path d='M{fx(7000)},{fy(22)} L{fx(7000)},{fy(23)} L{fx(11000)},{fy(23)} L{fx(11000)},{fy(22)}' fill='none' stroke='{INK2}'/>")
    b.append(text(fx(9000), fy(23) - 10, "Stalls near 18% for 4,000 steps", 15, INK2, anchor="middle"))
    page("fig5_training.svg", "It stalls, then it climbs",
         "Share of 2,000 sample test puzzles solved during training. Checked every 1,000 steps.", b)


if __name__ == "__main__":
    attn_steps, by_pass, full = read_log("mesh-attn.out")
    assert len(by_pass) == 64 and len(attn_steps) == 20, (len(by_pass), len(attn_steps))
    # The numbers written in fig 2 and fig 1 are derived here, so the pictures cannot drift from the rule.
    links = sum(relation(i, j) is not None for i in range(81) for j in range(81))
    assert links == 1620 and all(sum(relation(i, j) is not None for j in range(81)) == 20 for i in range(81))
    fig_mesh()
    fig_mask()
    fig_architecture()
    fig_passes(by_pass, full)
    fig_training(attn_steps)
    print("wrote 5 figures")
