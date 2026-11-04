"""Wing steps, in the style of the fabrication sketches in the slides: light blue carrier, maroon film, side views, Arial captions.
Run: python poc_steps.py -> steps/P_steps.png/.svg and steps/step_N.png/.svg"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Rectangle, Circle, FancyArrowPatch
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "svg.fonttype": "none", "font.size": 9})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "steps"); os.makedirs(OUT, exist_ok=True)
CAN, EDGE, FILM, FILM2, CF, GREY, BLUE, RED = "#d1e0ff", "#0f3a8c", "#750014", "#c98f99", "#40464c", "#8b959e", "#1966ff", "#ff1423"


def can(a, x, y, w, h=0.55, d=0.5):
    """The carrier, as the 'can' in the slides."""
    a.add_patch(Rectangle((x, y - h), w, h, fc=CAN, ec="none"))
    a.add_patch(Ellipse((x + w / 2, y - h), w, d, fc=CAN, ec=EDGE, lw=0.9, zorder=1))
    a.add_patch(Rectangle((x, y - h), w, h, fc=CAN, ec="none", zorder=1.1))
    a.plot([x, x], [y, y - h], color=EDGE, lw=0.9, zorder=1.2)
    a.plot([x + w, x + w], [y, y - h], color=EDGE, lw=0.9, zorder=1.2)
    a.add_patch(Ellipse((x + w / 2, y), w, d, fc=CAN, ec=EDGE, lw=0.9, zorder=1.3))


def layer(a, x, y, w, t=0.07, c=FILM, z=3):
    a.add_patch(Rectangle((x, y), w, t, fc=c, ec="#3a000a" if c == FILM else "k", lw=0.5, zorder=z))


def arrow(a, p, q, col="k", lw=1.0, ms=9, z=6):
    a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, zorder=z, shrinkA=0, shrinkB=0))


def label(a, x, y, s, ha="center", col="k", fs=7.5):
    a.text(x, y, s, ha=ha, va="center", fontsize=fs, color=col, zorder=7)


def bar(a, x, y, w=0.55, h=0.2, c=CF):
    a.add_patch(Rectangle((x, y), w, h, fc=c, ec="k", lw=0.5, zorder=4))


def wing_xy(xc=35.0, S=50.0, c=27.0, ct=3.0, curved=True, n=60):
    x = np.linspace(0, S, 121)
    t = np.clip((x - xc) / (S - xc), 0, 1)
    yle = (c - ct) * (1 - np.sqrt(1 - t ** 2)) if curved else 0 * x
    return x, yle


def plan(a, x0, y0, k=0.075, curved=True, cut_only=False, c=27.0):
    """Plan view with the LE along the top. Returns a map from wing mm to axes."""
    S = 50.0
    x, yl = wing_xy(c=c) if curved else (np.linspace(0, S, 40), np.zeros(40))
    if not curved:
        c = 25.0
    M = lambda xx, yy: (x0 + k * np.asarray(xx), y0 - k * np.asarray(yy))
    px, py = M(np.r_[x, S, 0], np.r_[yl, c, c])
    a.add_patch(Polygon(np.c_[px, py], fc=FILM2, ec=FILM, lw=1.0, alpha=.45 if not cut_only else .28, zorder=3))
    return M


# ---------- steps
def s1(a):
    can(a, 1.0, 1.8, 8)
    y = 1.86
    for i, (dx, c) in enumerate(((0, FILM), (1.0, FILM2), (0, FILM), (1.4, FILM2), (0, FILM), (1.0, FILM2), (0, FILM), (1.4, FILM2), (0, FILM))):
        if i % 2 == 0:
            layer(a, 1.6, y, 6.8, 0.09)
            y += 0.09
        else:
            xs = 2.1 + (0.5 if i % 4 == 3 else 0)
            layer(a, xs, y, 5.3, 0.05, CF)
            y += 0.05
    label(a, 8.75, 2.75, "½D top", ha="left")
    label(a, 8.75, 2.4, "½D bottom", ha="left")
    a.plot([8.4, 8.72], [2.28, 2.7], color="k", lw=.5); a.plot([8.4, 8.72], [1.92, 2.35], color="k", lw=.5)
    label(a, 5, 3.0, "4 electrodes, alternately A and B", col="k")
    arrow(a, (5, 2.85), (5, 2.4))
    a.text(5, 4.05, "", fontsize=1)


def s2(a):
    oct_ = np.array([[np.cos(t), np.sin(t)] for t in np.radians(22.5 + 45 * np.arange(8))])
    a.add_patch(Polygon(oct_ * [3.9, 2.05] + [5, 2.45], fc=CAN, ec=EDGE, lw=.9, zorder=1))
    M = plan(a, 3.25, 3.95, k=0.07, cut_only=True)
    x, yl = wing_xy()
    px, py = M(np.r_[0, x, 50, 0], np.r_[27, yl, 27, 27])
    a.plot(px, py, color=FILM, lw=1.1, ls="--", zorder=4)
    # roll strip below the fibre line, trim line
    a.add_patch(Rectangle((3.25, 3.95 - 0.07 * 47), 3.5, 0.07 * 20, fc=FILM2, alpha=.28, ec=FILM, lw=.9, ls="--", zorder=3))
    label(a, 4.6, 3.1, "Skin", col=FILM)
    label(a, 5, 1.32, "20 mm strip for the roll", col=FILM, fs=7)
    a.plot([8.0, 7.05], [3.6, 2.75], color="k", lw=2.0, solid_capstyle="round", zorder=6)
    label(a, 8.05, 3.85, "Scalpel\nor laser", fs=7)


def s3(a):
    can(a, 1.0, 1.8, 8)
    layer(a, 1.6, 1.86, 6.8, 0.14)
    # the frame seen along the chord: T section lying on the film, tip piece at the end
    a.add_patch(Rectangle((1.75, 2.0), 4.6, 0.09, fc=CF, ec="k", lw=.4, zorder=4))
    a.add_patch(Rectangle((1.75, 2.09), 4.6, 0.26, fc=CF, ec="k", lw=.4, zorder=4))
    a.add_patch(Rectangle((6.2, 2.0), 2.05, 0.08, fc=GREY, ec="k", lw=.4, zorder=4))
    for x in (2.6, 4.6, 7.2):
        arrow(a, (x, 3.05), (x, 2.5 if x < 6 else 2.2))
    label(a, 5, 3.3, "Press and cure with primer")
    label(a, 3.9, 1.0 + 1.67 - 0.0, "", fs=1)
    label(a, 3.6, 2.66, "T section", fs=7)
    label(a, 8.2, 2.6, "Tip piece", fs=7)
    label(a, 1.3, 2.45, "No\nbar", fs=7, col=RED)
    label(a, 8.7, 1.98, "", fs=1)


def s4(a):
    can(a, 1.0, 1.8, 8)
    layer(a, 1.6, 1.86, 3.95, 0.14)
    th = np.linspace(0, 5.6 * np.pi, 300)
    r = 0.06 + 0.052 * th / (2 * np.pi) * 1.15
    cx, cy = 6.0, 2.42
    a.plot(cx + r * np.cos(th), cy + r * np.sin(th), color=FILM, lw=1.3, zorder=4)
    a.plot([5.5, 5.92], [1.93, 2.1], color=FILM, lw=1.4, zorder=4)
    a.add_patch(Circle((cx, cy), 0.03, color="k", zorder=5))
    bar(a, 5.0, 2.0, 0.17, 0.95, CF)
    arrow(a, (7.5, 2.42), (6.75, 2.42))
    label(a, 5.1, 3.2, "Fence on the\nfibre line", fs=7)
    label(a, 8.55, 2.42, "Roll this way", fs=7)
    label(a, 6.0, 1.45, "", fs=1)
    a.annotate("Thread as axis", (cx, cy), (7.4, 3.2), fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", lw=.5), zorder=7)
    label(a, 2.6, 2.25, "Skin", col=FILM, fs=7)
    label(a, 5, 3.75, "Then glue the roll's tip end to the tip piece", fs=7.5)


def s5(a):
    can(a, 1.0, 1.8, 8)
    x = np.linspace(2.0, 8.2, 80)
    y = 1.9 + (np.maximum(x - 4.2, 0) / 4.0) ** 1.4 * 1.15
    a.plot(x, y, color=FILM, lw=2.0, zorder=4, solid_capstyle="butt")
    a.add_patch(Rectangle((7.75, y[-1] - 0.1), 0.5, 0.42, fc=CF, ec="k", lw=.5, zorder=5))
    arrow(a, (8.0, y[-1] + 0.34), (8.6, y[-1] + 1.05))
    a.add_patch(Polygon([[4.2, 2.16], [4.08, 1.98], [4.32, 1.98]], fc=BLUE, zorder=5))
    a.add_patch(Circle((4.2, 1.98), 0.12, color=BLUE, zorder=5))
    label(a, 3.2, 2.75, "A little IPA\nat the peel front", fs=7)
    a.plot([3.6, 4.1], [2.45, 2.1], color="k", lw=.5)
    label(a, 5.5, 3.65, "Lift by the leading edge: slow, shallow", fs=7.5)
    label(a, 9.1, 2.45, "T section,\nend view", fs=7)


def s6(a):
    M = plan(a, 3.0, 3.95, k=0.075)
    x, yl = wing_xy()
    m = x <= 35
    px, py = M(x[m], yl[m])
    a.plot(px, py, color="k", lw=3.0, zorder=5, solid_capstyle="butt")
    m = x >= 34
    px, py = M(x[m], yl[m])
    a.plot(px, py, color=CF, lw=1.7, zorder=5)
    px, py = M([0, 50], [27, 27])
    a.plot(px, py, color=GREY, lw=3.4, zorder=4, solid_capstyle="butt")
    ax_, ay_ = M(0, 27)
    a.plot([1.1, ax_], [ay_, ay_], color="k", lw=.9, zorder=5)
    a.add_patch(Rectangle((0.95, ay_ - 0.33), 0.12, 0.66, fc="k", zorder=5))
    arrow(a, (2.55, ay_ - 0.3), (1.75, ay_ - 0.3))
    label(a, 2.15, ay_ - 0.52, "Pull 4–5 mm", fs=7.5)
    label(a, 1.5, ay_ + 0.22, "Thread", fs=7)
    label(a, 0.95, ay_ - 0.85, "Stop", fs=7)
    label(a, 5, 2.9, "Skin tensions", col=FILM, fs=7.5)
    label(a, 5, 1.55, "Check the pull: about 50 mN", fs=7.5)
    label(a, 5, 1.2, "8 mm of pull loses almost all the authority", fs=7, col=RED)
    label(a, 1.95, 3.45, "Root edge\nstays free", fs=7, col=RED)


def s7(a):
    a.add_patch(Rectangle((4.3, 2.42), 1.4, 0.12, fc=CF, ec="k", lw=.4, zorder=3))
    a.add_patch(Rectangle((4.94, 2.54), 0.12, 0.72, fc=CF, ec="k", lw=.4, zorder=3))
    layer(a, 3.8, 2.33, 5.2, 0.09)
    a.add_patch(Rectangle((4.3, 2.23), 1.4, 0.09, fc=CF, ec="k", lw=.4, zorder=4))
    a.add_patch(Polygon([[5.7, 2.32], [6.05, 2.32], [5.7, 2.22]], fc=GREY, ec="k", lw=.3, zorder=4))
    arrow(a, (5.0, 1.3), (5.0, 2.12))
    label(a, 5.0, 1.12, "Cap strip from below, press and cure")
    a.annotate("Soft silicone fillet", (5.9, 2.28), (7.8, 1.75), fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", lw=.5), zorder=7)
    label(a, 7.6, 2.62, "Film", col=FILM, fs=7)
    label(a, 3.3, 2.95, "Already bonded in step 3", fs=7)
    a.plot([3.9, 4.85], [2.82, 2.6], color="k", lw=.5)


def s8(a):
    M = plan(a, 2.9, 3.6, k=0.085)
    x, yl = wing_xy()
    m = x <= 35
    px, py = M(x[m], yl[m])
    a.plot(px, py, color="k", lw=3.0, zorder=5, solid_capstyle="butt")
    m = x >= 34
    px, py = M(x[m], yl[m])
    a.plot(px, py, color=CF, lw=1.7, zorder=5)
    px, py = M([0, 50], [27, 27])
    a.plot(px, py, color=GREY, lw=3.6, zorder=4, solid_capstyle="butt")
    ax_, ay_ = M(0, 27)
    a.plot([1.2, ax_], [ay_, ay_], color="k", lw=.9, zorder=5)
    label(a, 1.5, ay_ + 0.2, "Thread to body", fs=7)
    px, py = M([35, 45.6], [0, 27])
    a.plot(px, py, color=CF, lw=1.3, ls="--", zorder=5)
    label(a, 4.4, 3.9, "Straight T section", fs=7)
    label(a, 8.2, 3.52, "Curved tip piece,\nfrom carbon plate", fs=7, ha="center")
    label(a, 2.1, 2.75, "Free\nroot edge", fs=7, col=RED)
    label(a, 5, 0.92, "Rolled trailing edge, about 2 mm across", fs=7)
    a.annotate("Rib, 0.5 × 0.2 mm", tuple(M(38, 12)), (8.0, 2.05), fontsize=7, ha="center", arrowprops=dict(arrowstyle="-", lw=.5), zorder=7)
    label(a, 4.6, 2.4, "Active skin", col=FILM, fs=7.5)


def s9(a):
    a.add_patch(Rectangle((1.3, 1.85), 2.5, 0.22, fc=CF, ec="k", lw=.5, zorder=3))
    a.add_patch(Rectangle((2.32, 2.52), 0.46, 0, fc=CF, ec="k", lw=.5, zorder=3))
    a.add_patch(Rectangle((1.3, 2.32), 2.5, 0.42, fc=CF, ec="k", lw=.5, zorder=3))
    a.add_patch(Rectangle((2.32, 2.74), 0.46, 0.0, fc=CF, ec="k", lw=.5, zorder=3))
    a.add_patch(Rectangle((1.0, 2.07), 7.9, 0.25, fc=FILM2, ec=FILM, lw=.6, zorder=2))
    a.add_patch(Rectangle((5.0, 2.16), 3.9, 0.06, fc="k", zorder=4))
    a.add_patch(Polygon([[3.8, 2.32], [4.5, 2.32], [3.8, 2.5]], fc=GREY, ec="k", lw=.3, zorder=4))
    a.add_patch(Polygon([[3.8, 2.07], [4.5, 2.07], [3.8, 1.88]], fc=GREY, ec="k", lw=.3, zorder=4))
    a.annotate("", (1.3, 1.62), (3.8, 1.62), arrowprops=dict(arrowstyle="<->", lw=.7))
    label(a, 2.55, 1.42, "1.5 mm", fs=7)
    a.annotate("", (1.3, 3.75), (5.0, 3.75), arrowprops=dict(arrowstyle="<->", lw=.7))
    label(a, 3.15, 3.93, "2 mm: no electrode", fs=7)
    a.plot([5.0, 5.0], [2.16, 3.75], color="k", lw=.5, ls=":")
    a.plot([1.3, 1.3], [2.74, 3.75], color="k", lw=.5, ls=":")
    a.add_patch(Rectangle((2.42, 2.74), 0.26, 0.56, fc=CF, ec="k", lw=.5, zorder=3))
    label(a, 7.0, 2.68, "Electrodes start here", fs=7)
    label(a, 6.9, 1.75, "Film overhangs the rear edge,\nso no carbon meets high voltage", fs=7)
    label(a, 6.0, 0.95, "Sand and seal the strip edges", fs=7)
    a.plot([3.9, 4.85], [1.95, 1.2], color="k", lw=.5)
    label(a, 8.3, 3.7, "Not to scale", fs=6.5, col=GREY)


STEPS = [(s1, "1. Stack and cure on the acrylic"), (s2, "2. Cut the outline, film stays on"), (s3, "3. Bond the leading edge frame, film stays on"),
         (s4, "4. Roll the trailing edge, still on the acrylic"), (s5, "5. Release, by the frame"), (s6, "6. Pull the roll end against a stop"),
         (s7, "7. Add the cap strip"), (s8, "8. Finished wing, from above"), (s9, "9. Leading edge section")]


def finish(a, t):
    a.set(xlim=(0, 10), ylim=(0.2, 4.4), aspect="equal")
    a.axis("off")
    a.text(5, 0.42, t, ha="center", va="center", fontsize=9.5, color="k", zorder=8, bbox=dict(fc="white", ec="none", pad=1.5))


fig, ax = plt.subplots(3, 3, figsize=(12, 5.5 * 3 * 0.5))
for a, (f, t) in zip(ax.ravel(), STEPS):
    f(a); finish(a, t)
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.02, hspace=0.02)
fig.savefig(os.path.join(OUT, "P_steps.png"), dpi=200)
fig.savefig(os.path.join(OUT, "P_steps.svg"))
plt.close(fig)
for i, (f, t) in enumerate(STEPS, 1):
    fig, a = plt.subplots(figsize=(5, 2.1))
    f(a); finish(a, t)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(os.path.join(OUT, f"step_{i}.png"), dpi=200)
    fig.savefig(os.path.join(OUT, f"step_{i}.svg"))
    plt.close(fig)
