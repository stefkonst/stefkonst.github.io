"""Wing shape and reinforcement. Run: python figures_shape.py -> figures/W_shape.png and .svg (needs data/wing_shape.jsonl)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
PS.apply()
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "wing_shape.jsonl"))]
FIG = os.path.join(HERE, "figures")
Q = 0.5 * 1.225 * 16 * 1e3


def dL(name, a):
    r = next(x for x in R if x["name"] == name and x["alpha"] == a)
    return (r["V1"]["CL"] - r["V0"]["CL"]) * Q * r["area_cm2"] * 1e-4 if r.get("ok0") and r.get("ok1") else None


# label, case name, root chord, tip chord, veins in (s, c) fractions
C = [("Taper 35–15", "taper 35-15, more stretch, 1370 V", 35, 15, ()),
     ("Taper 40–10", "taper 40-10, more stretch, 1370 V", 40, 10, ()),
     ("Taper 30–20", "taper 30-20, 1370 V", 30, 20, ()),
     ("Taper 45–5", "taper 45-5, more stretch, 1370 V", 45, 5, ()),
     ("One rib", "one chord rib at mid span (digit), 1370 V", 25, 25, (((.5, 0), (.5, 1)),)),
     ("Plain", "tensioned, tip batten, 1370 V (same field as 1.6 kV at 35 um)", 25, 25, ()),
     ("Diagonal", "one diagonal from the root, tip batten, 1370 V", 25, 25, (((0, 0), (.6, 1)),)),
     ("Half rib", "short rib, half chord, mid span, 1370 V", 25, 25, (((.5, 0), (.5, .5)),)),
     ("Fan of 2", "fan of 2 veins from the root (insect), 1370 V", 25, 25, (((0, 0), (1 / 3, 1)), ((0, 0), (2 / 3, 1)))),
     ("Two ribs", "two chord ribs, 1370 V", 25, 25, (((1 / 3, 0), (1 / 3, 1)), ((2 / 3, 0), (2 / 3, 1)))),
     ("Fan of 3", "fan of 3 veins from the root (insect), 1370 V", 25, 25, (((0, 0), (.25, 1)), ((0, 0), (.5, 1)), ((0, 0), (.75, 1))))]
fig = plt.figure(figsize=(11.2, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, .9, 1.2], wspace=0.08)
ax = fig.add_subplot(gs[0, 1])
sk = fig.add_subplot(gs[0, 0])
y = np.arange(len(C))[::-1]
for k, (lab, name, cr, ct, ve) in enumerate(C):
    yb = y[k] * 58
    sk.add_patch(Polygon([(0, yb), (50, yb), (50, yb - ct), (0, yb - cr)], fc="#dce7f5", ec="none"))
    sk.plot([0, 50], [yb, yb], color="k", lw=2)
    sk.plot([0, 50], [yb - cr, yb - ct], color=PS.ON, lw=2)
    for (s1, c1), (s2, c2) in ve:
        ch = lambda s: cr + (ct - cr) * s
        sk.plot([s1 * 50, s2 * 50], [yb - c1 * ch(s1), yb - c2 * ch(s2)], color="0.2", lw=1.2)
    sk.text(-4, yb - 12, lab, ha="right", va="center", fontsize=8.5)
sk.set(xlim=(-48, 54), ylim=(-50, 58 * len(C) - 40), title="Same span and area")
sk.axis("off")
a6, a10 = [dL(c[1], 6.0) for c in C], [dL(c[1], 10.0) for c in C]
ax.barh(y + 0.19, a6, height=0.36, color=PS.MATLAB[0], label="6° pitch")
ax.barh(y - 0.19, a10, height=0.36, color=PS.MATLAB[1], label="10° pitch")
ax.set(yticks=[], xlabel=r"Lift change with voltage, $\Delta L$ in mN", title="Authority at 1370 V", ylim=(-0.6, len(C) - 0.4))
ax.legend(fontsize=8, loc="lower right")
b = fig.add_subplot(gs[0, 2])
Ih = [(next(x for x in R if x["name"] == c[1] and x["alpha"] == 6.0)) for c in C]
CL0 = [r["V0"]["CL"] for r in Ih]
CL1 = [r["V1"]["CL"] for r in Ih]
for k, yy in enumerate(y):
    b.plot([CL0[k], CL1[k]], [yy, yy], color="0.6", lw=2, zorder=1)
b.plot(CL0, y, "o", color=PS.OFF, label="0 V", zorder=2)
b.plot(CL1, y, "o", color=PS.ON, label="1370 V", zorder=2)
b.set(yticks=[], xlabel=r"$C_L$ at 6° pitch", title="Lift coefficient", ylim=(-0.6, len(C) - 0.4))
b.legend(fontsize=8, loc="lower right")
PS.finish([ax, b])
fig.subplots_adjust(left=0.02, right=0.985, top=0.9, bottom=0.14)
fig.savefig(os.path.join(FIG, "W_shape.png"))
fig.savefig(os.path.join(FIG, "W_shape.svg"))
plt.close(fig)
