"""Figures for the POC page. Run: python poc_figures.py -> P_wings.png, P_transmission.png (and .svg). Needs ../flittermouse/data/poc_cases.jsonl"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
PS.apply()
R = [json.loads(l) for l in open(os.path.join(HERE, "..", "flittermouse", "data", "poc_cases.jsonl"))]
Q = 0.5 * 1.225 * 16 * 1e3


def dL(n, a=6.0):
    r = next(x for x in R if x["name"] == n and x["alpha"] == a)
    return (r["V1"]["CL"] - r["V0"]["CL"]) * Q * r["area_cm2"] * 1e-4


A_, B_, C_ = PS.MATLAB[0], PS.MATLAB[1], PS.MATLAB[3]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
a = ax[0]
for k, (lab, cr, ct) in enumerate((("A  Rectangle", 25, 25), ("B  Taper", 35, 15), ("C  Steep taper", 40, 10))):
    y0 = 2 * 52 - k * 52
    a.add_patch(Polygon([(0, y0), (50, y0), (50, y0 - ct), (0, y0 - cr)], fc="#dce7f5", ec="none"))
    a.plot([0, 50], [y0, y0], color="k", lw=2.4)
    a.plot([0, 50], [y0 - cr, y0 - ct], color="0.45", lw=3)
    a.plot([50, 50], [y0, y0 - ct], color="0.3", lw=1.6)
    a.plot([0, 0], [y0, y0 - cr], color="0.3", lw=1.6)
    a.text(25, y0 + 2.5, lab, ha="center", fontsize=9)
a.set(xlim=(-3, 53), ylim=(-45, 116), aspect="equal", title="As tensioned, 50 mm span")
a.axis("off")
a = ax[1]
x = np.arange(3)
names = ["rectangle", "taper", "steep"]
for lab, suf, c, dx in (("4 % stretch", "", A_, -.27), ("4 %, rib", ", rib", B_, -.09), ("6 % stretch", ", stretch 6 %", C_, .09), ("2 % stretch", ", stretch 2 %", "0.65", .27)):
    a.bar(x + dx, [dL(n + suf) for n in names], width=.17, color=c, label=lab)
a.axhline(dL("rectangle, rotation control"), color="0.25", lw=1.1, ls="--")
a.text(2.46, dL("rectangle, rotation control") + .2, "Tilted cut, rotation", ha="right", fontsize=7.5, color="0.25")
a.set_xticks(x); a.set_xticklabels(["A", "B", "C"])
a.set(ylabel=r"Lift change $\Delta L$ in mN", title="1370 V, 6° pitch, 4 m/s", ylim=(0, 14))
a.legend(fontsize=7.5, ncol=2, loc="upper right")
a = ax[2]
for lab, q, c in (("Skin + roll + LE, mg", None, A_),):
    pass
stations = [("A", "rectangle"), ("B", "taper"), ("C", "steep")]
for k, (lab, n) in enumerate(stations):
    r = next(x for x in R if x["name"] == n)
    r2 = next(x for x in R if x["name"] == n + ", roll 15 mm")
    a.bar(k - .17, r["m_skin"] + r["m_fibre"] + r["m_le"], width=.32, color=A_, label="20 mm roll" if k == 0 else None)
    a.bar(k + .17, r2["m_skin"] + r2["m_fibre"] + r2["m_le"], width=.32, color=B_, label="15 mm roll" if k == 0 else None)
a.set_xticks(range(3)); a.set_xticklabels(["A", "B", "C"])
a.set(ylabel="Wing mass in mg", title="3 layers, 30 µm", ylim=(0, 420))
a.legend(fontsize=8, loc="upper right", ncol=2)
PS.finish([ax[1], ax[2]])
fig.tight_layout()
fig.savefig(os.path.join(HERE, "P_wings.png"))
plt.close(fig)

fig, a = plt.subplots(figsize=(8.4, 3.6))
a.plot([0, 26], [0, 0], color="k", lw=3)
a.text(14, 1.2, "Wing (leading edge), to 50 mm", fontsize=8.5)
a.plot(0, 0, "o", color="k", ms=8)
a.text(0.9, -1.9, "Hinge: polyimide\nflexure, chord axis", fontsize=7.5, va="top")
a.plot([0, 0], [-3, 3], color="0.2", lw=2.4)
a.annotate("", (0.75, 3), (0.75, 0), arrowprops=dict(arrowstyle="<->", lw=.7))
a.text(1.0, 1.6, "r = 3 mm", fontsize=8)
for y, c, lab, txt in ((-3, PS.MATLAB[0], "Down core, 350 mg, 20 mm", "Pulls the wing down"), (3, PS.MATLAB[3], "Up core: spare, or carbon flexure", "Pulls the wing up")):
    a.add_patch(Rectangle((-27, y - 2.35), 20, 4.7, fc=c, alpha=.25, ec=c))
    a.plot([-7, 0], [y, y], color="0.2", lw=1)
    a.plot([-27, -29], [y, y], color="0.2", lw=1)
    a.plot([-29, -29], [y - 3, y + 3], color="k", lw=4)
    a.text(-17, y, lab, ha="center", va="center", fontsize=8)
    a.text(-3.5, y + (0.5 if y > 0 else -0.5), "Thread", fontsize=7, ha="center", va="bottom" if y > 0 else "top")
a.text(-29.7, 0, "Body", rotation=90, va="center", ha="right", fontsize=8)
a.set(xlim=(-33, 28), ylim=(-7, 7), aspect="equal", title="Antagonistic pair, seen from the front. Both stretched 10 %, no separate spring")
a.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "P_transmission.png"))
plt.close(fig)
