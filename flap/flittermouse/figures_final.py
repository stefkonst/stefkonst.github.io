"""Concept schematic, stall-limit study, feathering and size closure.
Run: python figures_final.py -> figures/S_concept.png, G_stall.png, C_closure.png (and .svg), data/closure_final.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
PS.apply()
FIG = os.path.join(HERE, "figures")
load = lambda s: [json.loads(l) for l in open(os.path.join(HERE, "data", f"timing_{s}.jsonl"))]
A_, B_, C_ = PS.MATLAB[0], PS.MATLAB[1], PS.MATLAB[3]

# ---------- schematic
fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.7))
a = ax[0]
a.add_patch(Polygon([(0, 0), (50, 0), (50, 25), (0, 25)], fc="#dce7f5", ec="none"))
a.plot([0, 50], [0, 0], color="k", lw=3.2)
a.plot([0, 8, 25, 42, 50], [25, 23.6, 22.7, 23.6, 25], color=PS.ON, lw=3.2)
a.plot([50, 50], [0, 25], color="0.3", lw=2)
a.plot([0, 0], [0, 25], color="0.3", lw=2)
a.text(25, -3.5, "Carbon leading edge, cross or tube", ha="center", fontsize=9)
a.text(25, 27.5, "Active rolled fibre, 20 mm strip", ha="center", fontsize=9, color=PS.ON)
a.text(25, 12, "Active skin, 2 layers", ha="center", fontsize=9)
a.text(52, 12, "Bonded\nbatten", fontsize=8.5, va="center")
a.text(-2, 12, "Root plate", fontsize=8.5, rotation=90, va="center", ha="right")
a.annotate("", (-2, 31), (-2, -6), arrowprops=dict(arrowstyle="->", lw=1, color="0.3"))
a.text(0, -8, "Stroke axis; flow runs up the page", fontsize=8.5, color="0.3")
a.set(xlim=(-10, 68), ylim=(-11, 33), aspect="equal", title="Wing from above, 50 × 25 mm")
a.axis("off")
a = ax[1]
a.plot([0, 20], [0, 0], color="k", lw=0)
a.add_patch(Rectangle((-14, -18), 8, 20, fc=PS.MATLAB[0], alpha=0.25, ec=PS.MATLAB[0]))
a.text(-10, -8, "Core\nDEA\n2 cm", ha="center", va="center", fontsize=8.5)
for ang, ls, col in ((30, "-", "k"), (-30, "--", "0.55")):
    t = np.radians(ang)
    a.plot([0, 50 * np.cos(t)], [0, 50 * np.sin(t)], ls, color=col, lw=2.6)
    a.plot([-6 * np.sin(t) * 0 - 6, 0], [2, 0], color="0.3", lw=1.5)
a.plot(0, 0, "o", color="k", ms=7)
a.text(-3, 6, "Hinge", fontsize=8.5, ha="right")
a.plot([-6, -6], [2, 2], "o", color="0.3")
a.annotate("Tension lifts the wing", (-6, 3.5), (-30, 26), fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8))
sx = np.linspace(4, 24, 60)
a.plot(sx, -22 + 2 * np.sin(sx * 1.6), color=PS.MATLAB[1], lw=1.5)
a.plot([4, 4], [-22, -6], color="0.3", lw=1)
a.text(14, -28, "Bias spring pulls the wing down\n(most of the stiffness at 25 Hz)", ha="center", va="top", fontsize=8.5)
a.text(46, 28, "Up, stiff", fontsize=8.5)
a.text(40, -32, "Down, soft", fontsize=8.5, color="0.4")
a.set(xlim=(-34, 62), ylim=(-44, 34), aspect="equal", title="Drive, seen from the front")
a.axis("off")
a = ax[2]
t = np.linspace(0, 2, 400)
a.plot(t, np.cos(2 * np.pi * t), color="k", label=r"Stroke $\phi$")
a.plot(t, 0.5 * (1 + np.sin(2 * np.pi * t)), color=PS.ON, label=r"Voltage term $v$, core and skin")
a.axvspan(0, 0.5, color="0.92")
a.axvspan(1, 1.5, color="0.92")
a.text(0.25, -1.26, "Down, soft", ha="center", fontsize=8.5)
a.text(0.75, -1.26, "Up, stiff", ha="center", fontsize=8.5)
a.set(xlabel=r"Time in strokes, $t/T$", ylabel="Normalised", ylim=(-1.4, 1.5), title="One channel per wing")
a.legend(loc="upper right", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "S_concept.png"))
plt.close(fig)

# ---------- stall limit
H = load("h")
kin = [(4.0, 25.0, 20.0, 16.0), (4.0, 20.0, 15.0, 8.0), (4.0, 25.0, 30.0, 8.0), (3.0, 25.0, 15.0, 16.0), (5.0, 25.0, 20.0, 12.0), (4.0, 15.0, 20.0, 12.0)]
g = lambda cap, k, m, d: next(r for r in H if r["cn_cap"] == cap and (r["U"], r["f"], r["Phi_deg"], r["alpha_deg"]) == k and r["mode"] == m and r["cn_cap_soft"] == d)
D = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8)
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
for i, cap in enumerate((1.2, 1.6)):
    for j, k in enumerate(kin):
        o = g(cap, k, "off", 0.0)["L"] * 1e3
        ax[i].plot(D, [g(cap, k, "down", d)["L"] * 1e3 - o for d in D], "o-", color=PS.MATLAB[j], ms=3, label=f"{k[0]:.0f} m/s, {k[1]:.0f} Hz, ±{k[2]:.0f}°, {k[3]:.0f}°")
    ax[i].axhline(0, color="0.3", lw=1)
    ax[i].axvspan(0.1, 0.2, color="0.88")
    ax[i].set(xlabel=r"Rise of the stall limit when soft, $\Delta c_{n,max}$", ylabel=r"Timing gain $\Delta L$ in mN per wing", title=rf"$c_{{n,max}}$ = {cap} when stiff")
ax[0].legend(fontsize=6.5)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "G_stall.png"))
plt.close(fig)

# ---------- closure: twist and size
I, Gs = load("i"), load("g")


def best(R, e, s, m=None):
    b = 0.0
    for r in R:
        if r["mode"] != "down":
            continue
        T = s * r["T"] + (1 - s) * r["T_ns"]
        if T < 0:
            continue
        mw = (m or r["mass_mg"]) * 1e-6
        mc = max(r["P"], 1e-4) / (e * r["f"] * 0.8)
        b = max(b, 2 * r["L"] / ((2 * (mc + mw) + 0.2e-3) * 9.81))
    return b


out = dict(feather={}, size={})
K = (0.0, 0.3, 0.6, 0.9)
fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.5))
for s, c in ((1.0, A_), (0.5, B_), (0.0, C_)):
    for cap, ls in ((None, "-"), (1.6, "--")):
        y = [best([r for r in I if r["feather"] == k and r.get("cn_cap") == cap], 1.25, s) for k in K]
        out["feather"][f"s{s}|{cap}"] = y
        ax[0].plot(K, y, "o" + ls, color=c, ms=4, label=f"Suction {s:g}" if cap is None else None)
ax[0].axhline(1, color=PS.ON, lw=1.2)
ax[0].set(xlabel="Ideal twist, share of the flap angle removed", ylabel=r"Best $L/W$, $e$ = 1.25 J/kg", ylim=(0, 1.2))
ax[0].legend(fontsize=7.5, loc="lower right")
sizes = [(0.05, 0.025), (0.05, 0.035), (0.07, 0.025), (0.07, 0.035), (0.10, 0.04)]
labs = ["5×2.5", "5×3.5", "7×2.5", "7×3.5", "10×4"]
Cm = load("c")
for e, c, nm in ((0.8, A_, "0.8 J/kg"), (1.25, B_, "1.25 J/kg"), (2.1, C_, "2.1 J/kg")):
    y, yc = [], []
    for b_, c_ in sizes:
        if b_ == 0.05 and c_ == 0.025:
            R0, Rc = Cm, [r for r in load("f") if r["cn_cap"] == 1.6]
        else:
            R0 = [r for r in Gs if r["b"] == b_ and r["c"] == c_ and r.get("cn_cap") is None]
            Rc = [r for r in Gs if r["b"] == b_ and r["c"] == c_ and r.get("cn_cap") == 1.6]
        y.append(best(R0, e, 1.0))
        yc.append(best(Rc, e, 1.0))
    out["size"][nm] = dict(attached=y, cap16=yc)
    ax[1].plot(labs, y, "o-", color=c, ms=4, label=nm)
    ax[1].plot(labs, yc, "o--", color=c, ms=4)
ax[1].axhline(1, color=PS.ON, lw=1.2)
ax[1].set(xlabel="Wing size in cm", ylabel=r"Best $L/W$, full suction", ylim=(0, 1.3))
ax[1].legend(fontsize=7.5)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "C_closure.png"))
plt.close(fig)
json.dump(out, open(os.path.join(HERE, "data", "closure_final.json"), "w"), indent=1)
print(json.dumps(out)[:1200])
