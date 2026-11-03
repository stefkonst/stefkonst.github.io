"""Finite-element reference on the 5 cm panel (fe_reference.py). Settled and unsettled cases are kept apart.
Run: python figures_fe.py -> figures/F_reference.png and .svg"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "fe_reference.jsonl"))]
R = [r for r in R if "V1" in r and r.get("N", 1) == 1 and r.get("U", 4.0) == 4.0 and r.get("active", True) and "h" not in r]
ok = lambda r: bool(r["ok0"] and r["ok1"])
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
A, B = PS.MATLAB[0], PS.MATLAB[1]
for r in R:
    a = r.get("alpha", 6.0)
    m = "o" if r.get("eta", 0.5) == 0.5 else "s"
    if ok(r):
        ax[0].plot(r["V0"]["CL"], r["V1"]["CL"], m, color=A if a == 10 else B, ms=5)
    else:
        ax[0].plot(r["V0"]["CL"], r["V1"]["CL"], m, mfc="none", color="0.65", ms=4)
ax[0].plot([-0.1, 1.4], [-0.1, 1.4], color="0.3", lw=1)
h = [plt.Line2D([], [], marker="o", ls="", color=A, label=r"Settled, $\alpha$ 10°, $\eta$ 0.5"), plt.Line2D([], [], marker="s", ls="", color=A, label=r"Settled, $\alpha$ 10°, $\eta$ 0.25"), plt.Line2D([], [], marker="o", ls="", mfc="none", color="0.65", label="Not settled")]
ax[0].legend(handles=h, loc="lower right")
ax[0].set(xlabel=r"$C_L$ at 0 V", ylabel=r"$C_L$ at 1.6 kV")
S = [r for r in R if ok(r)]
for eta, m, c in ((0.5, "o", A), (0.25, "s", B)):
    q = [r for r in S if r.get("eta", 0.5) == eta]
    ax[1].plot([r["V0"]["F_mN"] for r in q], [(r["V1"]["CL"] - r["V0"]["CL"]) for r in q], m, color=c, label=rf"$\eta$ = {eta}")
ax[1].axhline(0, color="0.3", lw=1)
ax[1].set(xlabel=r"Fibre force at 0 V, $F$ in mN", ylabel=r"Lift change $\Delta C_L$", ylim=(-0.2, 0.2))
ax[1].legend()
for eta, m, c in ((0.5, "o", A), (0.25, "s", B)):
    q = [r for r in S if r.get("eta", 0.5) == eta]
    ax[2].plot([r["V0"]["F_mN"] for r in q], [r["V0"]["zedge_mm"] for r in q], m, mfc="none", color=c, label=rf"0 V, $\eta$ = {eta}")
    ax[2].plot([r["V1"]["F_mN"] for r in q], [r["V1"]["zedge_mm"] for r in q], m, color=c, label=rf"1.6 kV, $\eta$ = {eta}")
ax[2].set(xlabel=r"Fibre force, $F$ in mN", ylabel=r"Trailing edge height $z_{TE}$ in mm")
ax[2].legend(ncol=1)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "F_reference.png"))
A_all = [json.loads(l) for l in open(os.path.join(HERE, "data", "fe_reference.jsonl"))]
s = [r for r in A_all if r.get("ok0") and r.get("ok1")]
d = [r["V1"]["CL"] - r["V0"]["CL"] for r in s]
dL = [(r["V1"]["CL"] - r["V0"]["CL"]) * 0.5 * 1.225 * r.get("U", 4) ** 2 * r["area_cm2"] * 1e-1 for r in s]
print(len(A_all), len(s), "dCL %.3f %.3f" % (min(d), max(d)), "dL mN %.2f %.2f" % (min(dL), max(dL)), "alpha", sorted({r.get("alpha", 6) for r in s}), "L0 mN", np.mean([r["V0"]["CL"] * 0.5 * 1.225 * 16 * r["area_cm2"] * 1e-1 for r in s]), "dz", np.round([r["V1"]["zedge_mm"] - r["V0"]["zedge_mm"] for r in s], 1))
