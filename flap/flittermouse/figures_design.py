"""Tension design space from the finite elements (fe_reference.py design -> data/fe_design.jsonl), one wing, 4 m/s, 1.6 kV, eta 0.5.
Run: python figures_design.py -> figures/D_tension.png and .svg, data/fe_design_summary.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "fe_design.jsonl"))]
ok = lambda r: bool(r["ok0"] and r["ok1"])
dCL = lambda r: r["V1"]["CL"] - r["V0"]["CL"]
dL = lambda r: dCL(r) * 0.5 * 1.225 * 16 * r["area_cm2"] * 1e-1
RHO = 1070.0


def mass(r):
    t = (r["N"] + 1) * 35e-6
    return RHO * t * (0.05 * 0.025 + r["ws"] * 0.05) * 1e6


fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.5))
C = {1: PS.MATLAB[0], 2: PS.MATLAB[1]}
M = {0.010: "o", 0.020: "s", 0.030: "^"}
for r in R:
    if r["alpha"] != 6.0 or r["lam_n"] != 1.06:
        continue
    ax[0].plot(r["theta"] + (r["ws"] - 0.02) * 15, dCL(r), M[r["ws"]], color=C[r["N"]], mfc=C[r["N"]] if ok(r) else "none")
ax[0].set(xlabel=r"Cut tilt $\theta$ in °", ylabel=r"Lift change $\Delta C_L$, 6°")
h = [plt.Line2D([], [], color=C[n], lw=2, label=f"{n} layer" + ("s" if n > 1 else "")) for n in C] + [plt.Line2D([], [], marker=M[w], ls="", color="0.3", label=f"Roll {w*1e3:.0f} mm") for w in M] + [plt.Line2D([], [], marker="o", ls="", mfc="none", color="0.3", label="Not settled")]
ax[0].legend(handles=h, fontsize=7)
for r in R:
    if ok(r):
        ax[1].plot(r["ns"]["N1"], dCL(r), M[r["ws"]], color=C[r["N"]], ms=4, alpha=0.8)
ax[1].axhline(0, color="0.3", lw=1)
ax[1].set(xlabel=r"Span tension at 0 V, no flow, $N_s$ in N/m", ylabel=r"Lift change $\Delta C_L$", xlim=(0.5, 6.5))
for r in R:
    if ok(r) and r["theta"] <= 8:
        ax[2].plot(mass(r), dL(r), M[r["ws"]], color=C[r["N"]], ms=4, alpha=0.8)
ax[2].set(xlabel=r"Skin + roll mass in mg", ylabel=r"Lift change $\Delta L$ in mN")
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "D_tension.png"))

S = [r for r in R if ok(r)]
best = sorted(S, key=lambda r: -dL(r))[:6]
keep = ("N", "ws", "theta", "lam_n", "alpha")
json.dump(dict(n=len(R), settled=len(S), best=[dict({k: r[k] for k in keep}, CL0=r["V0"]["CL"], CL1=r["V1"]["CL"], dL=dL(r), F0=r["V0"]["F_mN"], F1=r["V1"]["F_mN"], z0=r["V0"]["zedge_mm"], z1=r["V1"]["zedge_mm"], mass=mass(r)) for r in best]),
          open(os.path.join(HERE, "data", "fe_design_summary.json"), "w"), indent=1)
for r in best:
    print({k: r[k] for k in keep}, "CL %.2f %.2f dL %.2f mass %.0f" % (r["V0"]["CL"], r["V1"]["CL"], dL(r), mass(r)))
