"""Timing, lag, phase and force-cap results from timing.py. Run: python figures_timing.py -> figures/T_timing.png and .svg"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
load = lambda s: [json.loads(l) for l in open(os.path.join(HERE, "data", f"timing_{s}.jsonl"))]
A, B, E = load("a"), load("b"), load("e")
base = dict(design="two20", tip_fixed=True, zeta=0.05)
sel = lambda R, **k: [r for r in R if all(r.get(q) == v for q, v in {**base, **k}.items())]
fig, ax = plt.subplots(1, 4, figsize=(13.6, 3.4))
modes = (("off", "Always stiff", "0.25", "-"), ("down", "Soft on downstroke", PS.ON, "-"), ("sin", "Sine, shared drive", PS.MATLAB[1], "--"), ("on", "Always soft", PS.MATLAB[3], ":"), ("up", "Soft on upstroke", PS.OFF, "-"))
for m, lab, c, ls in modes:
    al = (4.0, 8.0, 12.0)
    ax[0].plot(al, [sel(A, mode=m, f=25.0, Phi_deg=30.0, alpha_deg=a)[0]["L"] * 1e3 for a in al], "o" + ls, color=c, label=lab)
ax[0].set(xlabel=r"Pitch $\alpha$ in °", ylabel=r"Lift per wing $L$ in mN")
ax[0].legend(fontsize=7)
off = sel(A, mode="off", f=25.0, Phi_deg=30.0, alpha_deg=8.0)[0]["L"] * 1e3
dn = sel(A, mode="down", f=25.0, Phi_deg=30.0, alpha_deg=8.0)[0]["L"] * 1e3
T = sorted(sel(B, mode="down", f=25.0), key=lambda r: r.get("tau", 0))
T = [r for r in T if "tau" in r]
ax[1].semilogx([r["tau"] * 1e3 for r in T], [r["L"] * 1e3 for r in T], "o-", color=PS.ON, label="Soft on downstroke")
ax[1].axhline(off, color="0.25", lw=1)
ax[1].text(1.8, off + 0.03, "Always stiff", fontsize=8)
ax[1].axvspan(10, 20, color="0.88")
ax[1].text(10.5, 8.33, "bench\nforce fall", fontsize=8)
ax[1].set(xlabel=r"Response lag $\tau$ in ms", ylabel=r"Lift per wing $L$ in mN")
P = sorted([r for r in sel(B, mode="sin") if "psi" in r], key=lambda r: r["psi"])
ax[2].plot([np.degrees(r["psi"]) for r in P], [r["L"] * 1e3 for r in P], "o-", color=PS.MATLAB[1])
ax[2].axhline(off, color="0.25", lw=1)
ax[2].set(xlabel=r"Voltage phase $\psi$ from mid downstroke in °", ylabel=r"Lift per wing $L$ in mN", xticks=(0, 90, 180, 270))
g = lambda m, **k: next(r for r in E if r["mode"] == m and all(r.get(q) == v for q, v in k.items()))
cases = [("Coarse", off, dn), ("Fine mesh", g("off", ns=20, nc=8)["L"] * 1e3, g("down", ns=20, nc=8)["L"] * 1e3), ("Free wake", g("off", free_wake=True)["L"] * 1e3, g("down", free_wake=True)["L"] * 1e3),
         (r"$c_n\leq1.6$", g("off", cn_cap=1.6)["L"] * 1e3, g("down", cn_cap=1.6)["L"] * 1e3), (r"$c_n\leq1.2$", g("off", cn_cap=1.2)["L"] * 1e3, g("down", cn_cap=1.2)["L"] * 1e3)]
x = np.arange(len(cases))
ax[3].bar(x - 0.19, [c[1] for c in cases], 0.36, color="0.45", label="Always stiff")
ax[3].bar(x + 0.19, [c[2] for c in cases], 0.36, color=PS.ON, label="Soft on downstroke")
ax[3].set_xticks(x, [c[0] for c in cases], fontsize=7)
ax[3].set(ylabel=r"Lift per wing $L$ in mN")
ax[3].legend(fontsize=7)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "T_timing.png"))
print([(c[0], round(c[1], 2), round(c[2], 2)) for c in cases])
