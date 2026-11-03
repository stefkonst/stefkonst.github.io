"""I_signflip: raw lift change against recording time, and the +10 minus -5 deg difference against speed.
Input: paper_tension/fig5_flapping/results/sign_flip_check.csv and lift_table_percent.csv. Run: python sign_flip_figure.py   (../results/I_signflip.png, svg)"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
T = pd.read_csv(r"C:\Users\stefk\Documents\paper_tension\fig5_flapping\results\sign_flip_check.csv")
M = pd.read_csv(os.path.join(HERE, "lift_table_percent.csv"))
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.axisbelow": True})
COL = {2.0: "#77AC30", 4.0: "#7E2F8E", 6.0: "#0072BD"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
a = ax[0]
w = T[(T.U > 0) & (T.hz == "1Hz")]
for U, g in w.groupby("U"):
    g = g.sort_values("start_s")
    a.plot(g.start_s, g.raw, "o-", color=COL[U], ms=7, lw=2, label=f"{U:g} m/s")
    for _, r in g.iterrows():
        a.annotate(f"{int(r.aoa):+d}°", (r.start_s, r.raw), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=COL[U])
c = T[(T.U == 0) & (T.run == "run006")].sort_values("start_s")
a.plot(c.start_s, c.raw, "s:", color="#8A8A8A", ms=5, label="No wind")
a.axhline(0, color="k", lw=0.8)
a.set_xlabel("Time in the run when the block starts, in s")
a.set_ylabel(r"Raw $\Delta L$ at 1.6 kV in mN, no tare")
a.legend(fontsize=9, frameon=False, loc="lower right")
a.grid(alpha=.3)
a.set_title("a) −5° is always first", loc="left")
b = ax[1]
Us = [2.0, 4.0, 6.0]
d, m = [], []
for U in Us:
    g = w[w.U == U]
    d.append(g[g.aoa == 10].raw.iloc[0] - g[g.aoa == -5].raw.iloc[0])
    q = M[M.U == U]
    m.append(q[q.aoa == 10].whole_mN.iloc[0] - q[q.aoa == -5].whole_mN.iloc[0])
b.plot(Us, d, "o-", color="k", ms=8, lw=2, label="Measured, no tare")
b.plot(Us, m, "s--", color="#D95319", ms=7, lw=2, label="Model, whole edge, both wings")
b.set_xticks(Us)
b.set_ylim(0, 130)
b.set_xlabel("Flow speed in m/s")
b.set_ylabel(r"$\Delta L$ at +10° minus at −5°, in mN")
b.legend(fontsize=9, frameon=False, loc="upper left")
b.grid(alpha=.3)
b.set_title("b) Size of the flip", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(RES, "I_signflip.png"), dpi=200)
fig.savefig(os.path.join(RES, "svg", "I_signflip.svg"))
print("diff", [round(x, 1) for x in d], "model", [round(x, 1) for x in m])
