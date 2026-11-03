"""I_tare: how much of the old cycle-mean result comes from the wind-off tare, and the estimates side by side at 4 m/s, 10 deg.
Input: paper_tension/fig5_flapping/results/cycle_mean_check.csv and contrast_attached.csv. Also rewrites the labels of I_all_percent through intuition_figures.py.
Run: python tare_figure.py   (../results/I_tare.png, svg)"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PT = r"C:\Users\stefk\Documents\paper_tension\fig5_flapping\results"
O = pd.read_csv(os.path.join(PT, "cycle_mean_check.csv"))
A = pd.read_csv(os.path.join(PT, "contrast_attached.csv"))
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.axisbelow": True})
GRAY, RED, BLACK, BLUE = "#8A8A8A", "#D95319", "#222222", "#0072BD"

w = O[(O.U > 0) & (O.aoa.isin([-5, 0, 10]))].sort_values(["U", "aoa"])
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
a = ax[0]
x = np.arange(len(w))
a.bar(x - 0.27, w.dL_noTare, 0.27, color=GRAY, label="Raw: no tare")
a.bar(x, w.dL - w.dL_noTare, 0.27, color=RED, label="Tare added")
a.bar(x + 0.27, w.dL, 0.27, color=BLACK, label="Old result: raw + tare")
a.axhline(0, color="k", lw=0.8)
a.set_xticks(x); a.set_xticklabels([f"{u:g} m/s\n{al:+d}°" for u, al in zip(w.U, w.aoa)], fontsize=9)
a.set_ylabel(r"$\Delta L$ at 1.6 kV in mN")
a.legend(fontsize=9, frameon=False, ncol=3, loc="upper left")
a.set_ylim(-110, 125)
a.grid(axis="y", alpha=.3)
a.set_title("a) Old cycle mean", loc="left")

r = O[(O.U == 4) & (O.aoa == 10)].iloc[0]
c = A[(A.U == 4) & (A.aoa == 10)].iloc[0]
b = ax[1]
rows = [("Cycle mean, tare", r.dL, 2 * 2.0, BLACK), ("Cycle mean, no tare", r.dL_noTare, 2 * 2.0, GRAY), ("Cycle mean, time term", r.a_B, 2 * r.a_B_se, BLACK), ("On minus off", c.a, 2 * c.se, BLACK), ("Placebo", c.placebo, 2 * c.placebo_se, GRAY)]
y = np.arange(len(rows))[::-1]
b.axvline(0, color="k", lw=0.8)
for yy, (n, v, e, col) in zip(y, rows):
    b.errorbar([v], [yy], xerr=[e], fmt="o", color=col, capsize=4, ms=7)
b.axvline(32, color=RED, ls="--", lw=1.5)
b.text(33, 3.72, "Model", color=RED, fontsize=9)
b.set_yticks(y); b.set_yticklabels([q[0] for q in rows])
b.set_ylim(-0.6, 4.6)
b.set_xlabel(r"$\Delta L$ at 1.6 kV in mN, ± 2 s.e.")
b.grid(axis="x", alpha=.3)
b.set_title("b) 4 m/s, 10°", loc="left")
fig.tight_layout()
fig.savefig(os.path.join(RES, "I_tare.png"), dpi=200)
fig.savefig(os.path.join(RES, "svg", "I_tare.svg"))
