"""Two figures for intuition. I_contrast: why the within-cycle contrast rejects drift, on the 4 m/s, 10 deg data. I_all_percent: every lift result in %.
Run after recalibrate_contrast.py and lift_table_percent.py:  python intuition_figures.py   (PNG in ../results, SVG in ../results/svg)"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
CODE = r"C:\Users\stefk\Documents\paper_tension\fig5_flapping\code\recalibrate_contrast.py"
g_ = {"__file__": CODE}
exec(open(CODE, encoding="utf-8").read().split("R = cycles()")[0], g_)
R = g_["cycles"]()
ph, ON, OFF = g_["ph"], g_["ON"], g_["OFF"]
BV, BS = g_["BENCH_V"], g_["BENCH_S"]
BLUE, RED, GRAY, BLACK, GREEN = "#0072BD", "#D95319", "#8A8A8A", "#222222", "#D9EAD3"
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.axisbelow": True})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RES, name + ".png"), dpi=200)
    fig.savefig(os.path.join(RES, "svg", name + ".svg"))
    plt.close(fig)


# ---------------------------------------------------- why it works
fig, ax = plt.subplots(1, 4, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[1, 1, 1.25, 1.25]))
x = np.linspace(0, 1, 200)
base = 40 * np.sin(2 * np.pi * (x + 0.25))
for a, y, tit, txt in ((ax[0], base + 18, "a) Drift, schematic", "Both phases move:\ncontrast unchanged"), (ax[1], base + 18 * (x < 0.5), "b) DEA, schematic", "Only the on phase moves:\ncontrast changes")):
    a.axvspan(0, 0.5, color=GREEN, lw=0)
    a.plot(x, base, color=GRAY, lw=2, label="Before")
    a.plot(x, y, color=RED, lw=2, label="After")
    a.set_title(tit, loc="left", fontsize=11)
    a.set_xlabel("Flap cycle")
    a.set_xticks([0, 0.5, 1]); a.set_yticks([])
    a.text(0.5, -0.33, txt, transform=a.transAxes, ha="center", va="top", fontsize=10)
    a.set_ylim(-55, 75)
ax[0].legend(fontsize=9, loc="upper right", frameon=False)
ax[0].text(0.25, -50, "DEA on", ha="center", fontsize=9, color="#3B7A2A")

g = R[(R.run == "run008") & (R.U == 4) & (R.aoa == 10)]
lo = np.vstack(g[g.V <= 800].P.values); hi = np.vstack(g[g.V == 1600].P.values)
a = ax[2]
a.axvspan(0, 0.5, color=GREEN, lw=0)
for P, c, l in ((lo, GRAY, "0.4 and 0.8 kV"), (hi, RED, "1.6 kV")):
    k = np.ones(5) / 5
    m = np.convolve(P.mean(0), k, mode="same")
    a.plot(ph[3:-3], m[3:-3], color=c, lw=2, label=l)
a.set_xlabel("Flap cycle (0: DEA on)")
a.set_ylabel("Lift in mN")
a.legend(fontsize=9, loc="best", frameon=False)
a.set_title("c) Real data, 4 m/s, 10°", loc="left", fontsize=11)
a.grid(alpha=.3)

D = pd.read_csv(os.path.join(r"C:\Users\stefk\Documents\paper_tension\fig5_flapping\results", "delta_lift.csv"))
d = D[(D.run.str.contains("run008")) & (D.U == 4) & (D.aoa == 10)].sort_values("V")
P = np.vstack(g.P.values)
g = g.assign(C=P[:, ON].mean(1) - P[:, OFF].mean(1))
c = g.groupby("V").C.agg(["mean", "sem"])
ref = c.loc[c.index.min(), "mean"]
a = ax[3]
a.axhline(0, color="k", lw=0.8)
a.errorbar(d.V / 1e3, d.dL, yerr=d.dL_sem, fmt="D-", mfc="none", color=GRAY, capsize=3, label="Old: cycle mean, in time order")
a.errorbar(c.index / 1e3, c["mean"] - ref, yerr=c["sem"], fmt="o-", color=BLACK, capsize=3, label="New: on minus off")
vv = np.linspace(0, 1600, 60)
a.plot(vv / 1e3, 31.6 * np.interp(vv, BV, BS), "--", color=RED, lw=2, label="Model, both wings: whole edge")
a.set_xlabel("DEA voltage in kV")
a.set_ylabel(r"$\Delta L$ in mN")
a.legend(fontsize=8.5, frameon=False, loc="lower left")
a.set_ylim(-48, 36)
a.set_title("d) Same run", loc="left", fontsize=11)
a.grid(alpha=.3)
save(fig, "I_contrast")

# ---------------------------------------------------- every result in %
rows = [
    ("Measured", "Gliding, free edge, all campaigns", 0.0, 1.5, None, None),
    ("Measured", "Flapping, 4 m/s, 10°: old cycle mean", 37.0, 3.4, None, None),
    ("Measured", "Flapping, 6 m/s, 10°: old cycle mean", 33.5, 4.6, None, None),
    ("Measured", "Flapping, 4 m/s, 10°: with time term", 11.4, 43.4, None, None),
    ("Measured", "Flapping, 4 m/s, 10°: on minus off", -21.5, 22.6, None, None),
    ("Measured", "Flapping, 6 m/s, 10°: on minus off", -3.6, 18.4, None, None),
    ("Model", "Fibres, free edge", None, None, 0.2, 0.4),
    ("Model", "Fibres, line inboard", None, None, 3.0, 8.9),
    ("Model", "Fibres, line whole edge", None, None, 6.4, 19.8),
    ("Model", "Full, n = 1", None, None, 12.0, 23.0),
    ("Model", "Full, n = 2", None, None, 17.0, 32.0),
    ("Model", "Full, n = 3", None, None, 19.0, 37.0),
]
fig, a = plt.subplots(figsize=(9.5, 4.4))
y = np.arange(len(rows))[::-1]
a.axvline(0, color="k", lw=0.8)
a.axvspan(-10, 10, color="#EEEEEE", lw=0, label="Rough limit, fast effect only: 10 %")
for yy, r in zip(y, rows):
    if r[0] == "Measured":
        col = GRAY if "old" in r[1] else BLACK
        a.errorbar([r[2]], [yy], xerr=[r[3]], fmt="o" if "old" not in r[1] else "D", mfc=col if "old" not in r[1] else "none", color=col, capsize=4, ms=7)
    else:
        a.barh(yy, r[5] - r[4], left=r[4], height=0.55, color=BLUE if r[1].startswith("Full") else RED, alpha=0.85)
a.set_yticks(y); a.set_yticklabels([r[1] for r in rows], fontsize=10)
a.set_xlabel("Lift change at 1.6 kV, % of lift (4 m/s, 10° unless stated)")
a.set_xlim(-55, 60)
a.grid(axis="x", alpha=.3)
a.axhline(4.5, color="#BBBBBB", lw=1)
a.text(-49, 11.45, "Measured, ± 2 s.e.", fontsize=9, color="#555555", va="top")
a.text(-49, 4.35, "Model", fontsize=9, color="#555555", va="top")
a.legend(fontsize=9, loc="lower left", frameon=False)
save(fig, "I_all_percent")
