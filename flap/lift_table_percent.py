"""Every flapping condition in mN and %: old cycle mean, within-cycle contrast, and the model, with both baselines.
% is of the cycle-mean lift at low voltage (measured) or at 0 V (model). Under 20 mN of baseline, % is not meaningful and is left blank.
Run after recalibrate_contrast.py:  python lift_table_percent.py   (lift_table_percent.csv, ../results/contrast_percent.png and svg)"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\stefk\Documents\website\site\flap")
import wing_analytical as W

HERE = os.path.dirname(os.path.abspath(__file__))
PT = r"C:\Users\stefk\Documents\paper_tension\fig5_flapping\results"
AREA = 84e-4
GEOM = dict(a=(0.06, 0.06), b=(0.07, 0.06), share=(0.55, 0.45), radius=(0.06, 0.12))
MINL = 20.0


def ten(w, i, V):
    Hm, G = w["H_m"], 2 * w["C1"]
    ex, ey = w["lam_x"] - 1, w["lam_y"] - 1
    nx = 2 * G * Hm * (2 * ex + ey) + w["E_x"] * Hm * ex
    ny = 2 * G * Hm * (ex + 2 * ey) + w["E_y"] * Hm * ey + w["r"][i] * max(w["F_D"] - w["dF"] * (V / 1600.0) ** 2, 0.0) / w["s"]
    return nx, ny, Hm, G + 0.5 * (w["E_x"] + w["E_y"]) / 6


def base(U, a, r_in, r_out, lam=1.01):
    w = dict(W.DISCRETE, lam_x=lam, lam_y=lam, dF=8.6e-3, s=0.010, r=(r_in, r_out), **GEOM)
    return float(W.cycle(w, ten, V=0.0, U=float(U), alpha_deg=float(a), area=AREA, Phi0_deg=30.0, f=1.0)[1].mean())


def pct(x, b):
    return 100 * x / abs(b) if abs(b) >= MINL else np.nan


A = pd.read_csv(os.path.join(PT, "contrast_attached.csv"))
D = pd.read_csv(os.path.join(PT, "delta_lift.csv"))
D = D[(D.V == 1600) & (D.f == 1)]
rows = []
for _, r in A.iterrows():
    d = D[(D.U == r.U) & (D.aoa == r.aoa)].iloc[0]
    Lw, Li = base(r.U, r.aoa, 0.81, 0.81), base(r.U, r.aoa, 0.81, 0.02)
    rows.append(dict(U=r.U, aoa=int(r.aoa), L_meas=r.L_low, L_model=Lw, old_mN=d.dL, old_pct=pct(d.dL, d.L0),
                     a_mN=r.a, a_se=r.se, a_pct=pct(r.a, r.L_low), a_pct_se=pct(r.se, r.L_low), placebo_mN=r.placebo, placebo_pct=pct(r.placebo, r.L_low),
                     whole_mN=r.m_whole, whole_pct=pct(r.m_whole, Lw), inboard_mN=r.m_in, inboard_pct=pct(r.m_in, Li), free_mN=r.m_free))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, "lift_table_percent.csv"), index=False)
pd.set_option("display.width", 220)
print(T.round(1).to_string(index=False))
B = pd.read_csv(os.path.join(PT, "contrast_blocks.csv"))
print(B[(B.U == 6) & (B.aoa >= 20)][["U", "aoa", "a", "se", "L_low"]].assign(pct=lambda x: 100 * x.a / x.L_low.abs(), pse=lambda x: 100 * x.se / x.L_low.abs()).round(1).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
x = np.arange(len(T))
lab = [f"{u:g} m/s\n{a:+d}°" for u, a in zip(T.U, T.aoa)]
ax[0].axhline(0, color="k", lw=0.8)
ax[0].errorbar(x, T.a_mN, yerr=2 * T.a_se, fmt="o", color="k", capsize=3, label="Measured, within-cycle, ± 2 s.e.")
ax[0].plot(x, T.old_mN, "D", mfc="none", color="#7F7F7F", ms=7, label="Old cycle mean against 0 V")
ax[0].plot(x, T.whole_mN, "_", color="#D95319", ms=22, mew=2.5, label="Model: line, whole edge")
ax[0].plot(x, T.inboard_mN, "_", color="#0072BD", ms=22, mew=2.5, label="Model: line, inboard")
ax[0].set_ylabel(r"$\Delta L$ at 1.6 kV in mN")
ax[0].set_title("a) In mN", loc="left")
ax[0].legend(fontsize=8.5, loc="upper right", ncol=2)
ax[0].set_ylim(-55, 88)
ax[1].axhline(0, color="k", lw=0.8)
m = T.a_pct.notna()
ax[1].errorbar(x[m], T.a_pct[m], yerr=2 * T.a_pct_se[m].abs(), fmt="o", color="k", capsize=3)
ax[1].plot(x, T.old_pct, "D", mfc="none", color="#7F7F7F", ms=7)
ax[1].plot(x, T.whole_pct, "_", color="#D95319", ms=22, mew=2.5)
ax[1].plot(x, T.inboard_pct, "_", color="#0072BD", ms=22, mew=2.5)
ax[1].set_ylabel(r"$\Delta L$ at 1.6 kV in % of lift")
ax[1].set_title("b) In %, where the baseline is above 20 mN", loc="left")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=9); a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "results", "contrast_percent.png"), dpi=200)
fig.savefig(os.path.join(HERE, "..", "results", "svg", "contrast_percent.svg"))
