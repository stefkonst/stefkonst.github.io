"""Predictions of experimentally measurable quantities for the default wing (model.DEFAULTS).
Writes validation/P_predictions.png and validation/predictions.csv (long format: sweep, curve, x, quantity, value).
Measurable with: camber -> side camera / laser line at mid-span; lift -> load cell; DEA strain -> markers on a DEA;
TE deflection -> camera; total chord tension -> clamp force on the bench (U = 0).
Run: python predict.py
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M

OUT = os.path.join(HERE, "validation")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
base = dict(nx=16, ny=8)
rows = []


def keep(name, sw, qty):
    for lab, rec in sw["curves"].items():
        for xv, yv in zip(sw["values"], rec[qty]):
            rows.append(dict(sweep=name, curve=lab, x=xv, quantity=qty, value=yv))


Vs = [0, 400, 800, 1000, 1200, 1400, 1600, 1800]
sv = M.sweep(base, "voltage", Vs, aoas=[0, 3, 6, 9, 12])
sa = M.sweep(base, "aoa", list(range(-8, 13, 2)), voltages=[0, 1600], aoas=[None] if False else [0])
# 'aoa' sweep: curves per voltage; aoas only sets the starting angle
sp = M.sweep(dict(base, flap_amp_deg=30, flap_f=3), "phase", list(range(0, 361, 30)), voltages=[0, 1600], aoas=[0, 6, 12])
sb = M.sweep(dict(base, U=0.0, clamp_TE=True, te_cord=False), "voltage", Vs, aoas=[0])
for name, sw in (("voltage", sv), ("aoa", sa), ("phase", sp), ("bench_U0", sb)):
    for q in M.SCALARS:
        if q != "converged":
            keep(name, sw, q)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "predictions.csv"), index=False)

fig, ax = plt.subplots(2, 3, figsize=(10, 6))
cm = plt.cm.viridis(np.linspace(0, 0.9, len(sv["curves"])))
for (lab, r), c in zip(sv["curves"].items(), cm):
    ax[0, 0].plot(Vs, r["camber_mid_pct"], "o-", ms=3, color=c, label=lab)
    ax[0, 1].plot(Vs, np.array(r["L_mN"]) - r["L_mN"][0], "o-", ms=3, color=c, label=lab)
    ax[0, 2].plot(Vs, r["dea_strain_pct"], "o-", ms=3, color=c, label=lab)
ax[0, 0].set(xlabel="voltage (V)", ylabel="camber at mid-span (% chord)", title="camber vs voltage (U = 3 m/s)")
ax[0, 1].set(xlabel="voltage (V)", ylabel="lift change vs 0 V (mN)", title="lift change vs voltage")
ax[0, 2].set(xlabel="voltage (V)", ylabel="DEA strain vs mounted length (%)", title="DEA strain vs voltage")
ax[0, 0].legend(frameon=False, fontsize=6)
for lab, r in sa["curves"].items():
    ax[1, 0].plot(sa["values"], r["L_mN"], "o-", ms=3, label=lab.split(", ")[1])
ax[1, 0].set(xlabel="angle of attack (deg)", ylabel="lift (mN)", title="lift vs AoA, DEA off/on")
ax[1, 0].legend(frameon=False, fontsize=6)
for lab, r in sp["curves"].items():
    ls = "-" if lab.endswith(" 0 V") else "--"
    ax[1, 1].plot(sp["values"], r["L_mN"], ls, marker="o", ms=2.5, label=lab)
    if not lab.endswith(" 0 V"):
        ax[1, 2].plot(sp["values"], r["dea_strain_pct"], "o-", ms=2.5, label=lab)
ax[1, 1].set(xlabel="flap phase (deg)", ylabel="lift (mN)", title="flap cycle (3 Hz, +-30 deg), off solid / on dashed")
ax[1, 2].set(xlabel="flap phase (deg)", ylabel="DEA strain (%)", title="DEA strain over the flap cycle (1.6 kV)")
ax[1, 1].legend(frameon=False, fontsize=5); ax[1, 2].legend(frameon=False, fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "P_predictions.png"), dpi=150)
r = sb["curves"]["AoA 0 deg"]
print("bench (U=0, LE/TE clamped): total chord tension change at 1.6 kV = %.1f %%" % (100 * (r["T_chord_total"][Vs.index(1600)] / r["T_chord_total"][0] - 1)))
print("rows", len(rows))
