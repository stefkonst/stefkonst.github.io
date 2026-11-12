"""Calibration figure: for each lever arm, the moving inertia (relative to the wing) and the core efficiency that
the measured resonance (8.38 Hz) and quasi-static stroke (4.22 deg fundamental at 1 Hz, 1.2 kV) require; and what
it takes to reach 20 Hz. Run: python figures_calib.py -> ../fig/T4_calib.png/svg"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common")]
import style as ST
import matplotlib.pyplot as plt
Cb = json.load(open(os.path.join(FLAP, "flapping", "data", "calibration.json"))); C = ST.C
ST.LETTER_START = 0
fig, ax = ST.make_fig(3)
for H, ls in ((30, "-"), (45, "--")):
    q = [r for r in Cb["rows"] if r["H_um"] == H]; r = [x["r_mm"] for x in q]
    if H == 30: ax[0].plot(r, [x["I_over_wing"] for x in q], color="k", ls=ls)
    ax[1].plot(r, [x["eta_core"] for x in q], color=C[0], ls=ls, label="film %d µm" % H)
for a in ax[:2]: a.axvline(2.5, color=C[1], lw=0.7, ls=":")
ax[0].axhline(1, color="0.6", lw=0.6); ST.note(ax[0], "wing alone", xy=(0.62, 0.10))
ax[0].set(yscale="log", xlabel="Lever arm $r$ (mm)", ylabel="Moving inertia / wing inertia"); ST.note(ax[0], "design lever", xy=(0.52, 0.92), color=C[1])
ax[1].axhspan(0.06, 0.17, color=C[2], alpha=0.15, lw=0); ST.note(ax[1], "bench roll fits", xy=(0.04, 0.10), color=C[2])
ax[1].set(xlabel="Lever arm $r$ (mm)", ylabel="Core efficiency $\\eta_c$ needed"); ax[1].legend(loc="upper left", fontsize=5)
a = ax[2]; f = np.linspace(5, 30, 100)
a.plot(f, (f / Cb["f_n"]) ** 2, color="k"); a.axvline(Cb["f_n"], color=C[0], lw=0.7, ls=":"); a.axvspan(20, 25, color=C[1], alpha=0.15, lw=0)
a.set(yscale="log", xlabel="Target resonance (Hz)", ylabel="Stiffness / inertia needed, relative to now")
ST.note(a, "now 8.4 Hz", xy=(0.04, 0.92), color=C[0]); ST.note(a, "design band", xy=(0.62, 0.30), color=C[1])
ST.save(fig, os.path.join(HERE, "..", "fig"), "T4_calib")
print("done")
