"""Model with weight against the 5 Nov side and front views. Run: python figures_gravity.py -> V_gravity.png/.svg"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
R = [r for r in map(json.loads, open(os.path.join(HERE, "data", "fit_gravity.jsonl"))) if "error" not in r]
W = json.load(open(os.path.join(HERE, "data", "views.json")))
side = [(v, W["%dV_1hz_side" % v]["fit_low"] + W["%dV_1hz_side" % v]["fit_amp"]) for v in (1000, 1400, 1800)]
fr = W["1800V_1hz_front_avi"]
front = [(1800, fr["fit_low"] + fr["fit_amp"])]
CASES = [((6.0, .15, .4, 1.0), "#2a9d8f", r"54 mN, $\eta_f$ 0.4"), ((9.0, .15, .8, 1.0), "#c0392b", r"83 mN, $\eta_f$ 0.8"), ((9.0, .05, .6, 2.0), "#e9a23b", r"83 mN, $\eta_f$ 0.6, weight ×2"), ((9.0, .15, .1, 1.0), "#4c78a8", r"83 mN, $\eta_f$ 0.1")]
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=False)
for (p, e, f, m), col, lab in CASES:
    r = next(r for r in R if (r["pull"], r["eta"], r["eta_f"], r["mass"]) == (p, e, f, m))["res"]
    V = [q["V"] for q in r]
    ax[0].plot(V, [r[0]["zmin"] - q["zmin"] for q in r], "-o", ms=3, color=col, label=lab)
    ax[1].plot(V, [q["sag"] - r[0]["sag"] for q in r], "-o", ms=3, color=col, label=lab)
ax[0].plot(*zip(*side), "k*", ms=10, label="Measured, side view")
ax[1].plot(*zip(*front), "k*", ms=10, label="Measured, front view")
ax[1].plot([1500], [0.45], "kD", ms=5, mfc="none", label="First run, top view")
ax[0].set(xlabel="Voltage in V", ylabel="Edge drop, out of plane, in mm", title="Out of plane", xlim=(0, 1900))
ax[1].set(xlabel="Voltage in V", ylabel="Edge bow, in plane, in mm", title="In plane", xlim=(0, 1900))
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_gravity.png")); fig.savefig(os.path.join(HERE, "V_gravity.svg"))
