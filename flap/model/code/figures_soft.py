"""Softening spring figure: a, sweep (data, softening model, linear model); b, free decay frequency against amplitude;
c, stroke at 8 Hz at 1.2 and 1.4 kV: measured, softening model (1.4 kV not fitted), linear, and V^2 scaling.
Run: python figures_soft.py -> ../fig/T3_soft.png/svg"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common")]
import style as ST
import matplotlib.pyplot as plt
S = json.load(open(os.path.join(FLAP, "flapping", "data", "softening.json"))); C = ST.C
ST.LETTER_START = 0
fig, ax = ST.make_fig(3)
a = ax[0]; d = np.array(S["sweep"]["data"])
a.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2], fmt="o", color="k", ms=3, label="measured, 1.2 kV")
a.plot(S["sweep"]["f"], S["sweep"]["soft"], color=C[0], label="softening spring")
a.plot(S["sweep"]["f"], S["sweep"]["lin"], color="0.5", ls="--", label="linear spring")
a.set(xlabel="Drive frequency (Hz)", ylabel="Stroke, fundamental (deg)", xlim=(0, 15)); a.legend(loc="upper right", fontsize=5)
ST.note(a, "$\\gamma$ = %.1f rad$^{-2}$, $f_n$ %.2f Hz" % (S["params"]["gamma"], S["params"]["f_n"]), xy=(0.04, 0.96))
a = ax[1]; dd = np.array(S["decay"]["data"]); a.plot(dd[:, 0], dd[:, 1], "o", color=C[3], ms=3, label="measured")
for key, col, ls, lab in (("soft", C[0], "-", "softening"), ("lin", "0.5", "--", "linear")):
    q = np.array(S["decay"][key])
    if len(q): o = np.argsort(q[:, 0]); a.plot(q[o, 0], q[o, 1], color=col, ls=ls, label=lab)
a.set(xlabel="Amplitude (deg)", ylabel="Free frequency (Hz)", xlim=(0, 6), ylim=(6.5, 10)); a.legend(loc="lower left", fontsize=5)
a = ax[2]; v = S["v1400"]
labels = ["measured", "softening\n(1.4 kV not fitted)", "linear", "$V^2$ from\n1.2 kV measured"]
vals12 = [S["sweep"]["data"][[x[0] for x in S["sweep"]["data"]].index(min(S["sweep"]["data"], key=lambda q: abs(q[0] - 8))[0])][1], v["at1200"], None, None]
r = [v["measured"] / vals12[0], v["soft"] / v["at1200"], None, (1400 / 1200) ** 2]
lin12 = S["sweep"]["lin"][S["sweep"]["f"].index(8.0)] if 8.0 in S["sweep"]["f"] else None
r[2] = v["lin"] / lin12 if lin12 else None
a.bar(range(4), r, color=["k", C[0], "0.6", "0.85"], width=0.6)
a.axhline(1, color="k", lw=0.5)
a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=5.5); a.minorticks_off()
a.set(ylabel="Stroke at 8 Hz, 1.4 kV / 1.2 kV")
ST.save(fig, os.path.join(HERE, "..", "fig"), "T3_soft")
print("done", np.round(r, 3))
