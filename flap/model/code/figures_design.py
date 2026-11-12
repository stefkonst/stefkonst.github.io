"""Design map: stroke amplitude reachable at resonance against the stroke thrust needs.
a, route 1 (stiffer spring, same inertia); b, route 2 (less moving inertia, same stiffness); lines for force lags
52 (measured), 20, 10, 5, 2 ms; shaded: below the Strouhal threshold (0.15) at 3 m/s; dashed: 2 and 4 m/s.
Run: python figures_design.py -> ../fig/T5_design.png/svg"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FLAP = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common")]
import style as ST
import matplotlib.pyplot as plt
Dm = json.load(open(os.path.join(FLAP, "flapping", "data", "design_map.json"))); f = np.array(Dm["f"])
cols = {"0.052": "k", "0.02": ST.C[1], "0.01": ST.C[5], "0.005": ST.C[2], "0.002": ST.C[0]}
ST.apply()
fig = plt.figure(figsize=(ST.WIDTH_MM * ST.MM, 80 * ST.MM))
ax = [fig.add_axes([0.08 + i_ * 0.5, 0.13, 0.4, 0.8]) for i_ in range(2)]
for i_, a_ in enumerate(ax): a_.text(-0.16, 1.02, "ab"[i_], transform=a_.transAxes, fontsize=9, fontweight="bold")
for a, route, ttl in ((ax[0], "stiffer", "stiffer spring, same moving mass"), (ax[1], "lighter", "less moving mass, same spring")):
    need3 = np.array(Dm["need"]["3"]); a.fill_between(f, 0, need3, color="0.88", lw=0)
    for U, ls in (("2", ":"), ("4", "--")): a.plot(f, Dm["need"][U], color="0.45", ls=ls, lw=0.8)
    for tau, c in cols.items():
        a.plot(f, Dm["routes"][tau][route], color=c, lw=1.4 if tau == "0.052" else 1.0, label="lag %g ms%s" % (float(tau) * 1e3, " (measured)" if tau == "0.052" else ""))
    a.plot([8.4], [12.2], "o", color="k", ms=4)
    a.set(xlabel="Resonance (Hz)", ylabel="Stroke amplitude (deg)", ylim=(0, 35), xlim=(6, 30))
    ST.note(a, ttl, xy=(0.30, 0.96))
    if route == "lighter":
        a.axvline(Dm["f_wing_only"], color=ST.C[3], lw=0.7, ls="-."); ST.note(a, "wing alone\nmoving", xy=((Dm["f_wing_only"] - 6) / 24 + 0.01, 0.62), color=ST.C[3])
ST.note(ax[0], "grey: no thrust at 3 m/s (St < 0.15)\ndotted 2 m/s, dashed 4 m/s", xy=(0.40, 0.80))
ax[1].legend(loc="upper right", fontsize=5)
ST.save(fig, os.path.join(HERE, "..", "fig"), "T5_design")
print("done")
