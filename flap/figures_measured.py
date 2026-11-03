"""Predicted lift change against what has been measured. Only discrete-DEA wings have been tested: the full membrane has not been built.
Input: paper_tension fig4 and fig5 results if present, otherwise model/measured_lift.json (derived numbers only). FE: full_membrane/validation/active_check.jsonl.
Run: python figures_measured.py -> model/M10_measured.png and .svg, model/measured_lift.json"""
import csv, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "plagiopatagiales"))
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
PT = r"C:\Users\stefk\Documents\paper_tension"
OUT = os.path.join(HERE, "model", "measured_lift.json")
CF, CD, CM = PS.MATLAB[0], PS.MATLAB[1], "k"
num = lambda a: float(a) if a not in ("", None) else None

if os.path.exists(PT):
    fl = [r for r in csv.DictReader(open(os.path.join(PT, "fig5_flapping", "results", "delta_lift.csv"))) if num(r["V"]) == 1600.0 and num(r["dL"]) is not None]
    gl = list(csv.DictReader(open(os.path.join(PT, "fig4_gliding", "results", "gliding_summary.csv"))))
    M = dict(flapping=[dict(f=num(r["f"]), U=num(r["U"]), aoa=num(r["aoa"]), dL=num(r["dL"]), sem=num(r["dL_sem"]), L0=num(r["L0"]), dL_on=num(r["dL_on"]), dL_off=num(r["dL_off"])) for r in fl], gliding=[dict(dataset=r["dataset"], membrane=r["membrane"], U=num(r["U"]), n=int(r["n"]), wind_mN=num(r["wind_part_mN"]), floor_mN=num(r["floor_mN"]), pct_lift=num(r["pct_lift"])) for r in gl])
    json.dump(M, open(OUT, "w"), indent=1)
M = json.load(open(OUT))
FE = [json.loads(l) for l in open(os.path.join(HERE, "full_membrane", "validation", "active_check.jsonl"))]
fe = sorted([(r["U"], r["dL_mN"], bool(r.get("ok0") and r.get("ok1"))) for r in FE if r["tag"] == "U" and "dL_mN" in r] + [(4.0, next(r["dL_mN"] for r in FE if r["tag"] == "design"), True)])
disc = [(3.0, 0.5), (4.0, 1.8), (6.0, 14.3)]                  # discrete DEAs with a tendon, T_c = 0.5 N, lambda_pre = 1.03: the main page's table
one = [r for r in M["flapping"] if r["f"] == 1.0]
wind = [r for r in one if r["U"] > 0]
off = [r["dL"] for r in one if r["U"] == 0]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
a = ax[0]
a.plot([q[0] for q in fe], [q[1] for q in fe], "o-", color=CF, label="Predicted: full membrane, one wing, not built")
b_ = [q for q in fe if not q[2]]
a.plot([q[0] for q in b_], [q[1] for q in b_], "o", mfc="white", color=CF, ms=5)
a.plot([q[0] for q in disc], [q[1] for q in disc], "s-", color=CD, label="Predicted: discrete DEAs with a tendon")
pk = [r for r in wind if r["aoa"] == 10.0]
a.errorbar([r["U"] for r in pk], [abs(r["dL"]) for r in pk], yerr=[r["sem"] for r in pk], fmt="D", mfc="white", color=CM, ms=5, capsize=2, label="Measured, flapping, 10°: effect not established")
g = M["gliding"]
a.plot([r["U"] + {"april": -.12, "march": 0, "may": .12}[r["dataset"]] for r in g], [r["floor_mN"] for r in g], "v", color="0.45", ms=5, label="Measured, gliding: nothing above this floor")
a.set(xlabel="Flow speed $U$ in m/s", ylabel=r"Lift change $|\Delta L|$ at 1.6 kV in mN", yscale="log", ylim=(.1, 3000), xlim=(1.5, 7.5))
a.legend(fontsize=7, loc="upper left")
a = ax[1]
a.axhspan(min(off), max(off), color="0.88", lw=0, label=f"No wind: {min(off):.0f} to {max(off):+.0f} mN")
a.axhline(0, color="k", lw=.7)
for u, c, m in ((2.0, PS.MATLAB[4], "^"), (4.0, PS.MATLAB[3], "o"), (6.0, PS.MATLAB[0], "s")):
    r = sorted([q for q in wind if q["U"] == u], key=lambda q: q["aoa"])
    a.errorbar([q["aoa"] for q in r], [q["dL"] for q in r], yerr=[q["sem"] for q in r], fmt=m + "-", color=c, ms=4, capsize=2, label=f"{u:g} m/s")
a.set(xlabel=r"Pitch $\alpha$ in °", ylabel=r"Measured $\Delta L$ at nominal 1.6 kV in mN", xticks=[-5, 0, 10, 20, 30])
a.legend(fontsize=7.5, loc="upper right")
PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(HERE, "model", "M10_measured.png")); plt.close(fig)
for r in wind:
    print(r["U"], r["aoa"], "dL %.1f +- %.1f, L0 %.1f, on %.1f, off %.1f" % (r["dL"], r["sem"], r["L0"], r["dL_on"], r["dL_off"]))
print("no wind", np.round(off, 1).tolist())
print("gliding 4 m/s", [(r["dataset"], r["membrane"], round(r["wind_mN"], 2), round(r["floor_mN"], 2), round(r["pct_lift"], 2)) for r in g if r["U"] == 4])
print("FE", [(q[0], round(q[1], 1), q[2]) for q in fe])
