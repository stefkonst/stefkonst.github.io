"""Model against the video. Run: python figures_fit.py <stack folder> -> V_fit.png, V_drive.png, V_wrinkle.png (and .svg)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
D = sys.argv[1]
G = json.load(open(os.path.join(HERE, "data", "geometry.json")))
E = json.load(open(os.path.join(HERE, "data", "ends_drive.json")))
R = [json.loads(l) for l in open(os.path.join(HERE, "data", "fit_cases.jsonl"))]
R = [r for r in R if "error" not in r]
B = [json.loads(l) for l in open(os.path.join(HERE, "data", "lift_bridge.jsonl"))]
get = lambda kind, name, **k: [r for r in R if r["kind"] == kind and r["name"] == name and all(r[q] == v for q, v in k.items())]
A_, P_ = PS.MATLAB[1], PS.MATLAB[0]
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
a = ax[0]
s = np.array(G["s_mm"])
a.plot(s, G["before"]["sag_mm"], color="k", lw=1.5, label="Measured, before")
a.plot(s, G["high"]["sag_mm"], color=PS.ON, lw=1.5, label="Measured, cycle high")
r0 = get("fit", "passive roll, eta .15")[0]["res"]
r1 = get("fit", "band 5 mm, eta .05")[0]["res"]
a.plot(r0[0]["s"], r0[0]["sag"], color="0.5", ls="--", label="Model, 0 V")
a.plot(r0[-1]["s"], r0[-1]["sag"], color=P_, ls="--", label=r"Model, passive roll, $\eta$ 0.15")
a.plot(r1[-1]["s"], r1[-1]["sag"], color=A_, ls="--", label=r"Model, 5 mm band, $\eta$ 0.05")
a.set(xlabel=r"Along the knot-to-tip line, $s$ in mm", ylabel=r"Sag $\delta$ in mm", title="Shape, pull 3.2 mm", xlim=(0, 66), ylim=(0, 2.8))
a.legend(fontsize=7, loc="lower center")
a = ax[1]
for N, ls, mk in ((2, "-", "o"), (4, "--", "s")):
    for nm, c, lb in (("band 5 mm, eta .05", A_, "5 mm band"), ("passive roll, eta .15", P_, "Passive")):
        q = sorted(get("pull", nm, N=N), key=lambda r: r["pull"])
        a.plot([r["pull"] for r in q], [max(r["res"][-1]["sag"]) - max(r["res"][0]["sag"]) for r in q], ls=ls, marker=mk, color=c, ms=3, label=f"{lb}, {N} layers")
a.axhline(0, color="0.4", lw=.7)
a.plot([3.2], [.18], "k*", ms=9, label="Measured swing")
a.set(xlabel="Pull in mm", ylabel=r"Sag change at 1500 V, $\Delta\delta$ in mm", title="Prediction: tension and layers", ylim=(-.5, .5))
a.legend(fontsize=7, ncol=1)
a = ax[2]
for nm, c, lb in (("passive roll, eta .15", P_, "Passive, $\eta$ 0.15"), ("band 5 mm, eta .05", A_, "5 mm band, $\eta$ 0.05")):
    for N, ls, mk in ((2, "-", "o"), (4, "--", "s")):
        q = sorted(get("lift", nm, N=N), key=lambda r: r["pull"])
        a.plot([r["pull"] for r in q], [r["L1"] - r["L0"] for r in q], ls=ls, marker=mk, color=c, ms=3, label=f"{lb}, {N} layers")
a.axhline(0, color="0.4", lw=.7)
a.set(xlabel="Pull in mm", ylabel=r"Lift change $\Delta L$ in mN", title="1500 V, 6° pitch, 4 m/s", xlim=(3, 5.2), ylim=(-.5, 2))
a.legend(fontsize=7)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_fit.png"))

# drive shape, wrinkle contrast
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
a = ax[0]
f = E["fit"]
x = np.array(f["x"]) * 1e3
a.plot(x, f["y"], "ko", ms=3, label="Measured")
a.plot(x, f["u"], color=PS.ON, label=r"$V_{min}/V_{max}$ = %.2f, fitted" % f["ratio"])
a.plot(x, E["fits"]["0 to 1500 V"]["u"], color=P_, ls="--", label="0 to 1500 V")
a.plot(x, E["fits"]["1200 to 1500 V"]["u"], color=A_, ls=":", label="1200 to 1500 V")
a.set(xlabel="Time from the cycle high in ms", ylabel="$u$ in mm", title=r"Cycle shape, $u = u_0 + a\,V^2(t)$", ylim=(.3, .52))
a.legend(fontsize=7.5)
a = ax[1]
reg = ("Root third", "Mid third", "Tip third")
vals = np.array([[4.47, 4.88, 5.16], [4.19, 4.60, 5.06], [6.12, 6.30, 6.55]])
for k, (lab, c) in enumerate((("Before voltage", PS.OFF), ("Cycle low", A_), ("Cycle high", PS.ON))):
    a.bar(np.arange(3) + (k - 1) * .25, vals[:, k], width=.23, color=c, label=lab)
a.set_xticks(range(3)); a.set_xticklabels(reg)
a.set(ylabel="Streak contrast, grey levels rms", title="Wrinkles", ylim=(0, 8))
a.legend(fontsize=7.5)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_drive.png"))

# wrinkle directions over the frame
z = np.load(os.path.join(D, "ref.npz"))
F = np.load(os.path.join(HERE, "data", "field_h15_band_5_mm.npz"))
PX = 8.24
u = np.array([330., -123.]); u /= np.linalg.norm(u)
v = np.array([-u[1], u[0]])
o = np.array([55., 165.])
c = F["x"][F["tri"]].mean(1)[:, :2] * 1e3
P = o + PX * (c[:, 0:1] * u + c[:, 1:2] * v)
th = .5 * np.arctan2(2 * F["c01"], F["c00"] - F["c11"])
d = np.c_[np.cos(th), np.sin(th)]
di = d[:, 0:1] * u + d[:, 1:2] * v
fig, a = plt.subplots(1, 1, figsize=(7, 5))
a.imshow(z["H"], cmap="gray", vmin=40, vmax=210)
keep = (F["state"] == 1) & (np.arange(len(th)) % 4 == 0)
L = 11
a.plot(np.c_[P[keep, 0] - L * di[keep, 0], P[keep, 0] + L * di[keep, 0]].T, np.c_[P[keep, 1] - L * di[keep, 1], P[keep, 1] + L * di[keep, 1]].T, color="#ffb000", lw=.9)
a.set(xlim=(0, 520), ylim=(450, 0), title="Cycle high, with the model's wrinkle lines (orange)")
a.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_wrinkle.png"))
