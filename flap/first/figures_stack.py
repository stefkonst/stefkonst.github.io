"""The stated drive and the stack as built. Run: python figures_stack.py -> V_drive, V_contrast, V_fit, V_lift (.png, .svg)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
DAT = os.path.join(HERE, "data")
G = json.load(open(os.path.join(DAT, "geometry.json")))
D = json.load(open(os.path.join(DAT, "drive_check.json")))
Fr = json.load(open(os.path.join(DAT, "first_rise.json")))
load = lambda f: [r for r in map(json.loads, open(os.path.join(DAT, f))) if "error" not in r]
S, F = load("fit_stack.jsonl"), load("fit_family.jsonl")
A_, P_ = PS.MATLAB[1], PS.MATLAB[0]
ETAS = (0.05, 0.15, 0.25, 0.4)
COL = dict(zip(ETAS, ("#4c78a8", "#2a9d8f", "#e9a23b", "#c0392b")))
swing = lambda r: max(r["res"][-1]["sag"]) - max(r["res"][0]["sag"])


def save(fig, nm):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, nm + ".png"))
    fig.savefig(os.path.join(HERE, nm + ".svg"))


# drive
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
a = ax[0]
x = (np.array(D["x"]) + .5) % 1 - .5
o = np.argsort(x)
x = x[o] * 1e3
a.plot(x, np.array(D["y"])[o], "ko", ms=3, label="Measured")
a.plot(x, np.array(D["fits"]["sin^4, no lag"]["u"])[o], color=P_, ls="--", label=r"$V^2$, rms %.1f µm" % D["fits"]["sin^4, no lag"]["rms_um"])
a.plot(x, np.array(D["power"]["u"])[o], color=PS.ON, label=r"$V^{%.1f}$, rms %.1f µm" % (D["power"]["p"], D["power"]["rms_um"]))
a.set(xlabel="Time from the cycle high in ms", ylabel="Edge shift $u$ in mm", title="Cycle shape", ylim=(.3, .53))
a.legend(fontsize=7.5)
a = ax[1]
V = np.array(Fr["V"]) * 1500
a.plot(V, Fr["up"]["0"], color="k", lw=1.6, label="First rise")
a.plot(V, Fr["down"]["0"], color="0.55", lw=1.6, label="First fall")
a.plot(V, Fr["late_up"], color=PS.ON, lw=1.6, label="Cycles 3–9, rise")
a.plot(V, Fr["late_down"], color=A_, ls="--", lw=1.2, label="Cycles 3–9, fall")
a.set(xlabel="Voltage in V, from the set waveform", ylabel="Edge shift $u$ in mm", title="Shift against voltage", xlim=(0, 1500), ylim=(0, .55))
a.legend(fontsize=7.5)
a = ax[2]
a.plot(D["t_max"], D["mid_max"], "-o", ms=3, color=PS.ON, label="Cycle high")
a.plot(D["t_min"], D["mid_min"], "-o", ms=3, color=A_, label="Cycle low, 0 V")
a.set(xlabel="Time in s", ylabel="Edge shift $u$ in mm", title="Each cycle", xlim=(0, 10), ylim=(0, .55))
a.legend(fontsize=7.5)
PS.finish(ax)
save(fig, "V_drive")

# wrinkle contrast
fig, a = plt.subplots(1, 1, figsize=(4.8, 3.2))
vals = np.array([[4.47, 4.88, 5.16], [4.19, 4.60, 5.06], [6.12, 6.30, 6.55]])
for k, (lab, c) in enumerate((("Before voltage", PS.OFF), ("Cycle low", A_), ("Cycle high", PS.ON))):
    a.bar(np.arange(3) + (k - 1) * .25, vals[:, k], width=.23, color=c, label=lab)
a.set_xticks(range(3)); a.set_xticklabels(("Root third", "Mid third", "Tip third"))
a.set(ylabel="Streak contrast, grey levels rms", title="Wrinkle contrast", ylim=(0, 8))
a.legend(fontsize=7.5)
PS.finish([a])
save(fig, "V_contrast")

# model
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
a = ax[0]
s = np.array(G["s_mm"])
pas = next(r for r in S if r["kind"] == "scan" and r["eta"] == .15 and r["eta_f"] == 0)["res"]
act = next(r for r in F if r["kind"] == "pull" and r["N"] == 1 and r["pull"] == 3.2 and r["eta"] == .15)["res"]
a.plot(s, G["before"]["sag_mm"], color="k", lw=1.5, label="Measured, before")
a.plot(s, G["high"]["sag_mm"], color=PS.ON, lw=1.5, label="Measured, cycle high")
a.plot(pas[0]["s"], pas[0]["sag"], color="0.5", ls="--", label="Model, 0 V")
a.plot(pas[-1]["s"], pas[-1]["sag"], color=P_, ls="--", label=r"Model, passive roll, $\eta$ 0.15")
a.plot(act[-1]["s"], act[-1]["sag"], color=A_, ls="--", label=r"Model, active roll, $\eta$ 0.15")
a.set(xlabel=r"Along the knot-to-tip line, $s$ in mm", ylabel=r"Sag $\delta$ in mm", title="Shape, pull 3.2 mm", xlim=(0, 66), ylim=(0, 2.8))
a.legend(fontsize=7, loc="lower center")
a = ax[1]
for e in ETAS:
    q = sorted((r for r in S if r["kind"] == "scan" and r["eta"] == e and r["eta_f"] * r["wa"] * 1e3 <= 3.5), key=lambda r: r["eta_f"] * r["wa"])
    a.plot([r["eta_f"] * r["wa"] * 1e3 for r in q], [swing(r) for r in q], "o", ms=3, color=COL[e])
    z = np.linspace(0, 3.5, 2)
    a.plot(z, -.73 * e + .155 * z, color=COL[e], lw=1, label=r"Skin $\eta$ %.2f" % e)
a.axhline(.18, color="k", lw=.9, ls="--")
a.text(.05, .2, "Measured swing", fontsize=7.5)
a.axhline(0, color="0.4", lw=.7)
a.set(xlabel=r"Roll, $\eta_f w_a$ in mm", ylabel=r"Sag change at 1500 V, $\Delta\delta$ in mm", title="Which fits", xlim=(0, 3.5), ylim=(-.4, .4))
a.legend(fontsize=7, loc="lower right")
a = ax[2]
for e in ETAS:
    for N, ls, mk, lb in ((1, "-", "o", "2 el."), (3, "--", "s", "4 el.")):
        q = sorted((r for r in F if r["kind"] == "pull" and r["N"] == N and r["eta"] == e and r["pull"] >= 2), key=lambda r: r["pull"])
        a.plot([r["pull"] for r in q], [swing(r) for r in q], ls=ls, marker=mk, ms=3, color=COL[e], label=r"$\eta$ %.2f, %s" % (e, lb))
a.axhline(0, color="0.4", lw=.7)
a.plot([3.2], [.18], "k*", ms=9, label="Measured")
a.set(xlabel="Pull in mm", ylabel=r"Sag change at 1500 V, $\Delta\delta$ in mm", title="Prediction: pull and electrodes", ylim=(-.2, .55))
a.legend(fontsize=6.5, ncol=2)
PS.finish(ax)
save(fig, "V_fit")

# lift
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
for a, N, ti in zip(ax, (1, 3), ("2 electrodes", "4 electrodes")):
    for e in ETAS:
        for act_, ls, mk, lb in ((True, "-", "o", "active roll"), (False, ":", "s", "passive roll")):
            q = sorted((r for r in F if r["kind"] == "lift" and r["N"] == N and r["eta"] == e and (r["eta_f"] > 0) == act_), key=lambda r: r["pull"])
            a.plot([r["pull"] for r in q], [r["L1"] - r["L0"] for r in q], ls=ls, marker=mk, ms=3, color=COL[e], label=r"$\eta$ %.2f, %s" % (e, lb))
    a.axhline(0, color="0.4", lw=.7)
    a.set(xlabel="Pull in mm", title=ti + ", 1500 V, 6°, 4 m/s", xlim=(3, 5.2))
ax[0].set(ylabel=r"Lift change $\Delta L$ in mN")
ax[1].legend(fontsize=6.5, ncol=2)
PS.finish(ax)
save(fig, "V_lift")
print("corner, measured 10 um along, 43 um normal")
for e in ETAS:
    r = next(r for r in F if r["kind"] == "pull" and r["N"] == 1 and r["pull"] == 3.2 and r["eta"] == e)["res"]
    print(e, "corner um", round((r[-1]["rootpt"][0] - r[0]["rootpt"][0]) * 1e3), round((r[-1]["rootpt"][1] - r[0]["rootpt"][1]) * 1e3), "sag peak at s", r[-1]["s"][int(np.argmax(r[-1]["sag"]))])
for r in sorted((r for r in F if r["kind"] == "lift" and r["pull"] == 4.0 and r["N"] == 1), key=lambda r: (r["eta"], r["eta_f"])):
    print("bulge N1 pull 4 eta %.2f ef %.3f z %.2f -> %.2f L0 %.1f dL %+.2f" % (r["eta"], r["eta_f"], r["z0"], r["z1"], r["L0"], r["L1"] - r["L0"]))
