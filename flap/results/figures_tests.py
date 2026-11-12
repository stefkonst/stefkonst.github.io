"""Figures from the robot tests (6 Nov 2026), same style as figures_results.py:
Fig0_robot   the robot in three views, with the motion range at 8 Hz
Fig6_flapping  frequency response, free decays, voltage, active against unpowered wing
Fig7_flight    the flight model at the measured kinematics against the design point
Run after flap/flapping/code/*.py and flap/air/measured_flight.py: python figures_tests.py"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FLAP = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.expanduser("~"), "Documents", "paper_tension", "common"), os.path.join(FLAP, "flapping", "code")]
import style as ST
import matplotlib.pyplot as plt
import shutil
OUTS = [os.path.join(HERE, "fig"), os.path.join(os.path.expanduser("~"), "Documents", "YEAR2", "robot", "results")]


def save(fig, name):
    ST.save(fig, OUTS[0], name)
    os.makedirs(os.path.join(OUTS[1], "svg"), exist_ok=True)
    for f in os.listdir(OUTS[0]):
        if f.startswith(name) and f.endswith(".png"):
            shutil.copy2(os.path.join(OUTS[0], f), OUTS[1])
        if f.startswith(name) and f.endswith(".svg"):
            shutil.copy2(os.path.join(OUTS[0], f), os.path.join(OUTS[1], "svg"))
C, INK2 = ST.C, ST.INK2
FD = lambda n: os.path.join(FLAP, "flapping", "data", n)
TR = lambda n: np.load(os.path.join(FLAP, "flapping", "data", "tracks", n + ".npz"))
SW = json.load(open(FD("sweepfit.json")))
DEC = json.load(open(FD("decays.json")))
RING = json.load(open(FD("ring1hz.json")))
PAIR = json.load(open(FD("pair.json")))
FS = 200.0
VID = r"C:/Users/stefk/MIT Dropbox/SMRL/Shared/Stefania/full_tension/"


def model(p, f, tau=None):
    K, fn, z, t = p["K"], p["fn"], p["zeta"], p["tau"] if tau is None else tau
    r = np.asarray(f) / fn
    return K / np.sqrt((1 - r ** 2) ** 2 + (2 * z * r) ** 2) / np.sqrt(1 + (2 * np.pi * np.asarray(f) * t) ** 2)


# ---------------- Fig. 6: flapping ----------------
ST.LETTER_START = 2
fig, ax = ST.make_fig(4)
P = SW["fits"]["zeta 0.07 (decay)"]
runs = SW["runs"]
sw = sorted([r for r in runs if r["V"] == 1200 and r["run"].startswith("full")], key=lambda r: r["f"])
a = ax[0]
ff = np.linspace(0.5, 16, 400)
a.errorbar([r["f"] for r in sw], [r["A"][0] for r in sw], yerr=[r["A1_sd"] for r in sw], fmt="o", color="k", ms=3, label="measured, 1200 V")
a.plot(ff, model(P, ff), color=C[0], label=r"one mode, lag $\tau$ = %.0f ms" % (P["tau"] * 1e3))
a.plot(ff, model(P, ff, tau=0.0), color=C[0], ls=":", lw=0.9, label="same, no lag")
un = next(r for r in runs if r["run"].startswith("only"))
a.plot([un["f"]], [un["A"][0]], "s", mfc="white", mec=C[1], ms=4, label="wing unpowered")
a.set(xlabel="Drive frequency (Hz)", ylabel="Stroke, fundamental (deg)", ylim=(0, 26), xlim=(0, 16))
a.legend(loc="upper left", fontsize=5)
ST.note(a, r"fit: $f_n$ %.1f Hz, $\zeta$ %.2f" % (P["fn"], P["zeta"]), xy=(0.55, 0.40))

a = ax[1]
for d, col, lab in ((DEC[0], C[3], "decay after 1400 V (side)"), (DEC[1], C[1], "decay, wing unpowered (diag.)")):
    fa = np.array([q for q in d["f_vs_amp"] if q[0] > 0.8])
    a.plot(fa[:, 0], fa[:, 1], "o-", ms=2.5, lw=0.7, color=col, label=lab)
rg = RING["full_DEA_core_wing_1hz_1000V_side"]
a.errorbar([rg["A"]], [rg["f_n"]], yerr=[0.05], fmt="D", color="k", ms=3, label="ringing at 1 Hz, 0 V")
a.axhline(P["fn"], color=C[0], lw=0.6, ls="--")
ST.note(a, "sweep fit", xy=(0.03, (P["fn"] - 7.2) / 3.0 + 0.045))
a.axhline(8.0, color="0.6", lw=0.5)
ST.note(a, "drive 8 Hz", xy=(0.03, (8.0 - 7.2) / 3.0 - 0.01))
a.set(xlabel="Amplitude, half peak to peak (deg)", ylabel="Free frequency (Hz)", ylim=(7.2, 11.0), xlim=(0, 7))
a.legend(loc="upper right", fontsize=5)

a = ax[2]
v1 = {r["V"]: r["A"][0] for r in runs if r["f_drive"] == 1 and r["run"].startswith("full")}
v8 = {r["V"]: r["A"][0] for r in runs if r["f_drive"] == 8 and r["run"].startswith("full")}
VV = np.linspace(950, 1450, 50)
for d, col, lab in ((v1, "k", "1 Hz"), (v8, C[0], "8 Hz")):
    ks = sorted(d)
    a.plot(ks, [d[k] for k in ks], "o", color=col, ms=3.5, label=lab)
    a.plot(VV, d[ks[0]] * (VV / ks[0]) ** 2, ls=":", color=col, lw=0.8)
a.set(xscale="log", yscale="log", xlabel="Voltage (V)", ylabel="Stroke, fundamental (deg)")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
a.xaxis.set_major_locator(FixedLocator([1000, 1200, 1400])); a.xaxis.set_major_formatter(FixedFormatter(["1000", "1200", "1400"])); a.xaxis.set_minor_locator(NullLocator())
a.yaxis.set_major_locator(FixedLocator([3, 5, 10, 20])); a.yaxis.set_major_formatter(FixedFormatter(["3", "5", "10", "20"])); a.yaxis.set_minor_locator(NullLocator())
ST.note(a, r"dotted: $\propto V^2$", xy=(0.04, 0.96))
a.legend(loc="lower right", fontsize=5)

a = ax[3]
fns = np.linspace(7.6, 9.6, 200)
a.plot(fns, model(dict(P, fn=None) | {"fn": fns}, 8.0), color=C[0], label="model at 8 Hz")
ac = next(r for r in sw if r["f_drive"] == 8)["A"][0]
a.axhline(ac, color="k", lw=0.7); ST.note(a, "wing powered", xy=(0.62, (ac - 4) / 16 + 0.06), color="k")
a.axhline(un["A"][0], color=C[1], lw=0.7); ST.note(a, "wing unpowered", xy=(0.03, (un["A"][0] - 4) / 16 - 0.02), color=C[1])
for x, col, lab in ((P["fn"], C[0], "sweep fit"), (DEC[0]["f_n"], C[3], "decay 12:28"), (DEC[1]["f_n"], C[1], "decay 12:43")):
    a.axvline(x, color=col, ls="--", lw=0.7)
    a.text(x - 0.02, 19.6, lab, rotation=90, fontsize=5, color=col, ha="right", va="top")
a.set(xlabel=r"Natural frequency $f_n$ (Hz)", ylabel="Stroke at 8 Hz (deg)", ylim=(4, 20))
save(fig, "Fig6_flapping")

# ---------------- Fig. 7: flight at the measured kinematics ----------------
DM = [r for r in map(json.loads, open(os.path.join(FLAP, "air", "data", "dyn_measured.jsonl"))) if "L_m" in r]
ST.LETTER_START = 2
fig, ax = ST.make_fig(3)
cases = (((8.4, 12.2), "k", "measured: 8.4 Hz, ±12°"), ((8.4, 20.0), C[0], "8.4 Hz, ±20°"), ((16.0, 20.0), C[2], "16 Hz, ±20°"), ((25.0, 20.0), C[1], "design: 25 Hz, ±20°"))
g = lambda f, Ph, d, a_: sorted([r for r in DM if r["f"] == f and r["Phi_deg"] == Ph and r["design"] == d and r["alpha_deg"] == a_], key=lambda r: r["U"])
for (f, Ph), col, lab in cases:
    rr = g(f, Ph, "passive", 10.0)
    ax[0].plot([r["U"] for r in rr], [r["T_m"] * 1e3 for r in rr], "o-", ms=3, color=col, label=lab)
    ax[1].plot([r["U"] for r in rr], [r["L_m"] * 1e3 for r in rr], "o-", ms=3, color=col, label=lab)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set(xlabel=r"Flight speed (m s$^{-1}$)", ylabel="Thrust per wing (mN)")
ax[0].legend(loc="lower left", fontsize=5)
ST.note(ax[0], r"$\alpha$ 10°, wing at 0 V", xy=(0.04, 0.96))
ax[1].axhline(18.4 / 2, color="k", ls="--", lw=0.6)
ST.note(ax[1], "half the robot weight", xy=(0.04, 10.3 / 12.0))
ax[1].set(xlabel=r"Flight speed (m s$^{-1}$)", ylabel="Lift per wing (mN)", ylim=(0, 12))
a = ax[2]
for (f, Ph), col, lab in cases:
    rr = g(f, Ph, "passive", 10.0) + g(f, Ph, "passive", 6.0)
    st = [f * 2 * 0.05 * np.sin(np.radians(Ph)) / r["U"] for r in rr]
    a.plot(st, [r["T_m"] * 1e3 for r in rr], "o", ms=3, color=col)
a.axhline(0, color="k", lw=0.5)
a.set(xscale="log", xlabel=r"Strouhal number $f A_{tip}/U$", ylabel="Thrust per wing (mN)")
ST.note(a, r"$\alpha$ 6° and 10°", xy=(0.04, 0.96))
save(fig, "Fig7_flight")

# ---------------- Fig. 0: the robot, three views ----------------
import cv2
sys.path.insert(0, os.path.join(FLAP, "first"))
from track_views import Stack
ST.apply()
views = (("full_DEA_core_wing_1hz_1000V_side", "side", 0.152), ("full_DEA_core_wing_8hz_1200V_diag_good", "diagonal", 0.152), ("full_DEA_core_wing_8hz_1400V", "front", 0.152))
rng = (("full_DEA_core_wing_8hz_1200V_side", 1000), ("full_DEA_core_wing_8hz_1200V_diag_good", 1000), ("full_DEA_core_wing_8hz_1400V", 420))
W_MM = ST.WIDTH_MM
fig = plt.figure(figsize=(W_MM * ST.MM, 92 * ST.MM))
for i, ((n, lab, px), (nr, nmax)) in enumerate(zip(views, rng)):
    try:
        s = Stack(VID + n + ".avi"); im = np.asarray(s[0], float)
        s2 = Stack(VID + nr + ".avi"); A = np.stack([np.asarray(s2[k], np.float32) for k in range(0, nmax, 4)])
        rgimg = A.max(0) - A.min(0)
    except Exception as e:
        print("video not available:", e); im = np.zeros((600, 832)); rgimg = im
    for j, img, cm in ((0, im, "gray"), (1, rgimg, "magma")):
        axx = fig.add_axes([0.01 + i * 0.33, 0.52 - j * 0.50, 0.32, 0.46])
        axx.imshow(img[60:540, 0:800], cmap=cm, vmin=0 if j else None, vmax=np.percentile(img, 99.5) if j else None)
        axx.set_axis_off()
        if j == 0:
            axx.plot([600, 600 + 10 / px], [440, 440], color="w", lw=2); axx.text(600 + 5 / px, 425, "10 mm", color="w", fontsize=6, ha="center")
        axx.text(8, 30, "abcdef"[i + 3 * j] + "  " + (lab if j == 0 else "motion range, 8 Hz"), color="w", fontsize=7, fontweight="bold")
fig.savefig(os.path.join(HERE, "fig", "Fig0_robot.png"), dpi=300); fig.savefig(os.path.join(HERE, "fig", "Fig0_robot.svg"))
import shutil
shutil.copy2(os.path.join(HERE, "fig", "Fig0_robot.png"), os.path.join(os.path.expanduser("~"), "Documents", "YEAR2", "robot", "results"))
plt.close(fig)
print("done")
