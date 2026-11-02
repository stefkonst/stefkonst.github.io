"""Bench observation: a device with the same architecture (active sheet + rolled edge fibre), two fibres on a ring, no flow, 1 Hz drive.
Motion measured from high-speed video (200 fps, top view) by registering a band around each fibre at mid-length; ruler period 12.57 px/mm.
Input: validation/vocal_cords_data.json (derived numbers; raw videos stay local). Run: python vocal_cords.py -> validation/F_vocal.png"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
V_ = os.path.join(HERE, "validation")
D = json.load(open(os.path.join(V_, "vocal_cords_data.json")))
V = np.array(D["V"], float)
L, R = np.array(D["left_fibre_midlength_mm"]), np.array(D["right_fibre_midlength_mm"])
S = 32.0


def arc(Vv, e0, a, b):
    """Arc height of a fibre with excess length a (V/1.6 kV)^2 - e0 beyond its pre-strain; b is the small motion below threshold."""
    return np.hypot(S * np.sqrt(3 / 8 * np.maximum(a * (Vv / 1600) ** 2 - e0, 0)), b)


def fit(y):
    best = None
    for e0 in np.linspace(0, 0.08, 161):
        for a in np.linspace(0.005, 0.08, 76):
            for b in (0.05, 0.1, 0.15):
                E = np.sum((arc(V, e0, a, b) - y) ** 2)
                if best is None or E < best[0]:
                    best = (E, e0, a, b)
    return best[1:], float(np.sqrt(best[0] / len(y)))


vv = np.linspace(1000, 1800, 100)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
a = ax[0]
a.plot(V / 1e3, R, "o", color=PS.MATLAB[0], label="Fibre 1"); a.plot(V / 1e3, L, "s", color=PS.MATLAB[1], label="Fibre 2")
a.plot(vv / 1e3, R[0] * (vv / 1000) ** 2, ":", color=PS.MATLAB[0], label=r"$V^2$ from 1 kV"); a.plot(vv / 1e3, L[0] * (vv / 1000) ** 2, ":", color=PS.MATLAB[1])
a.set(xlabel="Voltage $V$ in kV", ylabel="Mid-length motion in mm"); a.legend()
a = ax[1]
fits = {}
for y, c, lb in ((R, PS.MATLAB[0], "Fibre 1"), (L, PS.MATLAB[1], "Fibre 2")):
    p, rms = fit(y)
    fits[lb] = dict(e0=p[0], a=p[1], b=p[2], rms_mm=rms)
    a.plot(V / 1e3, y, "o", color=c); a.plot(vv / 1e3, arc(vv, *p), "-", color=c, label=f"{lb}: rms {rms:.2f} mm")
a.set(xlabel="Voltage $V$ in kV", ylabel="Mid-length motion in mm"); a.legend()
a = ax[2]
t = np.arange(len(D["trace_1800V_right_mm"])) / D["fps"]
a.plot(t, D["trace_1800V_right_mm"], color=PS.MATLAB[0], label="Fibre 1"); a.plot(t, D["trace_1800V_left_mm"], color=PS.MATLAB[1], label="Fibre 2")
a.set(xlabel="Time $t$ in s", ylabel="Displacement at 1.8 kV in mm"); a.legend()
PS.finish(ax); fig.tight_layout(); fig.savefig(os.path.join(V_, "F_vocal.png")); plt.close(fig)
print(json.dumps(fits, indent=1))
json.dump(fits, open(os.path.join(V_, "vocal_cords_fit.json"), "w"), indent=1)
