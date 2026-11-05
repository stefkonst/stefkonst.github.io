"""Second set of videos, 5 Nov. Run: python figures_views.py <track folder> <avi folder> -> V_views.png/.svg, clips/side_1800V.mp4, clips/diag_15Hz.mp4"""
import json, os, subprocess, sys
import numpy as np, cv2
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
from track_views import Stack
PS.apply()
TR, AV = sys.argv[1:3]
R = json.load(open(os.path.join(HERE, "data", "views.json")))
PX, FPS = 6.14, 200.0
A_, P_ = PS.MATLAB[1], PS.MATLAB[0]

# hanging string: roll weight w, span L, sag K/T; tension falls as c V^2
w = 1.1e3 * 23e-3 * 60e-6 * 9.81
K = w * .065 ** 2 / 8
Vs = np.array([1000, 1400, 1800.])
hi = np.array([R["%dV_1hz_side" % v]["fit_low"] + R["%dV_1hz_side" % v]["fit_amp"] for v in Vs])
lo = np.array([R["%dV_1hz_side" % v]["fit_low"] for v in Vs])
g = least_squares(lambda q: K / (q[0] - q[1] * (Vs / 1e3) ** 2) - K / q[0] - hi * 1e-3, [17e-3, 3e-3], bounds=([1e-3, 0], [.2, .1]))
T0, c = g.x
json.dump(dict(w_N_m=w, T0_mN=T0 * 1e3, c_mN_kV2=c * 1e3, rest_sag_mm=K / T0 * 1e3, rms_mm=float(np.std(g.fun) * 1e3)), open(os.path.join(HERE, "data", "views_string.json"), "w"), indent=1)
print("string fit T0 %.1f mN c %.2f mN/kV^2" % (T0 * 1e3, c * 1e3))

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))
a = ax[0]
for nm, col, lab in (("1800V_1hz_side", A_, "Side, out of plane"), ("1800V_1hz_front_avi", PS.ON, "Front, in plane"), ("1800V_1hz_diag_sine", P_, "Diagonal, plain sine")):
    z = np.load(os.path.join(TR, nm + ".npz"))
    r = R[nm]
    u = (-1 if r["view"] != "front" else 1) * z["te_u"][:, r["station"]] / PX
    t = np.arange(len(u)) / FPS
    m = t < 4
    a.plot(t[m], u[m] - u[t < .15].mean(), color=col, lw=1.1, label=lab)
a.set(xlabel="Time in s", ylabel="Trailing edge shift in mm", title="1800 V, 1 Hz")
a.legend(fontsize=7.5)
a = ax[1]
a.plot(Vs, hi, "o", color=A_, label="Cycle high")
a.plot(Vs, lo, "s", color="0.5", label="Cycle low")
v = np.linspace(0, 1850, 100)
a.plot(v, (K / (T0 - c * (v / 1e3) ** 2) - K / T0) * 1e3, color=A_, ls="--", lw=1, label="Hanging string, $T_0$ %.1f mN" % (T0 * 1e3))
a.set(xlabel="Peak voltage in V", ylabel="Out-of-plane shift in mm", title="Side view, sine squared, 1 Hz", xlim=(0, 1900))
a.legend(fontsize=7.5)
a = ax[2]
for ten, col, mk, lab in ((False, P_, "o", "As set"), (True, A_, "s", "More tension")):
    q = sorted((r for n, r in R.items() if r["view"] == "diag" and r["tension"] == ten), key=lambda r: r["f"])
    a.loglog([r["f"] for r in q], [r["A1_mm"] for r in q], "-" + mk, color=col, ms=4, label=lab + ", swing")
    a.loglog([r["f"] for r in q], [r["mean_mm"] for r in q], ":" + mk, color=col, ms=3, mfc="none", label=lab + ", average")
a.axhline(.002, color="0.6", lw=.7, ls="--")
a.text(1.05, .0024, "Noise", fontsize=7)
a.set(xlabel="Drive frequency in Hz", ylabel="Amplitude in mm", title="Diagonal view, 1800 V, plain sine", ylim=(.001, 1.5))
a.legend(fontsize=7)
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_views.png")); fig.savefig(os.path.join(HERE, "V_views.svg"))


def enc(frames, name, fps):
    h, w_ = frames[0].shape[:2]
    pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w_}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart", os.path.join(HERE, "clips", name)], stdin=subprocess.PIPE)
    for f in frames:
        pr.stdin.write(np.ascontiguousarray(f).tobytes())
    pr.stdin.close(); pr.wait()


S = Stack(os.path.join(AV, "2layer_1800V_1hz_side.avi"))
enc([cv2.resize(np.asarray(S[i][296:424, 240:656]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) for i in range(0, 800, 4)], "side_1800V.mp4", 50)
S = Stack(os.path.join(AV, "2layer_1800V_15hz_diag_sine.avi"))
enc([np.asarray(S[i][260:440, 240:656]) for i in range(0, 401)], "diag_15Hz.mp4", 25)
print("clips", [(n, os.path.getsize(os.path.join(HERE, "clips", n)) // 1024) for n in os.listdir(os.path.join(HERE, "clips"))])
