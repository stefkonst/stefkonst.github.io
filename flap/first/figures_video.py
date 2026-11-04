"""Figures from the video tracks. Run: python figures_video.py <stack folder> -> V_*.png + .svg"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
D = sys.argv[1]
T = json.load(open(os.path.join(HERE, "data", "tracks.json")))
G = json.load(open(os.path.join(HERE, "data", "geometry.json")))
z = np.load(os.path.join(D, "ref.npz"))
PX = T["px_per_mm"]
# outline on the frame
fig, ax = plt.subplots(1, 2, figsize=(11, 4.1))
a = ax[0]
ext = [0, 640 / PX, 480 / PX, 0]
a.imshow(z["A0"], cmap="gray", extent=ext, vmin=30, vmax=210)
ROOT, TIP = np.array([22., 410.]), np.array([458., 88.])
e = (TIP - ROOT) / np.linalg.norm(TIP - ROOT); n = np.array([e[1], -e[0]])
s = np.array(G["s_mm"])
for k, c, lab in (("before", PS.OFF, "Before voltage"), ("high", PS.ON, "Cycle high")):
    y = np.array(G[k]["sag_mm"])
    P = ROOT / PX + np.outer(s, e) + np.outer(y, n)
    a.plot(P[:, 0], P[:, 1], color=c, lw=.9, label=lab)
a.set(xlabel="Image $x$ in mm", ylabel="Image $y$ in mm", title="Outline on the frame, 8.24 px/mm")
a.legend(loc="lower right")
a = ax[1]
for k, c, lab in (("before", PS.OFF, "Before voltage"), ("low", PS.MATLAB[1], "Cycle low"), ("high", PS.ON, "Cycle high")):
    a.plot(s, G[k]["sag_mm"], color=c, label=lab)
a.set(xlabel=r"Along the knot-to-tip line, $s$ in mm", ylabel=r"Sag towards the leading edge, $\delta$ in mm", title="Trailing edge shape", xlim=(0, 66), ylim=(-.5, 2.5))
a.legend()
PS.finish(ax[1:], letters=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_shape.png"))
# time
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[1.7, 1, 1]))
a = ax[0]
a.plot(T["mid_t"], T["mid"], color=PS.MATLAB[0], lw=.7)
a.axhline(0, color="0.4", lw=.7)
a.set(xlabel="Time $t$ in s", ylabel=r"Mid trailing edge shift, $u$ in mm", title="10 s record", xlim=(0, 10), ylim=(-.05, .6))
a = ax[1]
a.plot(np.array(T["phase_avg_x"]) * 1e3, T["phase_avg"], "o-", color=PS.MATLAB[0], ms=2.5)
a.set(xlabel="Time from the cycle high in ms", ylabel=r"$u$ in mm", title="Averaged cycle, $t$ > 2 s", ylim=(0, .6))
a = ax[2]
sm = np.array(T["s_mm"])
a.plot(sm, T["u_high"], "o-", color=PS.ON, ms=2.5, label="Cycle high")
a.plot(sm, T["u_low"], "o-", color=PS.MATLAB[1], ms=2.5, label="Cycle low")
a.plot(sm, np.array(T["u_high"]) - np.array(T["u_low"]), "o-", color="0.4", ms=2.5, label="Swing")
a.set(xlabel="Along the root-to-tip chord in mm", ylabel=r"$u$ in mm", title="Along the edge", ylim=(0, .6))
a.legend()
PS.finish(ax)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "V_time.png"))
