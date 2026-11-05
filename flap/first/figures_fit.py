"""Model wrinkle lines on the video. Run: python figures_fit.py <stack folder> -> V_wrinkle.png"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "plagiopatagiales")]
import plotstyle as PS
import matplotlib.pyplot as plt
PS.apply()
D = sys.argv[1]
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
