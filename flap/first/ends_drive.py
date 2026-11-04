"""End points along the edge line, and what the cycle shape says about the drive. Run: python ends_drive.py <stack folder> -> data/ends_drive.json"""
import json, os, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1]
PX, FPS = 8.24, 200.0
B = np.load(os.path.join(D, "beads.npz"))
T = json.load(open(os.path.join(HERE, "data", "tracks.json")))
t = np.arange(B["tip"].shape[0]) / FPS
e = np.array([458 - 22, 88 - 410.]); e /= np.linalg.norm(e)
n = np.array([e[1], -e[0]])
tp = np.mean(np.array(T["t_max"])[1:] % 1)
hi = (np.abs(((t - tp + .5) % 1) - .5) < .06) & (t > 1.5)
lo = (np.abs(((t - tp) % 1) - .5) < .06) & (t > 1.5)
pre = t < .3
out = {}
for k in ("tip", "mark", "root"):
    v = B[k] / PX * 1e3
    out[k] = {a: dict(along=float((v[m] @ e).mean()), normal=float((v[m] @ n).mean())) for a, m in (("before", pre), ("low", lo), ("high", hi))}
    print(k, {a: (round(b["along"], 1), round(b["normal"], 1)) for a, b in out[k].items()})
# 1 Hz lock-in on each, um
for k in ("tip", "mark", "root"):
    v = B[k] / PX * 1e3
    q = t > 1.5
    for nm, d in (("along", e), ("normal", n)):
        y = v[q] @ d
        X = np.c_[np.ones(q.sum()), t[q] - 6, np.cos(2 * np.pi * (t[q] - tp)), np.sin(2 * np.pi * (t[q] - tp))]
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        out[k]["lock_" + nm] = [float(c[2]), float(c[3])]
        print(k, nm, "1 Hz cos %.2f sin %.2f um" % (c[2], c[3]))
# drive shape: u = u0 + a (1 + r cos(wt + p))^2
x = np.array(T["phase_avg_x"])
y = np.array(T["phase_avg"])
def res(p):
    return p[0] + p[1] * (1 + p[2] * np.cos(2 * np.pi * x + p[3])) ** 2 - y
f = least_squares(res, [0.3, 0.05, 0.5, 0.0], bounds=([-1, 0, 0, -1], [1, 5, 1, 1]))
r = f.x[2]
print("fit u0 %.3f a %.3f r %.2f p %.2f, rms %.4f; Vmin/Vmax %.2f" % (*f.x, np.std(f.fun), (1 - r) / (1 + r)))
fits = {}
for rr, lab in ((1.0, "0 to 1500 V"), (0.111, "1200 to 1500 V")):
    def res2(p):
        return p[0] + p[1] * (1 + rr * np.cos(2 * np.pi * x + p[2])) ** 2 - y
    g = least_squares(res2, [0.3, 0.05, 0.0])
    fits[lab] = dict(p=g.x.tolist(), rms=float(np.std(g.fun)), u=(y + g.fun).tolist())
    print(lab, g.x.round(3), "rms %.4f" % np.std(g.fun))
out["fit"] = dict(u0=f.x[0], a=f.x[1], r=r, p=f.x[3], rms=float(np.std(f.fun)), ratio=(1 - r) / (1 + r), u=(y + f.fun).tolist(), x=x.tolist(), y=y.tolist())
out["fits"] = fits
json.dump(out, open(os.path.join(HERE, "data", "ends_drive.json"), "w"), indent=1, default=float)
