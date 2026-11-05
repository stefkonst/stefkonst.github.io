"""Edge shift against voltage, assuming V = Vp sin^2(pi (t - t0)) at 1 Hz from 0 V, with t0 half a cycle before the fitted peak: first rise against later cycles. Run: python first_rise.py -> data/first_rise.json"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(HERE, "data", "tracks.json")))
t, u = np.array(T["mid_t"]), np.array(T["mid"])
rest = u[t < .2].mean()
q = t > 1.5
c = np.linalg.lstsq(np.c_[np.ones(q.sum()), np.cos(2 * np.pi * t[q]), np.sin(2 * np.pi * t[q])], u[q], rcond=None)[0]
t0 = ((np.arctan2(c[2], c[1]) / (2 * np.pi)) - .5) % 1
print("t0 %.3f s" % t0)
V = np.linspace(0, 1, 21)
th = np.arcsin(np.sqrt(V)) / np.pi
out = dict(t0=t0, rest=rest, V=V.tolist(), up={}, down={})
for k in range(9):
    out["up"][k] = (np.interp(t0 + k + th, t, u) - rest).tolist()
    out["down"][k] = (np.interp(t0 + k + 1 - th, t, u) - rest).tolist()
for k in (0, 1, 3, 6):
    print("cycle", k, "up", np.round(out["up"][k][::4], 3), "down", np.round(out["down"][k][::4], 3))
m = V >= .3
r = np.array(out["up"][0])
p0 = np.polyfit(np.log(V[m]), np.log(r[m]), 1)[0]
late_up = np.mean([out["up"][k] for k in range(2, 9)], 0)
late_dn = np.mean([out["down"][k] for k in range(2, 9)], 0)
s = late_up - late_up[0]
p1 = np.polyfit(np.log(V[m]), np.log(s[m]), 1)[0]
print("exponent, first rise %.2f, later rises %.2f" % (p0, p1))
print("late up", late_up[::4].round(3), "late down", late_dn[::4].round(3), "loop width at V=.5", late_dn[10] - late_up[10])
out.update(p_first=p0, p_late=p1, late_up=late_up.tolist(), late_down=late_dn.tolist())
print("first peak %.3f first low %.3f elastic part of first rise %.2f" % (r[-1], out["down"][0][0], (r[-1] - out["down"][0][0]) / r[-1]))
json.dump(out, open(os.path.join(HERE, "data", "first_rise.json"), "w"), indent=1, default=float)
